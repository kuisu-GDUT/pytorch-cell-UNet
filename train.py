import argparse
import logging
import sys
from pathlib import Path
import os
from collections import OrderedDict
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

import unet
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import yaml

from albumentations.augmentations import transforms
import albumentations as albu
from albumentations.core.composition import Compose, OneOf

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss, iou_score
from utils.utils import AverageMeter
from evaluate import evaluate
from unet import UNet, NestedUNet

ARCH_NAMES = unet.unet_model.__all__
# LOSS_NAMES.append('BCEWithLogitsLoss')




def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    #0. 数据增强
    train_transform = Compose([
        albu.RandomRotate90(),
        transforms.Flip(),
        OneOf([
            # transforms.HueSaturationValue(),
            # transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),  # 按照归一化的概率选择执行哪一个
        albu.Resize(config['input_h'], config['input_w']),
        # transforms.Normalize(),
    ])

    # val_transform = Compose([
    #     albu.Resize(config['input_h'], config['input_w']),
    #     transforms.Normalize(),
    # ])

    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale,transforms=train_transform)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale,transforms=train_transform)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    params = filter(lambda p: p.requires_grad, net.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=learning_rate, weight_decay=1e-4)
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9,
                              nesterov=False, weight_decay=1e-4)
    elif config['optimizer'] == 'RMS':
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5)
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2,
                                                   verbose=1)# goal: maximize Dice score
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    #5. save log
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_dice', []),
        ('val_iou', []),
    ])

    #6. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_preds = net(images)
                    loss = 0
                    for masks_pred in masks_preds:
                        l_criterion = criterion(masks_pred, true_masks)
                        l_dice = dice_loss(F.softmax(masks_pred, dim=1).float(),
                                           F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True)
                        loss = l_dice + l_criterion
                    # loss /= len(masks_preds)
                    iou = iou_score(F.one_hot(masks_preds[-1].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float(),
                                    F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float())

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                avg_meters['loss'].update(loss.item(),images.size(0))
                avg_meters['iou'].update(iou, images.size(0))
                avg_meters['dice'].update(1-l_dice.item(), images.size(0))
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch,
                    'avgloss': avg_meters['loss'].avg,
                    'iou':avg_meters['iou'].avg
                })

                #
                postfix = OrderedDict([
                    ('avgloss', avg_meters['loss'].avg),
                    ('iou', avg_meters['iou'].avg),
                    ('dice', avg_meters['dice'].avg),
                ])
                pbar.set_postfix(postfix)

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            try:
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                            except:
                                continue

                        val_ave = evaluate(net, val_loader, device,
                                           deep_supervision=config['deep_supervision'])
                        scheduler.step(val_ave['dice'].avg)

                        logging.info('Validation Dice score: {}, IoU score: {}'.format(val_ave['dice'].avg.item(),
                                                                                       val_ave['iou'].avg))

                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_ave['dice'].avg.item(),
                            'validation iou':val_ave['iou'].avg,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

                        # save log to csv
                        log['epoch'].append(epoch)
                        log['lr'].append(config['lr'])
                        log['loss'].append(avg_meters['loss'].avg)
                        log['iou'].append(avg_meters['iou'].avg)
                        log['val_dice'].append(val_ave['dice'].avg.item())
                        log['val_iou'].append(val_ave['iou'].avg)

                        pd.DataFrame(log).to_csv('checkpoints/%s/log.csv' %
                                                 config['name'], index=False)

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint/config['name']/ 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: NestedUNet)')
    parser.add_argument('--dataset', default='data',
                        help='dataset name')
    parser.add_argument('--deep_supervision', default=True, type=bool)


    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')

    #training
    parser.add_argument('--optimizer', default='RMS',
                        choices=['Adam', 'SGD','RMS'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD','RMS']) +
                             ' (default: SGD)')
    parser.add_argument('--scheduler', default='ReduceLROnPlateau',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    #Net
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dir_img = Path(os.path.join(args.dataset,"imgs"))
    dir_mask = Path(os.path.join(args.dataset,"masks"))
    dir_checkpoint = Path('./checkpoints/')

    config = vars(args)
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('checkpoints/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)
    with open('checkpoints/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    Model = eval(config['arch'])
    net = Model(n_channels=3, n_classes=2,
                     bilinear=args.bilinear,
                     deep_supervision=config['deep_supervision'])

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
