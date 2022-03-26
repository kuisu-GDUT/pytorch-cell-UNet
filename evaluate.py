import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import AverageMeter

from utils.dice_score import multiclass_dice_coeff, dice_coeff,iou_score


def evaluate(net, dataloader, device,deep_supervision=False):
    avg_meters = {'dice': AverageMeter(),
                  'iou': AverageMeter()}
    net.eval()
    num_val_batches = len(dataloader)
    # dice_score = 0
    # iou =0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_preds = net(image)
            mask_pred = mask_preds[-1]
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score = dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score = multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            iou = iou_score(mask_pred, mask_true)
            avg_meters['dice'].update(dice_score,image.size(0))
            avg_meters['iou'].update(iou,image.size(0))

           

    net.train()

    # Fixes a potential division by zero error
    # if num_val_batches == 0:
    #     return dice_score,iou
    return avg_meters
