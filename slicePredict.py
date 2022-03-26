#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：sahi 
@File    ：slicePredict.py
@Author  ：kuisu
@Email     ：kuisu_dgut@163.com
@Date    ：2022/3/23 22:13 
'''
import argparse
import numpy as np

from sahi.predict import get_sliced_prediction, get_prediction
from sahi.model import DetectionModelUNet
from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def readImg(filename=r"D:\dataset\BGI_EXAM\test_set\172.jpg"):
    big_image = Image.open(filename)
    big_image = big_image.convert("RGB")
    return big_image

def cut_image(image,size):
    """
    split image
    :param image: PIL image
    :param size: (int) one image be splited size x size iamges
    :return: (list) size x size images
    """
    width, height = image.size
    count_w,count_h = width/size,height/size
    count = int(min(count_h,count_w))
    # fix no image when size > width|height of input image
    if count ==0:
        count =1
        size = min(width,height)
    print("image cut to {}*{}".format(count,count))
    item_width = size
    item_height = size
    image_list = []
    # (left, upper, right, lower)
    for i in range(0,count):
        column = []
        for j in range(0,count):
            box = (j*item_width,i*item_height,(j+1)*item_width,(i+1)*item_height)
            column.append(box)
        row_image = [image.crop(box) for box in column]
        image_list.append(row_image)
    return image_list

# 按行拼接图片
def merge_image(image_list):
    """
    merge images
    :param image_list: (list) size x size images be merge one big image
    :return: one big image
    """
    target_width,target_height = 0,0
    for column_image in image_list[0]:
        target_width += column_image.size[0]
    for row_image in image_list:
        target_height += row_image[0].size[1]
    target = Image.new("RGB",(target_width,target_height))
    for i, row_images in enumerate(image_list):
        for j,column_images in enumerate(row_images):
            image = column_images
            item_width,item_height = image.size
            target.paste(image,(j*item_width,i*item_height,(j+1)*item_width,(i+1)*item_height))
    return target

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model_name', '-name', default='NestedUNet',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--model_weight', '-w', default='Model.pth',
                        metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT',  help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', help='Filenames of output images')
    parser.add_argument('--deep_supervision', '-d', default=True, type=bool,
                        help='deep_supervision when NestedUNet')
    parser.add_argument('--cut_size', default=1024,type=int,
                        help='size of cut image')
    parser.add_argument('--slice_size', default=256,type=int,
                        help='size of slice image')
    parser.add_argument('--over_ratio', default=0.0, type=float,
                        help='over ration when slice image')

    return parser.parse_args()


def main(filename,save_path):
    # init UNet model
    detection_model = DetectionModelUNet(model_name=config['model_name'],
                                         model_path=config['model_weight'],
                                         category_mapping={"0": "0", "1": "1"})
    # get image and split image
    big_image = readImg(filename)
    image_list = cut_image(big_image, size=config['cut_size'])

    target_list = []
    for row_images in image_list:
        target_column = []
        for image in row_images:
            # start sliced and predict mask
            imag_size = max(image.size)
            if imag_size>config['slice_size']:
                result = get_sliced_prediction(
                    image,
                    detection_model,
                    slice_height=config['slice_size'],
                    slice_width=config['slice_size'],
                    overlap_height_ratio=config['over_ratio'],
                    overlap_width_ratio=config['over_ratio']
                )
            else:
                result = get_prediction(
                    image,
                    detection_model,
                )
            # result of predict mask
            try:
                mask_result = result.object_prediction_list[1].mask.bool_mask
            except:
                w, h = image.size
                mask_result = np.zeros((h, w))
            mask_result = Image.fromarray(mask_result)
            target_column.append(mask_result)
        target_list.append(target_column)
    # merge all images of predict mask
    target_result = merge_image(target_list)
    # save result of predictions
    target_result.save(save_path, quality=100)

if __name__ == '__main__':
    args = get_args()

    # print parameters
    config = vars(args)
    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    # start predict
    main(filename=config['input'],save_path=config['output'])