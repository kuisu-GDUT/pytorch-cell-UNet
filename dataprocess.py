#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-UNet 
@File    ：dataprocess.py
@Author  ：kuisu
@Email     ：kuisu_dgut@163.com
@Date    ：2022/3/23 18:44 
'''
#将原始图像,和二至分割图像数据进行匹配, 并保存到data文件中
import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

#将大图通过滑窗方式截取为小图
def sliceImage(image_path,output_file_name,output_dir):
    from sahi.slicing import slice_image
    slice_image_result, num_total_invalid_segmentation = slice_image(
        image=image_path,
        output_file_name=output_file_name,
        output_dir=output_dir,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=True
    )
    return slice_image_result,num_total_invalid_segmentation


def match_data(train_dir,mask_dir):
    #将mask和origin_img进行匹配
    count1,count2,count3,count4 = 0,0,0,0
    for image_name in tqdm(os.listdir(mask_dir)):
        if os.path.exists(os.path.join(train_dir, image_name)):
            mask = Image.open(os.path.join(mask_dir,image_name))
            image = Image.open(os.path.join(train_dir,image_name))
            mask = mask.copy().convert("L")
            mask = mask.copy().convert("1")
            mask = np.array(mask,dtype=np.uint8)*255
            if mask.max() == 255:
                mask_img = Image.fromarray(mask)
                mask_img.save(os.path.join("./data/masks/","{}_mask.png".format(image_name.split(".")[0])))
                image.save(os.path.join("./data/imgs","{}.png".format(image_name.split(".")[0])))
                if 'img1' in image_name:
                    count1+=1
                elif "img2" in image_name:
                    count2+=1
                elif "img3" in image_name:
                    count3+=1
                elif "img4" in image_name:
                    count4+=1
        else:
            print("{} not in train_dir".format(image_name))
    print("img1: {}, img2:{}, img3: {}, img4:{}".format(count1,count2,count3,count4))

# 将tif格式转为png
def tif2jpg(path,save_path):
    tif_list = [x for x in os.listdir(path) if x.endswith(".tif")]   # 获取目录中所有tif格式图像列表
    for num,i in enumerate(tif_list):      # 遍历列表
        tifPath = os.path.join(path,i)
        lbp_1_8_real = cv2.imread(tifPath,-1)
        if "mask" in i:
            lbp_1_8_real*=255
        #  读取列表中的tif图像
        #转换为0-255
        img_norm = np.zeros_like(lbp_1_8_real)
        real_show = cv2.normalize(lbp_1_8_real, dst=img_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        save_image_path = os.path.join(save_path,i.split('.')[0]+".jpg")
        cv2.imwrite(save_image_path,real_show)    # tif 格式转 jpg 并按原名称命名, 保存为灰度图

if __name__ == '__main__':

    ## 1. 将tif转为png
    # path = r"D:\dataset\BGI_EXAM\train_set"  # 获取代码所在目录
    # save_path = r"D:\dataset\BGI_EXAM\train_set"
    # tif2jpg(path,save_path)


    # # 2. 滑窗截取小图,并过滤背景为空的mask
    # image_path = r"D:\dataset\BGI_EXAM\test_set\171.jpg"
    # output_file_name = "171"
    # output_dir = r"D:\dataset\BGI_EXAM\slices\test"
    # sliceImage(image_path, output_file_name, output_dir)

    # #3. 将截取mask和origin_image进行匹配
    # train_dir = r"D:\dataset\BGI_EXAM\slices\train2"
    # mask_dir = r"D:\dataset\BGI_EXAM\slices\mask2"
    # match_data(train_dir,mask_dir)
    pass
