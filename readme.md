## 利用Unet网络对cell进行分割

## 数据处理
```markdown
/data
    --imgs/
        --cell1.png
    --masks/
        --cell1_mask.png

- 将tif格式的原始图像转为jpg格式
- 将转后的jpg图像进行滑动截取, 截取后的图像大小为256x256
- 将截取后的灰度图和mask按照名称进行匹配,并放到对应的imgs和masks文件下
- 图像和mask保存要求
    1. 将imgs和masks保存在同一路径./data下
    2. imgs文件夹下保存原始的灰度图像(如图像名称为img1.png
    3. masks文件夹下保存每张原始图像对应的mask二值图像(如mask对应的名称为img1_mask.png)
```

## 模型构建
- UNet


- NestedUNet


## 数据增强
```markdown
train_transform = Compose([
        albu.RandomRotate90(),
        transforms.Flip(),
        transforms.RandomContrast(),
        albu.Resize(config['input_h'], config['input_w'])])
对输入图像进行随机旋转,翻转,随机裁切, 并同一缩放到指定尺寸.
```

## 模型训练
- UNet train
```bash
python train.py --arch UNet --batch-size 4 --dataset data --epochs 5
```
```markdown
arch: net structure
batch-size: batch size
dataset: file path of imgs and masks
epochs: training epoch
```

- NestedUNet
```bash
 python train.py --arch NestedUNet --deep_supervision True --batch-size 4 --dataset data --epochs 5
```

## 预测
- slice predict
```bash
python slicePredict.py --model_name UNet --model_weight checkpoints\UNet.pth --input D:\dataset\BGI_EXAM\test_set\172.jpg --output data/output/test_result_172.jpg
```
```markdown
model_name: Net structure name of model
model_weight: Specify the file in which the model is stored
input: image path of prediction
output: save path of predicted result
```


- dirct predict
```bash
python predict.py --model_name UNet --model_weight checkpoints\UNet.pth --input D:\dataset\BGI_EXAM\test_set\172.jpg --output data/output/test_result_172.jpg
```