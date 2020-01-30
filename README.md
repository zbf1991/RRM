# Reliability Does Matter: An End-to-End Weakly Supervised Semantic Segmentation Approach
AAAI 2020 (Spotlight).

Due to the coronavirus outbreak in China, I cannot return to my lab, this project is uploaded throuthing the remote desktop.
I will rewrite this README file and answer issuses after I can go back.

This project is based on [Regularized loss](https://github.com/meng-tang/rloss) and [PSA](https://github.com/jiwoon-ahn/psa).

## Before Running, build python extension module:
```
cd wrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
python setup.py install
```
More details please see [here](https://github.com/meng-tang/rloss/tree/master/pytorch)

## Download pretrained models to ./netWeights:
Google: due to the coronavirus outbreak in China, I will upload models after I can enter my lab. But you can
download “[ilsvrc-cls_rna-a1_cls1000_ep-0001.params]” and “[res38_cls.pth]” from [here](https://github.com/jiwoon-ahn/psa).

[BaiduYun](https://pan.baidu.com/s/15AwO6Jn9vQQtThE02QOefw)

 [ilsvrc-cls_rna-a1_cls1000_ep-0001.params] is an init pretained model.
 
 [res38_cls.pth] is a classification model pretrained on VOC 2012 dataset.
 
 [RRM_final.pth] is my final model. mIoU is about 63.7 on val set, which is a higher score than our paper (62.6)

## Training:
I suggest that it is better to use the 2nd method due to lower computing costs.
### Training from init model:
you need 4 GPUs and the pretrained model [ilsvrc-cls_rna-a1_cls1000_ep-0001.params]:
```
python train_from_init.py --voc12_root /your/path/VOCdevkit/VOC2012
```
 
### Training from a pretrained classification model:
you only need 1 GPU and the pretrained model [res38_cls.pth]
```
python train_from_cls_weight.py --IM_path /your/path/VOCdevkit/VOC2012/JPEGImages
```
## Inferencing:
you need 1 GPU and the final model [RRM_final.pth]:
```
python infer_RRM.py --IM_path /your/path/VOCdevkit/VOC2012/JPEGImages
```
