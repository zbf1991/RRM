# Reliability Does Matter: An End-to-End Weakly Supervised Semantic Segmentation Approach
AAAI 2020 (Spotlight).

The code of the new extended work now is available. In the further, I will try to merge these two work so that making the whole project is more elegant.

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
 
 [RRM_final.pth] is the final model (AAAI).
 
 [RRM(attention)_final.pth] is the final model of the new extended work (64.7 mIoU on Pascal Voc 2012 val set).

## Training of New Extended Work:

### Training from init model:
you need 4 GPUs and the pretrained model [ilsvrc-cls_rna-a1_cls1000_ep-0001.params]:
```
python train_from_init(attention).py --voc12_root /your/path/VOCdevkit/VOC2012
```
 
### Training from a pretrained classification model:
you only need 2 GPU and the pretrained model [res38_cls.pth]
```
python train_from_cls_weight(attention).py --IM_path /your/path/VOCdevkit/VOC2012/JPEGImages
```

## Training of AAAI Work:
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
## Inferencing of the extended work:
you need 1 GPU and the final model [RRM(attention)_final.pth]:
```
python infer_RRM.py --IM_path /your/path/VOCdevkit/VOC2012/JPEGImages
```
## Inferencing of AAAI work:
you need 1 GPU and the final model [RRM_final.pth]:
```
python infer_RRM.py --IM_path /your/path/VOCdevkit/VOC2012/JPEGImages

