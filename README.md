# SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning

This repository contains the code for SimpleShot introduced in the following paper

[SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning](https://arxiv.org/abs/1911.04623)

by [Yan Wang](https://www.cs.cornell.edu/~yanwang/), [Wei-Lun Chao](http://www-scf.usc.edu/~weilunc/), [Kilian Q. Weinberger](http://kilian.cs.cornell.edu/), [Laurens van der Maaten
](https://lvdmaaten.github.io/)

## Citation
If you find Simple Shot useful in your research, please consider citing:
```angular2
@article{wang2019simpleshot,
  title={SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning},
  author={Wang, Yan and Chao, Wei-Lun and Weinberger, Kilian Q.  and van der Maaten, Laurens},
  journal={arXiv preprint arXiv:1911.04623},
  year={2019}
}
```

## Introduction
Few-shot learners aim to recognize new object classes 
based on a small number of labeled training examples. 
To prevent overfitting, state-of-the-art few-shot learners 
use meta-learning on convolutional-network features and perform
classification using a nearest-neighbor classifier. This paper
studies the accuracy of nearest-neighbor baselines without meta-learning. 
Surprisingly, we find simple feature transformations suffice to obtain
competitive few-shot learning accuracies. For example, we find that
a nearest-neighbor classifier used in combination with mean-subtraction
and L2-normalization outperforms prior results in three out of five settings
on the miniImageNet dataset.

## Update:
* Dec. 17th. Add configuration and pretrained models of prototypical network with conv4 backbone.

## Usage
### 1. Dependencies
- Python 3.5+
- Pytorch 1.0+

### 2. Download Datasets
### 2.1 Mini-ImageNet
You can download the dataset from https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE

### 2.2 Tiered-ImageNet
You can download the dataset from https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view.
After downloading and unziping this dataset, you have to run the follow script to generate split files.
```angular2
python src/utils/tieredImagenet.py --data path-to-tiered --split split/tiered/
```
### 2.3 iNat2017
Please follow the instruction from https://github.com/daviswer/fewshotlocal to download the dataset.
And run the following script to generate split files.
```angular2
python ./src/inatural_split.py --data path-to-inat/setup --split ./split/inatural/
```

### 3 Train and Test
You can manually download the pretrained models. 
Then copy all the files to the corresponding folder.

Google Drives: https://drive.google.com/open?id=14ZCz3l11ehCl8_E1P0YSbF__PK4SwcBZ

BaiduYun: https://pan.baidu.com/s/1tC2IU1JBL5vPNmnxXMu2sA  code:d3j5


Or, you can run the follwing command to download them:
```angular2
cd ./src
python download_models.py
```
This repo includes `Resnet-10/18/34/50`, `Densenet-121`, `Conv-4`, `WRN`, `MobileNet` models.

For instance, to train a Conv-4 on Mini-ImageNet or Tiered-ImageNet,  
```angular2
python ./src/train.py -c ./configs/mini/softmax/conv4.config --data path-to-mini-imagenet/
```
```angular2
python ./src/train.py -c ./configs/tiered/softmax/conv4.config --data path-to-tiered-imagenet/data/
```
To evaluate the models on Mini/Tiered-ImageNet
```angular2
python ./src/train.py -c ./configs/mini/softmax/conv4.config --evaluate --enlarge --data path-to-mini-imagenet/
```
```angular2
python ./src/train.py -c ./configs/tiered/softmax/conv4.config --evaluate --enlarge --data  path-to-tiered-imagenet/data/
```
To evaluate INat models,
```angular2
python ./src/test_inatural.py -c ./configs/inatural/softmax/conv4.config --evaluate --enlarge --data path-to-inatural/setup/
```
New:
To train and evaluate the Conv-4 model on Mini-ImageNet with prototypical training:
```angular2
python ./src/train.py -c ./configs/mini/protonet/{conv4_shot1 | conv4_shot5}.config --data path-to-mini-imagenet/
python ./src/train.py -c ./configs/mini/protonet/{conv4_shot1 | conv4_shot5}.config --data path-to-mini-imagenet/ --evaluate
```


## Contact
If you have any question, please feel free to email us.

Yan Wang (yw763@cornell.edu)

