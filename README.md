# Learn to Threshold: ThresholdNet with Confidence-Guided Manifold Mixup for Polyp Segmentation

by [Xiaoqing Guo](https://guo-xiaoqing.github.io/).

## Summary:

### Intoduction:
This repository is for our IEEE TMI 2020 paper ["Learn to Threshold: ThresholdNet with Confidence-Guided Manifold Mixup for Polyp Segmentation"](https://ieeexplore.ieee.org/document/9305717)

### Framework:
![](https://github.com/Guo-Xiaoqing/ThresholdNet/raw/master/Figs/network.png)

## Usage:
### Requirement:
Pytorch 1.3
Python 3.6

### Preprocessing:
Clone the repository:
```
git clone https://github.com/Guo-Xiaoqing/ThresholdNet.git
cd ThresholdNet 
bash Threshold.sh
```

### Data preparation:
Dataset should be put into the folder './data'. For example, if the name of dataset is CVC, then the path of dataset should be './data/CVC/', and the folder structure is as following.
```
ThresholdNet
|-data
|--CVC
|---images
|---labels
|---train.txt
|---test.txt
|---valid.txt
```
The content of 'train.txt', 'test.txt' and 'valid.txt' should be just like:
```
26.png
27.png
28.png
...
```

### Pretrained model:
You should download the pretrained model from [Google Drive](https://drive.google.com/file/d/1yeZxwV6dYHQJmj2i5x9PnB6u-rqvlkCj/view?usp=sharing), and then put it in the './model' folder for initialization. 

### Well trained model:
You could download the trained model from [Google Drive](https://drive.google.com/file/d/1JURhma-F5c6SVBoBoGwFYh6QAaVzy_-W/view?usp=sharing), which achieves 87.307% in Dice score on the [EndoScene testing dataset](https://www.hindawi.com/journals/jhe/2017/4037190/). Put the model in directory './models'.

### Baseline model:
We also provide some codes of baseline methods, including polyp segmentation models and mixup related data augmentation baselines.

### Results:
Log files are listed in [log.out](https://github.com/Guo-Xiaoqing/ThresholdNet/raw/master/log.out) and [log1.out](https://github.com/Guo-Xiaoqing/ThresholdNet/raw/master/log1.out), which loss and accuracy of a mini-batch during training phase.

## Citation:
```
@article{guo2020learn,
  title={Learn to Threshold: ThresholdNet with Confidence-Guided Manifold Mixup for Polyp Segmentation},
  author={Guo, Xiaoqing and Yang, Chen and Liu, Yajie, Yuan, Yixuan},
  journal={IEEE Transactions on Medical Imaging},
  year={2020},
  publisher={IEEE}
}
```

## Questions:
Please contact "xiaoqingguo1128@gmail.com" 
