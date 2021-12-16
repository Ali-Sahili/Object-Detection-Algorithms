# YOLO Implementation
This part includes the Pytorch implementation of YOLO-v2 architecture for object detection publised by [Redmon et Farhadi](https://arxiv.org/abs/1612.08242).


## Introduction
YOLO stands for You Only Look Once. As the name says, network only looks the image once to detect multiple objects. The main improvement on this paper is the detection speed (~45 fps).
YOLO is different from all these methods as it treats the problem of image detection as a regression problem rather than a classification problem and supports a single convolutional neural network to perform all the above mentioned tasks.
There are 5 versions of YOLO namely version 1, version 2, version 3, version 4 and version 5. This part covers the second version.


## Requirements
The experiments were performed using Python 3.8.5 with the following Python packages:
- [numpy](http://www.numpy.org/)
- [torch](https://pytorch.org/)
- [torchvision](https://pypi.org/project/torchvision/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [pillow](https://pypi.org/project/Pillow/)
- [imgaug](https://pypi.org/project/imgaug/)

## Organization
This repository was organized as follows:
```
.
├─ data/
│  ├─ data_augmentation.py   <- For data augmentation (tranformations)
│  └─ voc_dataset.py         <- Read and load VOC dataset
├─ test/                    
│  ├─ pallate               
│  ├─ test.py             
│  └─ main.py
├─ loss.py                   <- defining the loss function
├─ model.py                  <- main model architecture
├─ utils.py                  <- utility functions
├─ train.py                  <- Train function
├─ main.py                   <- main file to train the whole model on VOC dataset
├─ yolov1_architecture.png
└─ README.md
```

## Usage
To begin with, you have to specify the different parameters needed for the training.
Next step is to download your dataset and change the path directory. And then enjoy your training by running the main.py file.


# Acknowledgment
[uvipen](https://github.com/uvipen/Yolo-v2-pytorch).
