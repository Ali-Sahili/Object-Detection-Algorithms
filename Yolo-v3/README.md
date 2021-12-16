# YOLO Implementation
This part includes the Pytorch implementation of YOLO-v1 architecture for object detection publised by [Redmon et Farhadi](https://arxiv.org/abs/1804.02767).

![](yolov3_architecture.png)

## Introduction
YOLO-v3 (You only look once) is one of the most popular deep learning models extensively used for object detection, semantic segmentation, and image classification.



## Requirements
The experiments were performed using Python 3.8.5 with the following Python packages:
- [numpy](http://www.numpy.org/)
- [torch](https://pytorch.org/)
- [torchvision](https://pypi.org/project/torchvision/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [pillow](https://pypi.org/project/Pillow/)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [pandas](https://pypi.org/project/pandas/)

## Organization
This repository was organized as follows:
```
.
├─ networks/
│  ├─ layers.py              <- For data augmentation (tranformations)
│  └─ darknet.py             <- Read and load VOC dataset
├─ pallete           
├─ yolov3.cfg                <- configuration file
├─ utils.py                  <- Utility functions
├─ test .py                  <- Test function
├─ main.py                   <- main file to test the model on VOC dataset
├─ yolov3_architecture.png
└─ README.md
```




## Training And Testing
To begin with, you have to specify the different parameters needed for the training.
Next step is to download your dataset and change the path directory. And then enjoy your training by running the main.py file.


# Acknowledgment
Thanks to the implementation done by [3epochs](https://github.com/3epochs/you-only-look-once).
