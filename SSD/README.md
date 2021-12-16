# SSD Implementation
Implememtation of Single Shot Detector (SSD) architecture introduced by [Liu et al.](https://arxiv.org/abs/1512.02325). 

<p float="center">
  <img src="SSD_architecture.jpg" width="520" />
</p>

## Object Detection notions

* **Single-Shot Detection**. Earlier architectures for object detection consisted of two distinct stages – a region proposal network that performs object localization and a classifier for detecting the types of objects in the proposed regions (e.g. [Faster-RCNN](https://arxiv.org/abs/1506.01497)). Computationally, these can be very expensive and therefore ill-suited for real-world, real-time applications. Single-shot models encapsulate both localization and detection tasks in a single forward sweep of the network, resulting in significantly faster detections while deployable on lighter hardware.

* **Multiscale Feature Maps**. In image classification tasks, we base our predictions on the final convolutional feature map – the smallest but deepest representation of the original image. In object detection, feature maps from intermediate convolutional layers can also be _directly_ useful because they represent the original image at different scales. Therefore, a fixed-size filter operating on different feature maps will be able to detect objects of various sizes.

* **Priors or default boxes**. These are pre-computed boxes defined at specific positions on specific feature maps, with specific aspect ratios and scales. They are carefully chosen to match the characteristics of objects' bounding boxes (i.e. the ground truths) in the dataset.

* **Multibox**. This is [a technique](https://arxiv.org/abs/1312.2249) that formulates predicting an object's bounding box as a _regression_ problem, wherein a detected object's coordinates are regressed to its ground truth's coordinates. In addition, for each predicted box, scores are generated for various object types. Priors serve as feasible starting points for predictions because they are modeled on the ground truths. Therefore, there will be as many predicted boxes as there are priors, most of whom will contain no object.

* **Hard Negative Mining**. This refers to explicitly choosing the most egregious false positives predicted by a model and forcing it to learn from these examples. In other words, we are mining only those negatives that the model found _hardest_ to identify correctly. In the context of object detection, where the vast majority of predicted boxes do not contain an object, this also serves to reduce the negative-positive imbalance.

* **Non-Maximum Suppression**. At any given location, multiple priors can overlap significantly. Therefore, predictions arising out of these priors could actually be duplicates of the same object. Non-Maximum Suppression (NMS) is a means to remove redundant predictions by suppressing all but the one with the maximum score.

# Single Shor Detector (SSD)
The SSD is a purely convolutional neural network (CNN) that we can organize into three parts:

* **Base convolutions** derived from an existing image classification architecture that will provide lower-level feature maps.

* **Auxiliary convolutions** added on top of the base network that will provide higher-level feature maps.

* **Prediction convolutions** that will locate and identify objects in these feature maps.


## Prepare dataset
We will use Pascal Visual Object Classes (VOC) data from the years 2007 and 2012.
Basically, you will need to download the VOC datasets for training and testing phases.

After that, you should create a folder "datasets" and extract the dataset there.

Before training, you need to prepare the dataset into .json files. Indeed, run the following command:
```
python3 data/create_data_lists.py --dataset_root [PATH]
```

## Training
First you have to set your configuration options:

|argument          |type|description|default|
|:-----------------|:----|:--------------------------------------------- |:--------------|
|--data_folder     |str  |dataset directory path                         |./datasets/    |
|--keep_difficult  |bool |use objects considered difficult to detect     |True           |
|--checkpoint      |str  |path to model checkpoint                       |None           |
|--batch_size      |int  |batch size                                     |16             |
|--num_workers     |int  |number of workers used in dataloading          |448            |
|--start_epoch     |int  |Resume training at this epoch                  |0              |
|--nb_iters        |int  |number of iterations to train                  |12000          |
|--lr              |float|learning rate                                  |1e-4           |
|--momentum        |float|Momentum value for optimizer                   |0.9            |
|--weight_decay    |float|Weight decay for optimizer                     |5e-4           |
|--decay_lr_at     |list |decay learning rate after these many iterations|[80000, 100000]|
|--decay_lr_to     |float|decay lr to this fraction of the existing lr   |0.1            |
|--grad_clip       |float|clip if gradients are exploding                |None           |
|--cuda            |bool |Use CUDA to train model                        |True           |
|--step_display    |int  |print training status every __ batches         |200            |



And then, put the following command into your terminal:
```
python3 main.py [--data_folder DATA_FOLDER] [--keep_difficult KEEP_DIFFICULT]
                [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--start_epoch START_EPOCH]
                [--nb_iters NB_ITERS] [--cuda CUDA] [--step_display STEP_DISPLAY] 
                [--lr LR] [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY] 
                [--decay_lr_at DECAY_LR_AT] [--decay_lr_to DECAY_LR_TO]
                [--grad_clip GRAD_CLIP]
                [--checkpoint CHECKPOINT] 

```

## Evaluation

The evaluation metric is the Mean Average Precision (mAP) (for more details, see [here](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)).
To evaluate your model, specify the weights path and run the following command:
```
python3 eval.py
```

## Testing

You need first to install font type for testing, For example: the Arial type as follows:
```
wget http://downloads.sourceforge.net/corefonts/arial32.exe
```

In short, for testing your model, specify the weights path and then run the test.py file.

## Acknowledgment
* Object detection tutorial by [sgrvinod](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection).
