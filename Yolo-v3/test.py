from __future__ import division

import os
import cv2
import time
import torch
import random
import numpy as np
import pandas as pd
import pickle as pkl
import os.path as osp
from torch.autograd import Variable
from networks.darknet import Darknet
from utils import write_results, prep_image


def test_input(filename="dog-cycle-car.png"):
    img = cv2.imread(filename)
    img = cv2.resize(img, (416, 416))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    model = Darknet("cfg/yolov3.cfg")
    model.load_weights('yolov3.weights')
    
    pred = model(inp, torch.cuda.is_available())
    print(pred)
    print(pred.size())


def test(args):

    CUDA = torch.cuda.is_available()

    print("Loading network......")
    model = Darknet(args.cfgfile)
    if args.weightsfile is not None: model.load_weights(args.weightsfile)
    print('Network successfully loaded')

    model.net_info['height'] = args.im_size
    inp_dim = int(model.net_info['height'])
    
    assert inp_dim % 32 == 0
    assert inp_dim > 32
    
    if CUDA:
        model.cuda()
    model.eval()
    
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    
    # Load images
    load_batch = time.time()
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(args.imgs_path)]
    
    loaded_ims = [cv2.imread(x) for x in imlist]
    
    im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    leftover = 0
    if len(im_dim_list) % args.batch_size:
        leftover = 1

    if args.batch_size != 1:
        num_batches = len(imlist) // args.batch_size + leftover
        im_batches = [torch.cat((im_batches[i * args.batch_size: min((i+1)*args.batch_size, 
                                            len(im_batches))])) for i in range(num_batches)]

    # List of classes of VOC dataset
    class_list = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 
                  'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
                  'horse', 'motorbike', 'person', 'pottedplant', 
                  'sheep', 'sofa', 'train', 'tvmonitor')
    num_classes = len(class_list)  # VOC dataset num classes



    write = 0
    start_det_loop = time.time()
    for i, batch in enumerate(im_batches):
        start = time.time()
        if CUDA:
            batch = batch.cuda()
            
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)
        prediction = write_results(prediction, args.confidence, num_classes,
                                               nms_conf=args.nms_thresh)
        end = time.time()

        if type(prediction) == int:
            for im_num, image in enumerate(imlist[i*args.batch_size: min((i+1)*args.batch_size, 
                                                  len(imlist))]):
                im_id = i * args.batch_size + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], 
                                                                (end - start) / args.batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", ""))
                print("----------------------------------------------------------")
            continue

        prediction[:, 0] += i * args.batch_size
        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(imlist[i * args.batch_size: min((i+1)*args.batch_size, 
                                                                               len(imlist))]):
            im_id = i * args.batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], 
                                                                (end - start)/args.batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")
        if CUDA:
            torch.cuda.synchronize()

    try:
        output
    except NameError:
        print('No detections were made')
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
    scaling_factor = torch.min(inp_dim/im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor*im_dim_list[:, 0].view(-1, 1))/2
    output[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim_list[:, 1].view(-1, 1))/2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    output_recast = time.time()
    class_load = time.time()
    colors = pkl.load(open('pallete', 'rb'))

    draw = time.time()

    def write(x, results):
        """
        add bounding box to img
        :param x: img
        :param results: output of our model
        :return: img with bounding box
        """
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = '{}'.format(classes[cls])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                     cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img


    list(map(lambda x: write(x, loaded_ims), output))
    det_names = pd.Series(imlist).apply(lambda x: '{}/det_{}'.format(args.det, x.split('/')[-1]))

    list(map(cv2.imwrite, det_names, loaded_ims))
    end = time.time()

    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", 
                                                         output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
    print("----------------------------------------------------------")

    torch.cuda.empty_cache()
