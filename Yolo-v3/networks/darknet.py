from __future__ import division

import torch
import numpy as np
import torch.nn as nn

from utils import parse_cfg
from networks.layers import create_modules




class Darknet(nn.Module):

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}  # cache outputs for route layer
        write = 0     # indicate whether we have met our first detection
        for i, module in enumerate(modules):
            module_type = (module['type'])
            # for conv and upsample layer, just forward through this module
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)

            # deal with route layer, this is where we actually route
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            # deal with shortcut(residual)
            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i-1] + outputs[i + from_]

            # deal with yolo layers
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info['height'])
                num_classes = int(module['classes'])

                x = x.data
                # cuda must be checked, otherwise you'll get a error calling func predict_transform()
                if CUDA:
                    x = x.cuda()
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        """
        loading weights from pretrained weightsfile, in this case ,we use
        coco 80 categories weightsfile
        :param weightfile: pretrained weightsfile
        :return: None, load weights to our model
        """
        fp = open(weightfile, 'rb')

        # The first 160 bytes of the weights file store 5 int32 values which
        # constitute the header of the file
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # store the rest weight file as a ndarray
        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]['type']

            # load weights when we meet conv layers since weights only exist at conv layer
            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                except:
                    batch_normalize = 0
                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    # get num of weights of bn layer
                    num_bn_biases = bn.bias.numel()
                    # load weights to torch tensor
                    bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # cast the loaded weights so that dims match model
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # copy to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr += num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)

                    conv.bias.data.copy_(conv_biases)

                # load weights for convolutional layers
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
