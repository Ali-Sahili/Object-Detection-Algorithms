
import torch.nn as nn




class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    """
    :param blocks:  blocks returned by func parser_cfg()
    :return: module list and network information described by blocks
    """
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for idx, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # conv module
        if x['type'] == 'convolutional':
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(idx), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(idx), bn)

            if activation == 'leaky':
                activ = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(idx), activ)

        # upsample module
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=stride, mode='nearest')
            module.add_module('upsample_{0}'.format(idx), upsample)

        # route module, cache this position(index)
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end = 0

            if start > 0:
                start = start - idx
            if end > 0:
                end = end - idx
            route = EmptyLayer()
            module.add_module('route_{0}'.format(idx), route)
            if end < 0:
                filters = output_filters[idx + start] + output_filters[idx + end]
            else:
                filters = output_filters[idx + start]

        # shortcut module, like ResNet
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{0}'.format(idx), shortcut)

        # yolo module, detection module
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('detection_{0}'.format(idx), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return net_info, module_list
