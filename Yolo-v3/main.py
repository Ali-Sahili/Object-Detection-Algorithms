
import argparse
from test import test




def arg_parse():
    """
    parse arguments to the detect module
    :return: parsed args
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument('--imgs_path', default='imgs', type=str,
                        help= 'Directory contains imgs to perform detection')
    parser.add_argument('--save_folder', help= 'Image / Directory to store detections to',
                        default='results', type=str)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=1)
    parser.add_argument('--confidence', type=float, help='Obj Conf to filter preds', default=.5)
    parser.add_argument('--nms_thresh', type=float, help='NMS threshold', default=0.4)
    parser.add_argument('--cfg', help='Config file', default='cfg/yolov3.cfg', type=str)
    parser.add_argument('--weights', help='weights file', default=None, type=str)
    parser.add_argument('--im_size', help='Input size of the network.', default=416, type=str)
    
    return parser.parse_args()



args = arg_parse()
test(args)
