import argparse
from test import test
from train import train
from utils.helpers import str2bool


# Parameters settings
parser = argparse.ArgumentParser(description='YOLO-v1 implementation on Pytorch')
# Dataset
parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--data_path', type=str,help='path to the data', 
                                            default="../datasets/VOC2007_train/VOCdevkit/VOC2007")
parser.add_argument('--checkpoint_path', type=str, help='path to a checkpoint', default='./')
parser.add_argument('--input_height', type=int, help='input height', default=448)
parser.add_argument('--input_width', type=int, help='input width', default=448)
# training
parser.add_argument('--batch_size', type=int, help='batch size', default=16)
parser.add_argument('--num_epochs', type=int, help='number of epochs', default=16000)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=1e-3)
parser.add_argument('--dropout', type=float, help='dropout probability', default=0.5)
parser.add_argument('--cuda', type=str2bool, help='use CUDA for training', default=True)
parser.add_argument('--step_display', default=200, type=int, 
                    help='print training status every __ batches')
# flags
parser.add_argument('--use_augmentation', type=str2bool, help='Image Augmentation', default=True)
parser.add_argument('--use_visdom', type=str2bool, help='visdom board', default=False)
parser.add_argument('--use_summary', type=str2bool, help='descripte Model summary', default=True)
parser.add_argument('--check_gt', type=str2bool, help='Ground Truth check flag', default=False)


def main(args):

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
