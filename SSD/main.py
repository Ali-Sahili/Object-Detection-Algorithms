import warnings
warnings.filterwarnings('ignore')

import torch
import argparse

from train import train
from networks.model import SSD300
from multibox_loss import MultiBoxLoss
from data.datasets import PascalVOCDataset
from utils.helpers import label_map, save_checkpoint, adjust_learning_rate, str2bool


parser = argparse.ArgumentParser(description='SSD Training With Pytorch')
# Dataset parameters
parser.add_argument('--data_folder', default='./datasets/', help='Dataset directory')
parser.add_argument('--checkpoint', default=None, help='path to model checkpoint')
parser.add_argument('--keep_difficult', default=True, type=str2bool, 
                     help='use objects considered difficult to detect.')
# Dataloading
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='nb_workers used in dataloading')
parser.add_argument('--start_epoch', default=0, type=int, help='Resume training at this epoch')
parser.add_argument('--nb_iters', default=120000, type=int, help='number of iterations to train')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
parser.add_argument('--step_display', default=200, type=int, 
                    help='print training status every __ batches')
# Optimizer parameters
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--decay_lr_at', default=[80000, 100000], type=list, 
                    help='decay learning rate after these many iterations')
parser.add_argument('--decay_lr_to', default=0.1, type=float, 
                    help='decay learning rate to this fraction of the existing learning rate')
parser.add_argument('--grad_clip', default=None, type=float, 
                    help='clip if gradients are exploding (sometimes at batch size of 32)')


# Main function
def main(args):

    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    n_classes = len(label_map)  # number of different types of objects

    # Initialize model or load checkpoint
    if args.checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        
        # Initialize the optimizer, with twice the default learning rate for biases
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * args.lr}, 
                                            {'params': not_biases}],
                                    lr=args.lr, momentum=args.momentum, 
                                    weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, device=device).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(args.data_folder, split='train',
                                     keep_difficult=args.keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                               shuffle=True, num_workers=args.num_workers,
                                               collate_fn=train_dataset.collate_fn, 
                                               pin_memory=True)
    print("Number of batches: ", len(train_loader))

    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 
    # 100,000 iterations
    num_epochs = args.nb_iters // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in args.decay_lr_at]
    
    print("Start training ...")
    for epoch in range(start_epoch, num_epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, args.decay_lr_to)

        # One epoch's training
        train(args, train_loader, model, criterion, optimizer, epoch, device)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)
        assert(0)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
