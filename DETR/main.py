import time
import torch
import random
import argparse
import datetime
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from models import build_model
from train import train_one_epoch
from datasets import build_dataset
from utils import get_rank, collate_fn



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    
    # Training parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='grad clip max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                 help="Path to the pretrained model. If set, only the mask head will be trained")
    # Backbone parameters
    parser.add_argument('--backbone', default='resnet50', type=str, help="the backbone to use")
    parser.add_argument('--dilation', action='store_true',
                         help="Replace stride with dilation in the last conv block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, 
                        choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer parameters
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                   help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='nb of distributed processes')
    parser.add_argument('--dist_url', default='env://', 
                         help='url used to set up distributed training')
    return parser


def main(args):

    # Set device
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Defining the model
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # Compute the number of trainable parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: ', n_parameters)
    
    # model parameters
    param_dicts = [
 {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
 {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
  "lr": args.lr_backbone }
                  ]
    
    # Defining optimizer
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    # Defining scheduling strategy
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Load and prepare datasets
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    
    # Sampler for dataloading
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, 
                                                                       drop_last=True)
    # Dataloading
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn, 
                                 num_workers=args.num_workers)
    print("Number of batches in trainset: ", len(data_loader_train))
    print("Number of batches in valset: ", len(data_loader_val))
    
    output_dir = Path(args.output_dir)

    # Load weights: pretrained model
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model.detr.load_state_dict(checkpoint['model'])

    # Training Phase
    start_time = time.time()
    print("Start training...")
    for epoch in range(args.start_epoch, args.epochs):

        loss_unscaled, loss_scaled = train_one_epoch(model, criterion, data_loader_train, 
                                                   optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        
        if args.output_dir and epoch % 1 == 0:
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                       }, output_dir / f'checkpoint{epoch:04}.pth')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training on CoCo dataset',parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


