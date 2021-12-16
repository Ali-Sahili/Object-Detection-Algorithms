
import torch
from losses.loss_func import Loss
from dataset.params import Config
from dataset.voc import VOCDataset
from trainer.trainer import Trainer
from model.centernet import CenterNet



def train(cfg):
    train_set = VOCDataset(cfg.root, mode=cfg.split, resize_size=cfg.resize_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, 
                                               num_workers=cfg.num_workers, 
                                               collate_fn=train_set.collate_fn, 
                                               pin_memory=True)

    model = CenterNet(cfg)
    if cfg.gpu:
        model = model.cuda()
    loss_func = Loss(cfg)

    num_epochs = 100
    cfg.max_iter = len(train_loader) * num_epochs
    cfg.steps = (int(cfg.max_iter * 0.6), int(cfg.max_iter * 0.8))

    trainer = Trainer(cfg, model, loss_func, train_loader, None)
    trainer.train()


if __name__ == '__main__':
    train(Config)
