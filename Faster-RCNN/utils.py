




def adjust_learning_rate(optimizer, epoch, init_lr, lr_decay_factor=0.1, lr_decay_epoch=10):
    """Sets the learning rate to the initial LR decayed by lr_decay_factor every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch == 0:
        lr = init_lr * (lr_decay_factor ** (epoch // lr_decay_epoch))
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
