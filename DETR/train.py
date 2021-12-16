import sys
import math
import torch

from utils import reduce_dict



def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, 
                    max_norm = 0, print_freq = 10):
    
    model.train()
    criterion.train()
    
    for i, (samples, targets) in enumerate(data_loader):
    
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Backprop
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if (i+1) % print_freq == 0:
            print("Epoch: {} | Batch: {} | Loss: {} |".format(epoch, i+1, loss_value))


    return loss_dict_reduced_unscaled, loss_dict_reduced_scaled

