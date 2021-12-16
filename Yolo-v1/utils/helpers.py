import torch
import numpy as np



def str2bool(v):
    """ For arguments parser """
    return v.lower() in ("yes", "true", "t", "1")


def one_hot(output, label, use_cuda):

    label = label.cpu().data.numpy()
    b, s1, s2, c = output.shape
    dst = np.zeros([b, s1, s2, c], dtype=np.float32)

    for k in range(b):
        for i in range(s1):
            for j in range(s2):

                dst[k][i][j][int(label[k][i][j])] = 1.

    result = torch.from_numpy(dst)

    if use_cuda: 
        result = result.type(torch.FloatTensor).cuda()
    else:
        result = result.type(torch.FloatTensor)

    return result


# visdom function
def create_vis_plot(viz, _xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 1)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(viz, iteration, loss, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 1)).cpu() * iteration,
        Y=torch.Tensor([loss]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')








