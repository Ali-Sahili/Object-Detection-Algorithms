import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from utils import adjust_learning_rate
from model.faster_rcnn import faster_rcnn
from chainercv.visualizations import vis_bbox
from chainercv.datasets import VOCBboxDataset
from chainercv.datasets import voc_bbox_label_names
from torchnet.meter import AverageValueMeter, MovingAverageValueMeter


# Parameters
lr = 0.001
num_epochs = 15

# Dataset loading
train_dataset = VOCBboxDataset(year='2007', split='train')
val_dataset = VOCBboxDataset(year='2007', split='val')
trainval_dataset = VOCBboxDataset(year='2007', split='trainval')
test_dataset = VOCBboxDataset(year='2007', split='test')

# Defining the model
model = faster_rcnn(20, backbone='vgg16')
if torch.cuda.is_available():
    model = model.cuda()

# Defining the optimizer
optimizer = model.get_optimizer(is_adam=False)

# Training settings
avg_loss = AverageValueMeter()
ma20_loss = MovingAverageValueMeter(windowsize=20)
model.train()

print("Start training ...")
for epoch in range(num_epochs):
    adjust_learning_rate(optimizer, epoch, lr, lr_decay_epoch=10)
    for i in range(len(trainval_dataset)):
        img, bbox, label = trainval_dataset[i]
        img = img/255

        loss = model.loss(img, bbox, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.cpu().data.numpy()[0]
        avg_loss.add(loss_value)
        ma20_loss.add(float(loss_value))
        
        print('[epoch:{}]  [batch:{}/{}]  [sample_loss:{:.4f}]  [avg_loss:{:.4f}]  [ma20_loss:{:.4f}]'.format(epoch, i, len(trainval_dataset), loss_value, avg_loss.value()[0], ma20_loss.value()[0]))

    # Saving weights
    modelweight = model.state_dict()
    trainerstate = {'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch}
    torch.save(modelweight, "epoch_"+str(epoch)+".modelweight")
    torch.save(trainerstate, "epoch_"+str(epoch)+".trainerstate")


# Evaluation phase
model.eval()
for i in range(len(test_dataset)):
    img, _, _ = test_dataset[i]
    imgx = img/255
    bbox_out, class_out, prob_out = model.predict(imgx, prob_threshold=0.95)

    vis_bbox(img, bbox_out, class_out, prob_out,label_names=voc_bbox_label_names) 
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(11, 5)
    fig.savefig('test_'+str(i)+'.jpg', dpi=100)



