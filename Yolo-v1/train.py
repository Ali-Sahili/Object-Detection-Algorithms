import warnings
warnings.filterwarnings("ignore")

import os
import torch
import visdom
from imgaug import augmenters as iaa
import torchvision.transforms as transforms


from model import YOLOv1
from loss import loss_function
from utils.summary import summary
from utils.visualize import visualize_GT
from data.data_augmentation import Augmenter
from data.dataloader import detection_collate, VOC, class_list
from utils.helpers import save_checkpoint, create_vis_plot, update_vis_plot



def train(args):

    num_classes = len(class_list)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    """
    if args.use_visdom:
        viz = visdom.Visdom(use_incoming_socket=False)
        vis_title = 'Yolo V1 Deepbaksu_vision (feat. martin, visionNoob) PyTorch on ' + 'VOC'
        vis_legend = ['Train Loss']
        iter_plot = create_vis_plot(viz, 'Iteration', 'Total Loss', vis_title, vis_legend)
        coord1_plot = create_vis_plot(viz, 'Iteration', 'coord1', vis_title, vis_legend)
        size1_plot = create_vis_plot(viz, 'Iteration', 'size1', vis_title, vis_legend)
        noobjectness1_plot = create_vis_plot(viz, 'Iteration', 'noobjectness1', vis_title, 
                                                                                vis_legend)
        objectness1_plot = create_vis_plot(viz, 'Iteration', 'objectness1', vis_title, 
                                                                            vis_legend)
        obj_cls_plot = create_vis_plot(viz, 'Iteration', 'obj_cls', vis_title, vis_legend)
    """
    # 2. Data augmentation setting
    if (args.use_augmentation):
        seq = iaa.SomeOf(2, [
            iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
            iaa.Affine(
                translate_px={"x": 3, "y": 10},
                scale=(0.9, 0.9)
            ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
            iaa.AdditiveGaussianNoise(scale=0.1 * 255),
            iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
            iaa.Affine(rotate=45),
            iaa.Sharpen(alpha=0.5)
        ])
    else:
        seq = iaa.Sequential([])

    # 3. Load Dataset
    # composed
    # transforms.ToTensor
    train_dataset = VOC(root=args.data_path, transform=transforms.Compose([Augmenter(seq)]))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               collate_fn=detection_collate)
    print("Dataset loaded!")
    print("Number of batches: ", len(train_loader))
    print()
    
    # Define model
    model = YOLOv1(dropout_prop = args.dropout, num_classes = num_classes)
    model.to(device)
    
    if args.use_summary:
        summary(model, (3, args.input_height, args.input_width), use_cuda=use_cuda)

    # 7.Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Train the model
    total_step = len(train_loader)

    print("Start training ...")
    for epoch in range(1, args.num_epochs + 1):

        if (epoch==200) or (epoch==400) or (epoch==600) or (epoch==20000) or (epoch == 30000):
            scheduler.step()

        for i, (images, labels, sizes) in enumerate(train_loader):

            if args.check_gt:
                visualize_GT(images, labels)

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calc Loss
            loss, \
            obj_coord1_loss, \
            obj_size1_loss, \
            obj_class_loss, \
            noobjness1_loss, \
            objness1_loss = loss_function(outputs, labels, use_cuda)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.step_display == 0:
                print('epoch: [{}/{}], batch step [{}/{}], lr: {}, total_loss: {:.4f}, coord1: {:.4f}, size1: {:.4f}, noobj_clss: {:.4f}, objness1: {:.4f}, class_loss: {:.4f}'.format(epoch + 1, args.num_epochs, i + 1, total_step, ([param_group['lr'] for param_group in optimizer.param_groups])[0], loss.item(), obj_coord1_loss, obj_size1_loss, noobjness1_loss, objness1_loss, obj_class_loss))

                """
                if args.use_visdom:
                    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), loss.item(), 
                                    iter_plot, None, 'append')
                    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), obj_coord1_loss, 
                                    coord1_plot, None, 'append')
                    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), obj_size1_loss, 
                                    size1_plot, None, 'append')
                    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), obj_class_loss, 
                                    obj_cls_plot, None, 'append')
                    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), noobjness1_loss, 
                                    noobjectness1_plot, None, 'append')
                    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), objness1_loss, 
                                    objectness1_plot, None, 'append')
                """

        """
        if ((epoch % 1000) == 0) and (epoch != 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': "YOLOv1",
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, False, filename=os.path.join(args.checkpoint_path, 'ep{:05d}_loss{:.04f}_lr{}.pth.tar'.format(epoch, loss.item(), ([param_group['lr'] for param_group in optimizer.param_groups])[0])))
         """
