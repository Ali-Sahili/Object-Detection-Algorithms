import torch
import random
import requests
from PIL import Image
import torchvision.transforms as T
from box_ops import rescale_bboxes
from models.detr_models import detr_resnet50
from visualize import plot_results, show_patches


# standard PyTorch mean-std input image normalization
image_transform = T.Compose([
    T.Resize(420),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
patch_aug_transform = T.Compose([
    T.RandomApply([
      T.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    T.RandomGrayscale(p=0.2),
])
patch_transform = T.Compose([
    T.Resize((128,128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load detr model
model = detr_resnet50(pretrained=True)#, num_classes=2)
model.eval()


# Image in the paper
url = 'https://p0.piqsels.com/preview/126/30/480/animals-cats-cute-feline.jpg'
image = Image.open(requests.get(url, stream=True).raw)


# To make visualization beautiful, we manually crop some reasonable objects here.
patches = []
patches = [image.crop((19,112,193,284)), image.crop((238,19,399,403)), 
           image.crop((408,80,579,249)), image.crop((540,141,772,349)), 
           image.crop((728,78,872,214)), image.crop((25,19,872,411)) ]
           
# You can also try to randomly crop "objects".
# w,h = image.size
# for i in range(5):
#   x = random.randint(0,w-32) # avoid too small objects.
#   y = random.randint(0,h-32)
#   xx = random.randint(x+32,w)
#   yy = random.randint(y+32,h)
#   patches.append(image.crop((x, y, xx, yy)))

# Make the task difficult by applying data augmentations.
patches = [patch_aug_transform(patch) for patch in patches]

# Single batch
image_tensor = image_transform(image).unsqueeze(0)
patches_tensor = torch.stack([patch_transform(patch) for patch in patches], dim=0).unsqueeze(0)
outputs = model(image_tensor)#, patches_tensor)

probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
max_probas = probas.max(-1).values
keep = torch.zeros(max_probas.shape[0],dtype=torch.bool)

# 10 object queries are assigned into a group.
# get argmax() index for every 10 object queries.
for i in range(patches_tensor.shape[1]):
  keep[i*10+max_probas[i*10:(i+1)*10].argmax().item()] = 1

# convert boxes from [0; 1] to image scales
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, :], image.size)
plot_results(image, probas, bboxes_scaled)

bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], image.size)
plot_results(image, probas[keep], bboxes_scaled)

# show patches
show_patches(patches)
