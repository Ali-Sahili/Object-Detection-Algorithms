
import matplotlib.pyplot as plt
from model.faster_rcnn import faster_rcnn
from chainercv.visualizations import vis_bbox
from chainercv.datasets import voc_bbox_label_names



path = 'pretrained-model-path'
#model = faster_rcnn(n_class=20, model_path=path).cuda()
model = faster_rcnn(n_class=20, backbone='vgg16').cuda() 
model.eval()

filename = 'example.jpg'
img = plt.imread(filename)
img = img.transpose(2,0,1)
imgx = img/255
bbox_out, class_out, prob_out = model.predict(imgx, prob_threshold=0.95)
vis_bbox(img, bbox_out, class_out, prob_out,label_names=voc_bbox_label_names) 
plt.show()
