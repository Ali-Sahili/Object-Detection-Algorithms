import os
import cv2
import glob
import pickle
import numpy as np
from utils import *
from model import YOLOv2

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 
           'train', 'tvmonitor']


def test(opt):
    if torch.cuda.is_available():
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path)
        else:
            model = YOLOv2(len(CLASSES))
            model.load_state_dict(torch.load(opt.pre_trained_model_path))
    else:
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path, map_location=lambda storage, 
                                                           loc: storage)
        else:
            model = YOLOv2(len(CLASSES))
            model.load_state_dict(torch.load(opt.pre_trained_model_path, 
                                             map_location=lambda storage, loc: storage))
    model.eval()
    colors = pickle.load(open("src/pallete", "rb"))

    for image_path in glob.iglob(opt.input + os.sep + '*.jpg'):
        if "prediction" in image_path:
            continue
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image = cv2.resize(image, (opt.image_size, opt.image_size))
        image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
        image = image[None, :, :, :]
        width_ratio = float(opt.image_size) / width
        height_ratio = float(opt.image_size) / height
        data = Variable(torch.FloatTensor(image))
        if torch.cuda.is_available():
            data = data.cuda()
        with torch.no_grad():
            logits = model(data)
            predictions = post_processing(logits, opt.image_size, CLASSES, model.anchors, 
                                          opt.conf_threshold, opt.nms_threshold)
        if len(predictions) != 0:
            predictions = predictions[0]
            output_image = cv2.imread(image_path)
            for pred in predictions:
                xmin = int(max(pred[0] / width_ratio, 0))
                ymin = int(max(pred[1] / height_ratio, 0))
                xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
                ymax = int(min((pred[1] + pred[3]) / height_ratio, height))
                color = colors[CLASSES.index(pred[5])]
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(pred[5] + ' : %.2f' % pred[4], 
                                            cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, 
                                                           ymin + text_size[1] + 4), color, -1)
                cv2.putText(
                    output_image, pred[5] + ' : %.2f' % pred[4],
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)
                print("Object: {}, Bounding box: ({},{}) ({},{})".format(pred[5], xmin, xmax, 
                                                                                  ymin, ymax))
            cv2.imwrite(image_path[:-4] + "_prediction.jpg", output_image)

