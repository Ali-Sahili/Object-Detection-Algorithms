import numpy as np
import imgaug as ia
from PIL import Image
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from mpl_toolkits.axes_grid1 import ImageGrid


def CvtCoordsXXYY2XYWH(image_width, image_height, xmin, xmax, ymin, ymax):
    # calculate bbox_center
    bbox_center_x = (xmin + xmax) / 2
    bbox_center_y = (ymin + ymax) / 2

    # calculate bbox_size
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    # normalize
    normalized_x = bbox_center_x / image_width
    normalized_y = bbox_center_y / image_height
    normalized_w = bbox_width / image_width
    normalized_h = bbox_height / image_height

    return normalized_x, normalized_y, normalized_w, normalized_h


def CvtCoordsXYWH2XXYY(normed_lxywh, image_width, image_height):
    centered_x = normed_lxywh[1] * image_width
    centered_y = normed_lxywh[2] * image_height
    object_width = normed_lxywh[3] * image_width
    object_height = normed_lxywh[4] * image_height

    xmin = centered_x - object_width / 2
    xmax = centered_x + object_width / 2
    ymin = centered_y - object_height / 2
    ymax = centered_y + object_height / 2

    return xmin, xmax, ymin, ymax

def GetImgaugStyleBBoxes(normed_lxywhs, image_width, image_height):
    bbs = ia.BoundingBoxesOnImage([], shape=(image_width, image_height))

    for normed_lxywh in normed_lxywhs:
        xxyy = CvtCoordsXYWH2XXYY(normed_lxywh, image_width, image_height)
        bbs.bounding_boxes.append(ia.BoundingBox(x1=xxyy[0], x2=xxyy[1], y1=xxyy[2], y2=xxyy[3], label='None'))

    return bbs


def GetYoloStyleBBoxes(normed_lxywhs, bbs_aug, image_width, image_height):
    normed_bbs_aug = []

    for i in range(len(bbs_aug.bounding_boxes)):
        after = bbs_aug.bounding_boxes[i]
        coord = CvtCoordsXXYY2XYWH(image_width, image_height, xmin=after.x1, xmax=after.x2, ymin=after.y1, ymax=after.y2)
        normed_bbs_aug.append([normed_lxywhs[i][0], coord[0], coord[1], coord[2], coord[3]])

    return normed_bbs_aug


def augmentImage(image, normed_lxywhs, image_width, image_height, seq):

    bbs = GetImgaugStyleBBoxes(normed_lxywhs, image_width, image_height)

    seq_det = seq.to_deterministic()

    image_aug = seq_det.augment_images([image])[0]

    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    bbs_aug = bbs_aug.remove_out_of_image().cut_out_of_image()

    if(False):
        image_before = bbs.draw_on_image(image, thickness=5)
        image_after = bbs_aug.draw_on_image(image_aug, thickness=5, color=[0, 0, 255])

        fig = plt.figure(1, (10., 10.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, 2),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    normed_bbs_aug = GetYoloStyleBBoxes(normed_lxywhs, bbs_aug, image_width, image_height)

    return image_aug, normed_bbs_aug


class Augmenter(object):

    def __init__(self, seq):
        self.seq = seq

    def __call__(self, sample):

        image = sample[0]  # PIL image
        normed_lxywhs = sample[1]
        image_width, image_height = image.size

        image = np.array(image)  # PIL image to numpy array

        image_aug, normed_bbs_aug = augmentImage(image, normed_lxywhs, image_width, image_height, self.seq)

        image_aug = Image.fromarray(image_aug)  # numpy array to PIL image Again!
        return image_aug, normed_bbs_aug
