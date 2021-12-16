import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from data.dataloader import class_list
import torchvision.transforms as transforms




""" Visualization of ground truth """
def visualize_GT(images, labels):

    images = images.cpu()
    labels = labels.cpu()

    Ibatch, Ic, Iw, Ih = images.shape
    Lbatch, Lw, Lh, Lc = labels.shape

    assert (Ibatch == Lbatch)

    for i in range(Ibatch):

        img = images[i, :, :, :]
        label = labels[i, :, :]

        # Convert PIL Image
        img = transforms.ToPILImage()(img)
        W, H = img.size

        # declare draw object
        draw = ImageDraw.Draw(img)

        # Draw 7x7 Grid in Image
        dx = W // 7
        dy = H // 7

        y_start = 0
        y_end = H

        for i in range(0, W, dx):
            line = ((i, y_start), (i, y_end))
            draw.line(line, fill="red")

        x_start = 0
        x_end = W
        for i in range(0, H, dy):
            line = ((x_start, i), (x_end, i))
            draw.line(line, fill="red")

        obj_coord = label[:, :, 0]
        x_shift = label[:, :, 1]
        y_shift = label[:, :, 2]
        w_ratio = label[:, :, 3]
        h_ratio = label[:, :, 4]
        cls = label[:, :, 5]

        for i in range(7):
            for j in range(7):
                if obj_coord[i][j] == 1:

                    x_center = dx * i + int(dx * x_shift[i][j])
                    y_center = dy * j + int(dy * y_shift[i][j])
                    width = int(w_ratio[i][j] * Iw)
                    height = int(h_ratio[i][j] * Ih)

                    xmin = x_center - (width // 2)
                    ymin = y_center - (height // 2)
                    xmax = xmin + width
                    ymax = ymin + height

                    draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="blue")

                    draw.rectangle(((dx * i, dy * j), (dx * i + dx, dy * j + dy)), outline='#00ff88')
                    draw.ellipse(((x_center - 2, y_center - 2),
                                  (x_center + 2, y_center + 2)),
                                 fill='blue')
                    draw.text((dx * i, dy * j), class_list[int(cls[i][j])])

        plt.figure()
        plt.imshow(img)
        plt.show()
        plt.close()
