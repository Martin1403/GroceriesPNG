import os
import cv2
import numpy as np
from random import sample, randint

from library.utils import load_image_mask, polygon_from_mask, get_max_list


class PngImage:
    def __init__(self, path, position, back_size, thresh=127, rotate=90, max_size=800):
        self.path = path
        self.position = position

        self.rgb, self.mask = load_image_mask(path, thresh, rotate, max_size)
        self.off_x = randint(0, back_size[1] - self.rgb.shape[1])
        self.off_y = randint(0, back_size[0] - self.rgb.shape[0])
        polygon_list = polygon_from_mask(self.mask)
        self.polygon = np.array(get_max_list(polygon_list), dtype=np.int32).reshape((-1, 2))

        self.mask_fill = np.copy(self.mask)
        contours, _ = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours[1:]:
            cv2.fillPoly(self.mask_fill, pts=[contour], color=(255, 255, 255))


def __repr__(self):
    return f"Image path: {self.path}"


def load_images(images_path, background, num=5, max_size=800):
    images = []
    for count, image_name in enumerate(sample(os.listdir(images_path), num)):
        image_path = os.path.join(images_path, image_name)
        angle = randint(0, 360)
        try:
            images.append(PngImage(image_path, count, background, thresh=127, rotate=angle, max_size=max_size))
        except (ValueError, IndexError):
            print(image_name)
    return images
