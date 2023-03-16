import cv2
import os
import numpy as np
from random import randint


def make_polygons(images, labels, masks):
    for image_path, label_path in zip(os.listdir(images), os.listdir(labels)):
        assert image_path.split(".")[0] == label_path.split(".")[0]
        image = images + os.sep + image_path
        label = labels + os.sep + label_path
        mask = masks + os.sep + image_path

        image_array = cv2.imread(image)

        with open(label, "r") as f:
            lines = f.read().splitlines()

        for i in range(len(lines)):
            points = lines[i].split(" ")[1:]

            X = np.expand_dims(
                np.array([float(i) * image_array.shape[1] for c, i in enumerate(points) if c % 2 == 0], dtype=np.int32),
                axis=1)
            Y = np.expand_dims(
                np.array([float(i) * image_array.shape[0] for c, i in enumerate(points) if c % 2 == 1], dtype=np.int32),
                axis=1)

            pts = np.concatenate([X, Y], axis=1)

            pts = pts.reshape((-1, 1, 2))
            isClosed = True
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            thickness = 5
            image_array = cv2.polylines(image_array, [pts], isClosed, color, thickness)

        cv2.imwrite(mask, image_array)
