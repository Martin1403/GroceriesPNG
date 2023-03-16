import os
import shutil

import cv2
import numpy as np

from library.utils import polygon_from_mask


def preprocess_images(args):
    os.makedirs(args.moved, exist_ok=True)
    for image_name in os.listdir(args.icons):
        image_path = os.path.join(args.icons, image_name)
        image_array = cv2.imread(image_path, -1)  # load image with alpha (-1) for load all channels with alpha
        image_array_mask = image_array[:, :, -1]  # Mask from PNG
        image_array_mask_thresh = np.where(image_array_mask <= 127, 0, 255).astype(np.uint8)  # Setup threshold

        num_polygons = len(polygon_from_mask(image_array_mask_thresh))
        if num_polygons > 1:
            shutil.move(image_path, os.path.join(args.moved, f"P-{num_polygons}-{image_name}"))
