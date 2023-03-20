import csv
from random import randint

import cv2
import numpy as np

from library.png_image import load_images
from library.utils import polygon_from_mask, place_object_rgb, counter, load_background, place_object


def create_image(args, c):
    background_array = load_background(args)
    black_background = np.zeros(args.background, dtype=np.uint8)

    images = load_images(args.icons, args.background, num=randint(args.groceries[0], args.groceries[1]), max_size=args.max_size)

    name = f"IMG_{args.version}_{c}"
    with open(f"{args.labels}/{name}.txt", 'w', newline='\n') as file:
        writer = csv.writer(file, delimiter=' ')

        for pos, image in enumerate(images, 0):
            # Place RGB images
            background_array = place_object_rgb(polygon=image.polygon,
                                            background=background_array,
                                            image_mask=image.mask,
                                            image=image.rgb,
                                            off_x=image.off_x, off_y=image.off_y)

            # Place MASK in normal mode
            black_background = place_object(polygon=image.polygon,
                                            background=black_background,
                                            image_mask=image.mask_fill,
                                            image=image.mask_fill,
                                            off_x=image.off_x, off_y=image.off_y, inv=False)

            for image_inv in images[pos + 1:]:
                # Place inverted MASK
                black_background = place_object(polygon=image_inv.polygon,
                                                background=black_background,
                                                image_mask=image_inv.mask_fill,
                                                image=image_inv.mask_fill,
                                                off_x=image_inv.off_x, off_y=image_inv.off_y, inv=True)
            try:
                polygon_list = polygon_from_mask(black_background,  thresh=True)

                for polygon in polygon_list:
                    polygon_array = np.array(polygon, dtype=np.int32).reshape((-1, 2))
                    area = cv2.contourArea(polygon_array)
                    if area > args.area:
                        polygon_array_flatten = polygon_array.flatten() / args.background[0]
                        polygon_array_list = polygon_array_flatten.tolist()
                        polygon_array_list.insert(0, 0)
                        writer.writerow(polygon_array_list)

                cv2.imwrite(f"{args.images}/{name}.jpg", background_array)
                black_background = np.zeros(args.background, dtype=np.uint8)
            except ValueError:
                pass
                # print(f"No polygon found: {error}")


def create_images(args):
    for i in range(args.number):
        create_image(args, counter(i, 6))
