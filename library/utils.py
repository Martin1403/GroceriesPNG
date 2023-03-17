import os
import cv2
import numpy as np
from random import choice, randint


def rotate_image(mat, angle):
    """Rotates an image (angle in degrees) """
    height, width = mat.shape[: 2]  # image shape has 3 dimensions
    image_center = (width / 2, height / 2)

    # rotation_mat = cv2.getRotationMatrix2D(image_center, angle, randint(8, 12) / 10.)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to orig and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def resize_image(img_array, max_size=300):
    width = img_array.shape[1]
    height = img_array.shape[0]
    scale = max_size / max([width, height])
    width = int(width * scale)
    height = int(height * scale)
    return cv2.resize(img_array, (width, height), interpolation=cv2.INTER_AREA)


def load_image_mask(image_path, thresh=127, rotate=45, max_size=800):
    image_array = cv2.imread(image_path, -1)  # Load image
    #image_array_res = resize_image(image_array, max_size=max_size)  # Resize image with scale
    image_array_res = image_array  # Resize image with scale
    #image_array_rot = rotate_image(image_array_res, angle=randint(0, 180))  # Random rotate
    image_array_rot = rotate_image(image_array_res, angle=0)  # randint(0, 180))  # Random rotate
    image_array_rgb = image_array_rot[:, :, :3]  # Index only BGR channel
    # image_array_rgb = cv2.cvtColor(image_array_bgr, cv2.COLOR_BGR2RGB)  # Convert not needed
    image_array_mask = image_array_rot[:, :, -1]  # Index alpha channel
    image_array_mask_thresh = np.where(image_array_mask <= thresh, 0, 255).astype(np.uint8)  # Mask threshold

    kernel = np.ones((5, 5), np.uint8)
    eroded_img = cv2.erode(image_array_mask_thresh, kernel, iterations=1)

    image_array_rgb = cv2.bitwise_and(image_array_rgb, image_array_rgb, mask=eroded_img)
    
    return cut_image(image_array_rgb, eroded_img)


def polygon_from_mask(masked_array):
    """The function return polygon from mask"""
    contours, _ = cv2.findContours(masked_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    valid_poly = 0
    for c, contour in enumerate(contours):
        if contour.size >= 6:  # Valid polygons have >= 6 coordinates (3 points)
            segmentation.append(contour.astype(float).flatten().tolist())
            valid_poly += 1
    # if valid_poly == 0:
    #     raise ValueError
    return segmentation


def cut_image(image, mask):
    polygon_list = polygon_from_mask(mask)
    polygon = np.array(polygon_list[0], dtype=np.int32).reshape((-1, 2))

    (min_x, min_y), (max_x, max_y) = np.min(polygon, axis=0), np.max(polygon, axis=0)
    image_crop = image[min_y: max_y, min_x: max_x]
    mask_crop = mask[min_y: max_y, min_x: max_x]    
    return image_crop, mask_crop


def place_object(polygon, background, image_mask, image, off_x=0, off_y=0, inv=False):
    (min_x, min_y), (max_x, max_y) = np.min(polygon, axis=0), np.max(polygon, axis=0)
    roi = background[min_y+off_y: max_y+off_y, min_x+off_x: max_x+off_x]
    mask = image_mask[min_y: max_y, min_x: max_x] 
    
    mask_inv = cv2.bitwise_not(mask)

    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    src_cut = image[min_y: max_y, min_x: max_x]
    dst = cv2.add(img_bg, src_cut)
    # dst = cv2.bitwise_or(img_bg, src_cut)
    if inv:
        dst = img_bg

    final = background.copy()
    
    final[min_y+off_y: max_y+off_y, min_x+off_x: max_x+off_x] = dst
    
    return final


def get_max_list(inp_list):
    inp_list.sort(key=len, reverse=True)
    return inp_list[0]


def counter(num=1, length=3):
    """Counter etc. 0001, 0002
    Attributes:
    num (int) integer etc. 1 ==> 0001
        length (int) length of counter etc. 3 ==> 001
    Return:
        (str) etc. 0001
    """
    number = '0' * length + str(num)
    number = number[len(number)-length:]
    return number


def load_background(args):
    results = []
    for name in os.listdir(args.backgrounds):
        path = os.path.join(args.backgrounds, name)
        try:
            results.append(cv2.resize(cv2.imread(path), args.background, interpolation=cv2.INTER_AREA))
        except:
            print(f"Background resize error: {name}")
    return choice(results)