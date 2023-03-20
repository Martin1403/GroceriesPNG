import os
from argparse import ArgumentParser

from library.create_images import create_images
from library.create_polygons import make_polygons
from library.preprocess import preprocess_images


def main(args):
    # Preprocess images
    # preprocess_images(args)  # Move all images with polygons > 1
    # Make directories
    os.makedirs(args.images, exist_ok=True)
    os.makedirs(args.masks, exist_ok=True)
    os.makedirs(args.labels, exist_ok=True)
    # # Create images
    create_images(args)
    # # Draw polygon on images
    make_polygons(args.images, args.labels, args.masks)


parser = ArgumentParser(description='Settings for creating datasets')
parser.add_argument('--icons', type=str, default=r"data/our_rembg", help='Path to images')
parser.add_argument('--backgrounds', type=str, default=r"data/backgrounds_our", help='Path to backgrounds')
parser.add_argument('--images', type=str, default=r"data/images", help='Path to images')
parser.add_argument('--masks', type=str, default=r"data/masks", help='Path to masks')
parser.add_argument('--labels', type=str, default=r"data/labels", help='Path to labels')
parser.add_argument('--moved', type=str, default=r"data/moved", help='Path to labels')

parser.add_argument('--background', type=tuple, default=(800, 800), help='Output background size (square)')
parser.add_argument('--max_size', type=int, default=0, help='Image will be scaled, 0-turn off')
parser.add_argument('--groceries', type=tuple, default=(5, 10), help='Number of groceries. (min, max)')
parser.add_argument('--number', type=int, default=100, help='Number of images created')
parser.add_argument('--area', type=int, default=200, help='Minimum polygon area ')
parser.add_argument('--version', type=str, default="005", help='Version number')

if __name__ == '__main__':
    main(parser.parse_args())
