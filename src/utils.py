import os
import cv2
import matplotlib.pyplot as plt
import glob
import random
from src.config import RAW_IMG_DIR
import xml.etree.ElementTree as ET


def get_all_image_paths(directory=RAW_IMG_DIR):
    """
    Recursively finds all .bmp image paths, strictly excluding .xml or other files.
    """
    # Use a specific pattern for .bmp files
    # This looks into all subdirectories (**) for files ending in .bmp
    pattern = os.path.join(directory, "**", "*.bmp")

    # glob.glob with recursive=True handles the subfolder search
    # We also add a case-insensitive check just in case
    image_paths = glob.glob(pattern, recursive=True)
    image_paths_upper = glob.glob(os.path.join(directory, "**", "*.BMP"), recursive=True)

    # Combine and remove duplicates
    full_list = list(set(image_paths + image_paths_upper))

    # Final safety filter: ensure no .xml files accidentally slipped in
    clean_list = [p for p in full_list if p.lower().endswith('.bmp')]

    print(f"Total BMP images found: {len(clean_list)}")
    return sorted(clean_list)


def load_image(path):
    """
    Loads an image from a path in grayscale.
    """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image at {path}")
    return image


import numpy as np


def im_show(image, title="Angiography Frame", figsize=(5, 5), cmap='gray'):
    """
    General purpose image display function for the entire pipeline.

    Parameters:
    - image: numpy array. Can be grayscale, BGR, or float32/64.
    - title: string title for the plot.
    - figsize: tuple (width, height) in inches.
    - cmap: colormap to use for grayscale images.
    """
    plt.figure(figsize=figsize)

    # 1. Handle case where image is a float (e.g., 0.0 to 1.0)
    # If values are > 1 but float, we should normalize for display
    if image.dtype == np.float32 or image.dtype == np.float64:
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:  # Avoid division by zero
            image_disp = (image - img_min) / (img_max - img_min)
        else:
            image_disp = image
    else:
        image_disp = image.copy()

    # 2. Handle Color (3 channels) vs Grayscale (2 channels)
    if len(image_disp.shape) == 3:
        # OpenCV uses BGR, Matplotlib uses RGB.
        # We check the size of the 3rd dimension to ensure it's a color image
        if image_disp.shape[2] == 3:
            image_disp = cv2.cvtColor(image_disp.astype('uint8'), cv2.COLOR_BGR2RGB)
        plt.imshow(image_disp)
    else:
        # 3. Display grayscale
        plt.imshow(image_disp, cmap=cmap)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def get_xml_path(image_path):
    """
    Given an image path, returns the expected path of its .xml annotation.
    Assumes the xml is in the same folder with the same name.
    """
    return image_path.rsplit('.', 1)[0] + '.xml'


def parse_stenosis_xml(xml_path):
    """
    Parses the XML file and returns a list of bounding boxes.
    Each box is a dictionary: {'xmin':, 'ymin':, 'xmax':, 'ymax':}
    """
    if not os.path.exists(xml_path):
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []

    # Look for 'object' tags (standard in VOC format)
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        if bbox is not None:
            boxes.append({
                'xmin': int(float(bbox.find('xmin').text)),
                'ymin': int(float(bbox.find('ymin').text)),
                'xmax': int(float(bbox.find('xmax').text)),
                'ymax': int(float(bbox.find('ymax').text))
            })
    return boxes


def draw_bboxes(image, boxes, color=(0, 0, 255), thickness=2):
    """
    Draws bounding boxes on a grayscale image.
    Note: Converts image to BGR first so the box can be colored (e.g. Red).
    """
    # Convert grayscale to BGR for colored drawing
    img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for box in boxes:
        start_point = (box['xmin'], box['ymin'])
        end_point = (box['xmax'], box['ymax'])
        cv2.rectangle(img_bgr, start_point, end_point, color, thickness)

    return img_bgr