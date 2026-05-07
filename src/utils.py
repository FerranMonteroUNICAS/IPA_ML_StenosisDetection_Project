import os
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import xml.etree.ElementTree as ET

from src.config import RAW_IMG_DIR


def get_all_image_paths(directory=RAW_IMG_DIR):
    """
    Recursively finds all .bmp and .png image paths.
    """
    extensions = ["*.bmp", "*.BMP", "*.png", "*.PNG"]
    found = []
    for ext in extensions:
        pattern = os.path.join(directory, "**", ext)
        found.extend(glob.glob(pattern, recursive=True))

    clean_list = list(set(p for p in found if p.lower().endswith(('.bmp', '.png'))))
    print(f"Total images found: {len(clean_list)}")
    return sorted(clean_list)


def load_image(path):
    """
    Loads an image from a path in grayscale.
    """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image at {path}")
    return image


def get_xml_path(image_path):
    """
    Given an image path, returns the expected path of its .xml annotation.
    Assumes the xml is in the same folder with the same name.
    """
    return image_path.rsplit('.', 1)[0] + '.xml'


def parse_stenosis_xml(xml_path):
    """
    Parses a VOC-format XML file and returns a list of bounding boxes.
    Each box is a dict: {'xmin':, 'ymin':, 'xmax':, 'ymax':}
    """
    if not os.path.exists(xml_path):
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []

    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        if bbox is not None:
            boxes.append({
                'xmin': int(float(bbox.find('xmin').text)),
                'ymin': int(float(bbox.find('ymin').text)),
                'xmax': int(float(bbox.find('xmax').text)),
                'ymax': int(float(bbox.find('ymax').text)),
            })
    return boxes


def draw_bboxes(image, boxes, color=(0, 0, 255), thickness=2):
    """
    Draws bounding boxes on a grayscale image.
    Converts to BGR so the box can be colored.
    """
    img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for box in boxes:
        cv2.rectangle(img_bgr, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), color, thickness)
    return img_bgr


def im_show(image, title="Angiography Frame", figsize=(5, 5), cmap='gray'):
    """
    General-purpose image display for the entire pipeline.
    Handles grayscale, BGR color, and float images.
    """
    plt.figure(figsize=figsize)

    if image.dtype in (np.float32, np.float64):
        img_min, img_max = image.min(), image.max()
        image = (image - img_min) / (img_max - img_min) if img_max > img_min else image

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2RGB)
        plt.imshow(image)
    else:
        plt.imshow(image, cmap=cmap)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

