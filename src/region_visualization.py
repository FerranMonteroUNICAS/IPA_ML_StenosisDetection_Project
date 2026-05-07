import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from src.utils import load_image, get_xml_path, parse_stenosis_xml


# ─────────────────────────────────────────────────────────────
# ROI cropping
# ─────────────────────────────────────────────────────────────

def crop_roi(image, box, padding=4):
    """
    Crops a single bounding box region from a grayscale image.

    Parameters:
        image  : grayscale numpy array
        box    : dict with 'xmin', 'ymin', 'xmax', 'ymax'
        padding: extra pixels of context around the box (default 4)

    Returns:
        Cropped grayscale numpy array, or None if the crop is empty.
    """
    h, w = image.shape[:2]
    xmin = max(0, box['xmin'] - padding)
    ymin = max(0, box['ymin'] - padding)
    xmax = min(w, box['xmax'] + padding)
    ymax = min(h, box['ymax'] + padding)

    crop = image[ymin:ymax, xmin:xmax]
    return crop if crop.size > 0 else None


def _group_paths_by_serie(image_paths):
    """
    Groups image paths by their serieID, inferred from the folder structure:
        .../patientID/serieID/filename.png

    Returns:
        dict mapping 'patientID/serieID' → list of paths
    """
    groups = defaultdict(list)
    for path in image_paths:
        parts = Path(path).parts
        # Last two folders are patientID/serieID
        serie_key = f"{parts[-3]}/{parts[-2]}"
        groups[serie_key].append(path)
    return groups


def collect_all_rois(image_paths, padding=4, images_per_serie=1, seed=42):
    """
    Collects ground-truth stenosis crops, sampling a fixed number of
    images per serie to avoid redundancy from highly similar frames.

    Parameters:
        image_paths      : list of all image file paths
        padding          : extra pixels of context around each bounding box
        images_per_serie : how many images to sample from each serie (default 3)
        seed             : random seed for reproducibility

    Returns:
        List of dicts, each with:
            'roi'   : cropped grayscale numpy array
            'path'  : source image path
            'box'   : original bounding box dict
            'serie' : 'patientID/serieID' key
    """
    rng     = random.Random(seed)
    groups  = _group_paths_by_serie(image_paths)
    results = []

    for serie_key, paths in sorted(groups.items()):
        # Only keep paths that actually have an annotation
        annotated = [p for p in paths if parse_stenosis_xml(get_xml_path(p))]
        if not annotated:
            continue

        sampled = rng.sample(annotated, min(images_per_serie, len(annotated)))

        for path in sampled:
            image = load_image(path)
            if image is None:
                continue
            boxes = parse_stenosis_xml(get_xml_path(path))
            for box in boxes:
                roi = crop_roi(image, box, padding=padding)
                if roi is not None:
                    results.append({'roi': roi, 'path': path, 'box': box, 'serie': serie_key})

    print(f"Series found      : {len(groups)}")
    print(f"Images per serie  : {images_per_serie}")
    print(f"Total ROIs collected: {len(results)}")
    return results


# ─────────────────────────────────────────────────────────────
# Mosaic builder
# ─────────────────────────────────────────────────────────────

def build_mosaic(rois, tile_size=(64, 64), n_cols=10):
    """
    Resizes all ROIs to a fixed tile size and arranges them
    into a grid (mosaic) image.

    Parameters:
        rois     : list of grayscale numpy arrays
        tile_size: (width, height) of each tile in the mosaic
        n_cols   : number of columns in the grid

    Returns:
        Single grayscale numpy array representing the full mosaic.
    """
    n       = len(rois)
    n_rows  = int(np.ceil(n / n_cols))
    tw, th  = tile_size
    mosaic  = np.zeros((n_rows * th, n_cols * tw), dtype=np.uint8)

    for idx, roi in enumerate(rois):
        row = idx // n_cols
        col = idx % n_cols
        tile = cv2.resize(roi, (tw, th), interpolation=cv2.INTER_AREA)
        mosaic[row * th:(row + 1) * th, col * tw:(col + 1) * tw] = tile

    return mosaic


# ─────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────

def show_mosaic(mosaic, title="Stenosis ROI Mosaic", figsize=(18, None)):
    """
    Displays the mosaic image using matplotlib.

    Parameters:
        mosaic : 2D numpy array (output of build_mosaic)
        title  : plot title
        figsize: (width, height) in inches. If height is None,
                 it is computed automatically from the aspect ratio.
    """
    h, w = mosaic.shape
    width  = figsize[0]
    height = figsize[1] if figsize[1] is not None else width * (h / w)

    plt.figure(figsize=(width, height))
    plt.imshow(mosaic, cmap='gray')
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()