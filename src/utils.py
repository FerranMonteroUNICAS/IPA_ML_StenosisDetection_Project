import os
import random
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from src.config import RAW_IMG_DIR
from src.config import PROCESSED_IMG_DIR

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

def seprate_processed_files(all_image_paths):
    """
    Function used to obtain masks, skeletons and processed images separately after processing.
    :param all_image_paths: paths of all the .png files in the processed folder. They can be obtained using get_all_image_paths().
    :return: 3 lists of paths, one for eah type of processing pipeline output.
    """
    processed_images_paths = []
    masks_paths = []
    skeletons_paths = []
    for path in all_image_paths:
        try:
            if path.endswith('processed.png'):
                processed_images_paths.append(path)
            elif path.endswith('mask.png'):
                masks_paths.append(path)
            elif path.endswith('skeleton.png'):
                skeletons_paths.append(path)
        except:
            print(f"Warning: Could not classify file {path}")
    print(f"Processed images: {len(processed_images_paths)}, Masks: {len(masks_paths)}, Skeletons: {len(skeletons_paths)}")
    separated_paths = [processed_images_paths, masks_paths, skeletons_paths]
    return  separated_paths


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

def group_paths_by_serie(image_paths):
    """
    Groups image paths by their serieID, inferred from folder structure:
        .../patientID/serieID/filename.png

    Returns:
        dict mapping 'patientID/serieID' → sorted list of paths
    """
    groups = defaultdict(list)
    for path in image_paths:
        parts = Path(path).parts
        serie_key = f"{parts[-3]}/{parts[-2]}"
        groups[serie_key].append(path)

    return {k: sorted(v) for k, v in groups.items()}

def get_middle_frame(serie_paths):
    """
    Returns the middle frame path from a sorted list of serie paths.
    This is the frame where the contrast agent is typically best visualised.
    """
    return serie_paths[len(serie_paths) // 2]

# ─────────────────────────────────────────────────────────────
# Shared grid helper
# ─────────────────────────────────────────────────────────────
def _sample_patient_frames(series_dict, n, seed):
    """
    Shared helper used by all browse_*_grid functions.
    Selects one middle frame per patient, samples n patients randomly,
    and returns a list of (patient_id, loaded_image) tuples.

    Parameters:
        series_dict : output of group_paths_by_serie
        n           : number of patients to sample
        seed        : random seed for reproducibility

    Returns:
        List of (patient_id: str, image: np.ndarray) tuples,
        one per sampled patient. Images are already loaded into memory.
    """
    rng = random.Random(seed)

    # Collect one middle frame path per patient
    patients = defaultdict(list)
    for key, frame_paths in series_dict.items():
        patient_id = key.split("/")[0]
        patients[patient_id].append(get_middle_frame(frame_paths))

    patient_ids       = sorted(patients.keys())
    selected          = rng.sample(patient_ids, min(n, len(patient_ids)))
    sample_frames     = [(pid, rng.choice(patients[pid])) for pid in selected]

    # Load images, skip failures
    loaded = []
    for pid, path in sample_frames:
        img = load_image(path)
        if img is not None:
            loaded.append((pid, img))

    return loaded

# ─────────────────────────────────────────────────────────────
# Grid and single image saving
# ─────────────────────────────────────────────────────────────
def save_grid(series_dict, output_path, pipeline_fn, n=9, seed=42, dpi=150, **pipeline_kwargs):
    """
    Renders a 3×3 grid of preprocessed patient images and saves it as a PNG.
    Decoupled from the interactive browse functions — call it any time with
    whichever pipeline function and parameters you have settled on.

    Parameters:
        series_dict  : output of group_paths_by_serie
        output_path  : full path for the saved PNG (e.g. 'output/grid_bgclahe.png')
        pipeline_fn  : a function (image) → processed image, built from your
                       chosen pipeline steps. See examples below.
        n            : number of patients in the grid (default 9)
        seed         : random seed — use the same seed as your browse calls
                       to get the same patients
        dpi          : image resolution (default 150; increase for publication)
        **pipeline_kwargs: not used directly — bake parameters into pipeline_fn
                           via a lambda (see examples)

    Examples:
        # BG subtraction only
        save_grid(series, 'output/grid_bg.png',
                  pipeline_fn=lambda img: subtract_background(img, kernel_size=41))

        # BG sub → CLAHE
        save_grid(series, 'output/grid_bgclahe.png',
                  pipeline_fn=lambda img: apply_clahe(
                      subtract_background(img, kernel_size=41), clip_limit=2.0))

        # Full pipeline: BG sub → CLAHE → NLM
        save_grid(series, 'output/grid_full_nlm.png',
                  pipeline_fn=lambda img: apply_nlm(
                      apply_clahe(
                          subtract_background(img, kernel_size=41), clip_limit=2.0),
                      h=10, template_window=7, search_window=21))

        # Full pipeline: BG sub → CLAHE → Bilateral
        save_grid(series, 'output/grid_full_bilateral.png',
                  pipeline_fn=lambda img: apply_bilateral(
                      apply_clahe(
                          subtract_background(img, kernel_size=41), clip_limit=2.0),
                      d=9, sigma_color=75, sigma_space=75))
    """
    loaded = _sample_patient_frames(series_dict, n, seed)
    cols = 3
    rows = int(np.ceil(len(loaded) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()

    for i, (pid, img) in enumerate(loaded):
        axes[i].imshow(pipeline_fn(img), cmap='gray')
        axes[i].set_title(f"P{pid}", fontsize=9)
        axes[i].axis('off')

    for j in range(len(loaded), len(axes)):
        axes[j].axis('off')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Grid saved → {output_path}")

def save_single(series_dict, output_dir, pipeline_fn, positions, seed=42, dpi=300):
    """
    Saves selected images from the 3×3 grid as individual PNG files.
    Each selected position is saved as a separate file named by patient ID.

    Grid positions are 1-indexed, left-to-right, top-to-bottom:
        1  2  3
        4  5  6
        7  8  9

    Parameters:
        series_dict : output of group_paths_by_serie
        output_dir  : folder where the images will be saved
        pipeline_fn : a function (image) → processed image (same as save_grid)
        positions   : list of 1-indexed grid positions to save (e.g. [1, 5, 9])
        seed        : random seed — use the same as save_grid to get the same patients
        dpi         : image resolution (default 300)

    Example:
        save_single(series, 'output/singles',
                    pipeline_fn=lambda img: apply_nlm(
                        apply_clahe(
                            subtract_background(img, kernel_size=41), clip_limit=2.0),
                        h=10, template_window=7, search_window=21),
                    positions=[1, 5, 9])
    """
    loaded = _sample_patient_frames(series_dict, n=9, seed=seed)

    invalid = [p for p in positions if p < 1 or p > len(loaded)]
    if invalid:
        raise ValueError(f"Positions {invalid} are out of range. "
                         f"Valid range is 1–{len(loaded)}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pos in positions:
        pid, img = loaded[pos - 1]  # convert 1-indexed to 0-indexed

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(pipeline_fn(img), cmap='gray')
        ax.set_title(f"P{pid}", fontsize=10)
        ax.axis('off')

        plt.tight_layout()
        out_path = output_dir / f"P{pid}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {out_path}")


def get_processed_image_paths(
        directory=PREPROCESSED_IMG_DIR
):
    """
    Returns ONLY *_processed.png images
    from dataset_preprocessed.
    """

    pattern = os.path.join(
        directory,
        "**",
        "*_processed.png"
    )

    found = glob.glob(
        pattern,
        recursive=True
    )

    clean_list = sorted(list(set(found)))

    return clean_list

def get_skeleton_path(processed_path):

    return processed_path.replace(
        "_processed.png",
        "_skeleton.png"
    )


def get_bbox_xml_path(processed_path):

    return processed_path.replace(
        "_processed.png",
        "_bbox.xml"
    )