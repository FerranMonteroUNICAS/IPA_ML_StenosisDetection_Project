import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider, VBox, HBox

from src.utils import load_image

# ─────────────────────────────────────────────────────────────
# Serie grouping
# ─────────────────────────────────────────────────────────────

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
# Core matching function
# ─────────────────────────────────────────────────────────────

def match_to_reference(image, reference):
    """
    Matches the histogram of a grayscale image to a reference image using OpenCV.

    Parameters:
        image    : grayscale uint8 numpy array (source)
        reference: grayscale uint8 numpy array (target histogram)

    Returns:
        Histogram-matched grayscale uint8 numpy array
    """
    # 1. Calculate histograms for both images
    # We use 256 bins for values 0-255
    src_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    ref_hist = cv2.calcHist([reference], [0], None, [256], [0, 256])

    # 2. Calculate Cumulative Distribution Functions (CDF)
    src_cdf = src_hist.cumsum()
    src_cdf_normalized = src_cdf / src_cdf.max()

    ref_cdf = ref_hist.cumsum()
    ref_cdf_normalized = ref_cdf / ref_cdf.max()

    # 3. Create a Lookup Table (LUT)
    # We map each intensity value (0-255) from the source to the reference
    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # Find the pixel value in the reference CDF that is closest to the
        # current value in the source CDF.
        diff = np.abs(ref_cdf_normalized - src_cdf_normalized[i])
        lookup_table[i] = np.argmin(diff)

    # 4. Apply the mapping using OpenCV's LUT function
    matched = cv2.LUT(image, lookup_table)
    return matched


# ─────────────────────────────────────────────────────────────
# Approach A — Middle frame reference (retrospective)
# ─────────────────────────────────────────────────────────────

def histogram_match_serie(serie_paths, reference_path=None):
    """
    [Approach A — Retrospective]
    Matches all frames in a serie to the middle frame's histogram.
    Requires all frames to be available upfront — NOT suitable for real-time.

    Parameters:
        serie_paths   : sorted list of image paths for one serie
        reference_path: optional path override. If None, middle frame is used.

    Returns:
        results  : list of dicts per frame:
                       'path'     : original image path
                       'original' : original grayscale numpy array
                       'matched'  : histogram-matched grayscale numpy array
                       'is_ref'   : True if this frame is the reference
        ref_path : path of the frame used as reference
    """
    ref_path = reference_path if reference_path else get_middle_frame(serie_paths)
    ref_image = load_image(ref_path)
    if ref_image is None:
        raise ValueError(f"Could not load reference image: {ref_path}")

    results = []
    for path in serie_paths:
        image = load_image(path)
        if image is None:
            print(f"  [WARN] Could not load: {path}")
            continue

        is_ref = (Path(path) == Path(ref_path))
        # Use our OpenCV implementation
        matched = ref_image.copy() if is_ref else match_to_reference(image, ref_image)

        results.append({
            'path': path,
            'original': image,
            'matched': matched,
            'is_ref': is_ref,
        })

    return results, ref_path


# ─────────────────────────────────────────────────────────────
# Approach B — Fixed global reference (real-time compatible)
# ─────────────────────────────────────────────────────────────

def compute_global_reference(image_paths, n_samples=50, seed=42):
    """
    [Approach B — helper]
    Computes a global reference image by averaging the histograms of a
    random sample of images across the entire dataset.
    Run this ONCE offline to produce a fixed reference for real-time use.

    Parameters:
        image_paths: full list of dataset image paths
        n_samples  : number of images to sample for averaging (default 50)
        seed       : random seed for reproducibility

    Returns:
        reference: grayscale uint8 numpy array (mean image of the sample)
    """
    rng = np.random.default_rng(seed)
    sampled = rng.choice(image_paths, size=min(n_samples, len(image_paths)), replace=False)

    stack = []
    for path in sampled:
        img = load_image(path)
        if img is not None:
            stack.append(img.astype(np.float32))

    if not stack:
        raise ValueError("No images could be loaded for reference computation.")

    h, w = stack[0].shape
    resized = [img if img.shape == (h, w) else
               cv2.resize(img, (w, h)) for img in stack]

    reference = np.mean(resized, axis=0).astype(np.uint8)
    print(f"Global reference computed from {len(resized)} images.")
    return reference


def histogram_match_realtime(image, global_reference):
    """
    [Approach B — Real-time compatible]
    Matches a single frame to a precomputed global reference histogram.
    Can be applied to frames one at a time without needing the full serie.

    Parameters:
        image           : grayscale uint8 numpy array (single incoming frame)
        global_reference: grayscale uint8 numpy array (precomputed offline)

    Returns:
        Histogram-matched grayscale uint8 numpy array
    """
    return match_to_reference(image, global_reference)

# ─────────────────────────────────────────────────────────────
# Approach C — CLAHE
# ─────────────────────────────────────────────────────────────

def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    # OpenCV requires clipLimit > 0
    limit = max(0.01, clip_limit)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid_size)
    return clahe.apply(image)


def browse_clahe_interactive(series_paths):
    """
    Two-slider widget: One for the frame index and one for the CLAHE intensity.
    """

    def update(frame_idx, clip_val):
        frame_path = series_paths[frame_idx]
        frame = load_image(frame_path)

        if frame is None:
            return

        fname = Path(frame_path).name

        # Apply the selected CLAHE level
        enhanced = apply_clahe(frame, clip_limit=clip_val)

        # Plotting Original vs Enhanced
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].imshow(frame, cmap='gray')
        axes[0].set_title(f"Original Frame: {fname}")

        axes[1].imshow(enhanced, cmap='gray')
        axes[1].set_title(f"CLAHE Enhanced (clip={clip_val:.1f})")

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    # Define the sliders
    frame_slider = IntSlider(
        min=0,
        max=len(series_paths) - 1,
        step=1,
        value=len(series_paths) // 2,  # Start at middle frame
        description='Frame'
    )

    clip_slider = FloatSlider(
        min=0.1,
        max=10.0,
        step=0.1,
        value=2.0,
        description='Clip Limit'
    )

    # Use interact to create the UI
    interact(
        update,
        frame_idx=frame_slider,
        clip_val=clip_slider,
        continuous_update=False  # Performance: only update when you let go of slider
    )
