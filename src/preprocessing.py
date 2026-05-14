import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider, VBox, HBox
import random

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
# Histogram matching functions --> NOT CURRENTLY USED
# ─────────────────────────────────────────────────────────────
# Matching function (core) --> NOT CURRENTLY USED
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

# Middle frame reference (retrospective) --> NOT CURRENTLY USED
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

# Fixed global reference (real-time compatible) --> NOT CURRENTLY USED
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
# CLAHE
# ─────────────────────────────────────────────────────────────
def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    # OpenCV requires clipLimit > 0
    limit = max(0.01, clip_limit)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid_size)
    return clahe.apply(image)

def browse_clahe_interactive(series_paths):
    """
    Two-slider widget to explore CLAHE on a single serie:
    one slider for the frame index, one for the clip limit.
    """
    def update(frame_idx, clip_val):
        frame = load_image(series_paths[frame_idx])
        if frame is None:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(frame, cmap='gray')
        axes[0].set_title(f"Original — {Path(series_paths[frame_idx]).name}")
        axes[1].imshow(apply_clahe(frame, clip_limit=clip_val), cmap='gray')
        axes[1].set_title(f"CLAHE (clip={clip_val:.1f})")
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    interact(
        update,
        frame_idx=IntSlider(min=0, max=len(series_paths)-1, step=1,
                            value=len(series_paths)//2, description='Frame'),
        clip_val=FloatSlider(min=0.1, max=10.0, step=0.1, value=2.0,
                             description='Clip Limit'),
        continuous_update=False
    )

def browse_clahe_grid(series_dict, n=9, seed=42):
    """
    3×3 grid with one clip limit slider.
    Shows CLAHE result for n randomly selected patients (one frame each).
    """
    loaded = _sample_patient_frames(series_dict, n, seed)

    def update(clip_val):
        rows = int(np.ceil(len(loaded) / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(3 * 3.5, rows * 3.5))
        axes = axes.flatten()
        for i, (pid, img) in enumerate(loaded):
            axes[i].imshow(apply_clahe(img, clip_limit=clip_val), cmap='gray')
            axes[i].set_title(f"P{pid}", fontsize=8)
            axes[i].axis('off')
        for j in range(len(loaded), len(axes)):
            axes[j].axis('off')
        plt.suptitle(f"CLAHE grid — clip={clip_val:.1f}", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.show()

    interact(
        update,
        clip_val=FloatSlider(min=0.1, max=10.0, step=0.1, value=2.0,
                             description='Clip limit',
                             style={'description_width': 'initial'},
                             layout={'width': '500px'}),
        continuous_update=False
    )

# ─────────────────────────────────────────────────────────────
# Background subtraction
# ─────────────────────────────────────────────────────────────
def subtract_background(image, kernel_size=31):
    """
    Removes slow-varying background illumination using a large Gaussian blur.

        background = GaussianBlur(image, kernel_size)
        residual   = image - background
        output     = rescaled residual to [0, 255]

    Parameters:
        image      : grayscale uint8 numpy array
        kernel_size: Gaussian kernel size (must be odd; default 41)

    Returns:
        Background-subtracted grayscale uint8 numpy array
    """
    # Ensure kernel size is odd
    ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    # Estimate background as the low-frequency component
    background = cv2.GaussianBlur(image.astype(np.float32), (ks, ks), 0)
    #background = cv2.medianBlur(image.astype(np.float32), ks)

    # Subtract and shift to avoid negative values
    residual = image.astype(np.float32) - background

    # Rescale to full [0, 255] uint8 range
    r_min, r_max = residual.min(), residual.max()
    if r_max > r_min:
        rescaled = ((residual - r_min) / (r_max - r_min) * 255).astype(np.uint8)
    else:
        rescaled = np.zeros_like(image)

    return rescaled

def browse_background_grid(series_dict, n=9, seed=42):
    """
    3×3 grid with one kernel size slider.
    Shows background-subtracted result for n randomly selected patients.
    """
    loaded = _sample_patient_frames(series_dict, n, seed)

    def update(kernel_size):
        rows = int(np.ceil(len(loaded) / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(3 * 3.5, rows * 3.5))
        axes = axes.flatten()
        for i, (pid, img) in enumerate(loaded):
            axes[i].imshow(subtract_background(img, kernel_size=kernel_size), cmap='gray')
            axes[i].set_title(f"P{pid}", fontsize=8)
            axes[i].axis('off')
        for j in range(len(loaded), len(axes)):
            axes[j].axis('off')
        plt.suptitle(f"BG subtraction grid — k={kernel_size}px", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.show()

    interact(
        update,
        kernel_size=IntSlider(min=1, max=201, step=10, value=21,
                              description='Kernel size',
                              style={'description_width': 'initial'},
                              layout={'width': '500px'}),
        continuous_update=False
    )

def browse_bg_then_clahe_grid(series_dict, kernel_size=41, n=9, seed=42):
    """
    3×3 grid with a single CLAHE clip limit slider.
    Background subtraction runs once at load time with the fixed kernel_size;
    only CLAHE is recomputed on each slider change.
    Shows only the final BG+CLAHE result.

    Parameters:
        series_dict : output of group_paths_by_serie
        kernel_size : fixed Gaussian kernel for BG subtraction (set manually)
        n           : number of patients to show (default 9)
        seed        : random seed for reproducibility
    """
    # BG subtraction is fixed — precompute once
    loaded = [(pid, subtract_background(img, kernel_size=kernel_size))
              for pid, img in _sample_patient_frames(series_dict, n, seed)]

    def update(clip_val):
        rows = int(np.ceil(len(loaded) / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(3 * 3.5, rows * 3.5))
        axes = axes.flatten()
        for i, (pid, bg_sub) in enumerate(loaded):
            axes[i].imshow(apply_clahe(bg_sub, clip_limit=clip_val), cmap='gray')
            axes[i].set_title(f"P{pid}", fontsize=8)
            axes[i].axis('off')
        for j in range(len(loaded), len(axes)):
            axes[j].axis('off')
        plt.suptitle(f"BG sub (k={kernel_size}) → CLAHE (clip={clip_val:.1f})",
                     fontsize=13, y=1.01)
        plt.tight_layout()
        plt.show()

    interact(
        update,
        clip_val=FloatSlider(min=0.0, max=10.0, step=0.5, value=2.0,
                             description='Clip limit',
                             style={'description_width': 'initial'},
                             layout={'width': '450px'}),
        continuous_update=False
    )

# ─────────────────────────────────────────────────────────────
# NLM filtering
# ─────────────────────────────────────────────────────────────
def apply_nlm(image, h=10, template_window=7, search_window=21):
    """
    Applies Non-Local Means denoising to a grayscale image.

    NLM suppresses background noise while preserving thin vessel edges
    by averaging pixels with similar patch neighborhoods across a large
    search window, rather than just looking at local neighbors.

    Parameters:
        image          : grayscale uint8 numpy array
        h              : filter strength — higher = more smoothing, more detail loss
                         (default 10; try 5–15 for angiographic images)
        template_window: size of the patch used for similarity comparison (must be odd)
                         (default 7)
        search_window  : size of the area searched for similar patches (must be odd)
                         (default 21)

    Returns:
        Denoised grayscale uint8 numpy array
    """
    return cv2.fastNlMeansDenoising(
        image, None,
        h=h,
        templateWindowSize=template_window,
        searchWindowSize=search_window
    )

def browse_nlm_grid(series_dict, kernel_size=41, clip_limit=2.0, n=9, seed=42):
    """
    3×3 grid with three NLM sliders:
        - h              : filter strength — main quality control parameter.
                           Higher = more smoothing, more detail loss.
        - template window: patch size for similarity comparison (must be odd).
                           Larger = more robust but slower. Usually left at 7.
        - search window  : area searched for similar patches (must be odd).
                           Larger = better quality but much slower. Usually left at 21.

    BG subtraction and CLAHE are fixed and precomputed once at load time;
    only NLM is recomputed on each slider change.

    Parameters:
        series_dict : output of group_paths_by_serie
        kernel_size : fixed BG subtraction kernel (set manually)
        clip_limit  : fixed CLAHE clip limit (set manually)
        n           : number of patients to show (default 9)
        seed        : random seed for reproducibility
    """
    # Precompute BG sub + CLAHE once — only NLM changes with the sliders
    loaded = [
        (pid, apply_clahe(subtract_background(img, kernel_size=kernel_size),
                          clip_limit=clip_limit))
        for pid, img in _sample_patient_frames(series_dict, n, seed)
    ]

    def update(h, template_window, search_window):
        rows = int(np.ceil(len(loaded) / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(3 * 3.5, rows * 3.5))
        axes = axes.flatten()
        for i, (pid, clahe_img) in enumerate(loaded):
            axes[i].imshow(
                apply_nlm(clahe_img, h=h,
                          template_window=template_window,
                          search_window=search_window),
                cmap='gray'
            )
            axes[i].set_title(f"P{pid}", fontsize=8)
            axes[i].axis('off')
        for j in range(len(loaded), len(axes)):
            axes[j].axis('off')
        plt.suptitle(
            f"BG (k={kernel_size}) → CLAHE (clip={clip_limit}) → "
            f"NLM (h={h}, tw={template_window}, sw={search_window})",
            fontsize=11, y=1.01
        )
        plt.tight_layout()
        plt.show()

    slider_style  = {'description_width': 'initial'}
    slider_layout = {'width': '450px'}

    interact(
        update,
        h=IntSlider(min=1, max=30, step=1, value=10,
                    description='h (filter strength)',
                    style=slider_style, layout=slider_layout),
        template_window=IntSlider(min=3, max=15, step=2, value=7,
                                  description='Template window',
                                  style=slider_style, layout=slider_layout),
        search_window=IntSlider(min=11, max=41, step=2, value=21,
                                description='Search window',
                                style=slider_style, layout=slider_layout),
        continuous_update=False
    )

# ─────────────────────────────────────────────────────────────
# Bilateral filtering
# ─────────────────────────────────────────────────────────────
def apply_bilateral(image, d=9, sigma_color=75, sigma_space=75):
    """
    Applies bilateral filtering to a grayscale image.

    Unlike Gaussian blur, bilateral filtering preserves edges by weighting
    neighbors both by spatial distance (sigma_space) and intensity similarity
    (sigma_color), so pixels across a strong edge don't contribute to each other.

    Parameters:
        image      : grayscale uint8 numpy array
        d          : diameter of the pixel neighborhood (default 9).
                     If negative, it is computed from sigma_space.
        sigma_color: intensity range — how large an intensity difference is still
                     considered "similar". Higher = smoother, more edge blurring.
                     (default 75; try 25–150)
        sigma_space: spatial range — how far away pixels can be and still influence
                     the result. Higher = larger neighborhood.
                     (default 75; try 25–150)

    Returns:
        Filtered grayscale uint8 numpy array
    """
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

def browse_bilateral_grid(series_dict, kernel_size=41, clip_limit=2.0, n=9, seed=42):
    """
    3×3 grid with three bilateral filter sliders:
        - d           : neighborhood diameter
        - sigma_color : intensity similarity range
        - sigma_space : spatial range

    BG subtraction and CLAHE are fixed and precomputed once at load time;
    only bilateral filtering is recomputed on each slider change.

    Parameters:
        series_dict : output of group_paths_by_serie
        kernel_size : fixed BG subtraction kernel (set manually)
        clip_limit  : fixed CLAHE clip limit (set manually)
        n           : number of patients to show (default 9)
        seed        : random seed for reproducibility
    """
    # Precompute BG sub + CLAHE once
    loaded = [
        (pid, apply_clahe(subtract_background(img, kernel_size=kernel_size),
                          clip_limit=clip_limit))
        for pid, img in _sample_patient_frames(series_dict, n, seed)
    ]

    def update(d, sigma_color, sigma_space):
        rows = int(np.ceil(len(loaded) / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(3 * 3.5, rows * 3.5))
        axes = axes.flatten()
        for i, (pid, clahe_img) in enumerate(loaded):
            axes[i].imshow(
                apply_bilateral(clahe_img, d=d,
                                sigma_color=sigma_color,
                                sigma_space=sigma_space),
                cmap='gray'
            )
            axes[i].set_title(f"P{pid}", fontsize=8)
            axes[i].axis('off')
        for j in range(len(loaded), len(axes)):
            axes[j].axis('off')
        plt.suptitle(
            f"BG (k={kernel_size}) → CLAHE (clip={clip_limit}) → "
            f"Bilateral (d={d}, σ_c={sigma_color}, σ_s={sigma_space})",
            fontsize=11, y=1.01
        )
        plt.tight_layout()
        plt.show()

    slider_style  = {'description_width': 'initial'}
    slider_layout = {'width': '450px'}

    interact(
        update,
        d=IntSlider(min=3, max=25, step=2, value=9,
                    description='d (diameter)',
                    style=slider_style, layout=slider_layout),
        sigma_color=IntSlider(min=10, max=200, step=5, value=75,
                              description='sigma_color',
                              style=slider_style, layout=slider_layout),
        sigma_space=IntSlider(min=10, max=200, step=5, value=75,
                              description='sigma_space',
                              style=slider_style, layout=slider_layout),
        continuous_update=False
    )


# ─────────────────────────────────────────────────────────────
# Grid export
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


# ─────────────────────────────────────────────────────────────
# Reordered pipeline: BG subtraction → NLM → CLAHE
# ─────────────────────────────────────────────────────────────

def browse_bg_nlm_clahe_grid(series_dict, n=9, seed=42):
    """
    Interactive 3×3 grid for the reordered pipeline:
        BG subtraction → NLM → CLAHE

    NLM denoises before CLAHE so that contrast enhancement acts on a
    cleaner signal, reducing noise amplification.

    Three sliders:
        - Kernel size (ks) : BG subtraction aggressiveness.
                             Too small → embossing artifact (vessels partially subtracted).
                             Too large → background not fully removed.
                             Recommended starting point: 41–61.
        - h                : NLM filter strength.
                             Higher = smoother background, more detail loss.
                             Recommended starting point: 5–15.
        - Clip limit       : CLAHE contrast enhancement strength.
                             Higher = more local contrast, more noise amplification.
                             Recommended starting point: 1.0–3.0.

    Note: all three steps are recomputed on every slider change since
    all parameters are variable. Expect slightly slower updates than
    the fixed-precompute browse functions.

    Parameters:
        series_dict : output of group_paths_by_serie
        n           : number of patients in the grid (default 9)
        seed        : random seed for reproducibility
    """
    # Load raw images once — only reloading never happens
    loaded = _sample_patient_frames(series_dict, n, seed)

    def update(kernel_size, h, clip_val):
        rows = int(np.ceil(len(loaded) / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(3 * 3.5, rows * 3.5))
        axes = axes.flatten()

        for i, (pid, img) in enumerate(loaded):
            bg_sub = subtract_background(img, kernel_size=kernel_size)
            denoised = apply_nlm(bg_sub, h=h)
            result = apply_clahe(denoised, clip_limit=clip_val)

            axes[i].imshow(result, cmap='gray')
            axes[i].set_title(f"P{pid}", fontsize=8)
            axes[i].axis('off')

        for j in range(len(loaded), len(axes)):
            axes[j].axis('off')

        plt.suptitle(
            f"BG sub (k={kernel_size}) → NLM (h={h}) → CLAHE (clip={clip_val:.1f})",
            fontsize=12, y=1.01
        )
        plt.tight_layout()
        plt.show()

    slider_style = {'description_width': 'initial'}
    slider_layout = {'width': '450px'}

    interact(
        update,
        kernel_size=IntSlider(min=11, max=101, step=2, value=41,
                              description='Kernel size (ks)',
                              style=slider_style, layout=slider_layout),
        h=IntSlider(min=1, max=30, step=1, value=10,
                    description='NLM strength (h)',
                    style=slider_style, layout=slider_layout),
        clip_val=FloatSlider(min=0.1, max=10.0, step=0.1, value=2.0,
                             description='Clip limit',
                             style=slider_style, layout=slider_layout),
        continuous_update=False
    )
