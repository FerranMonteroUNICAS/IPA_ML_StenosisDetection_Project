import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from pathlib import Path
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown
from skimage.filters import frangi, sato, meijering, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects

from src.utils import load_image
from src.preprocessing import preprocess_bs_nlm_clahe

# ─────────────────────────────────────────────────────────────
# Shared patient sampler (mirrors preprocessing.py)
# ─────────────────────────────────────────────────────────────
def _sample_patient_frames(series_dict, n=9, seed=42):
    """
    Selects one middle frame per patient, samples n patients randomly.
    Returns list of (patient_id, loaded_image) tuples.
    """
    rng = random.Random(seed)

    patients = defaultdict(list)
    for key, frame_paths in series_dict.items():
        patient_id = key.split("/")[0]
        mid = frame_paths[len(frame_paths) // 2]
        patients[patient_id].append(mid)

    patient_ids = sorted(patients.keys())
    selected    = rng.sample(patient_ids, min(n, len(patient_ids)))

    loaded = []
    for pid in selected:
        path = rng.choice(patients[pid])
        img  = load_image(path)
        if img is not None:
            loaded.append((pid, img))

    return loaded

# ─────────────────────────────────────────────────────────────
# Ridge filters
# ─────────────────────────────────────────────────────────────
def apply_frangi(image, scale_min=1, scale_max=5, scale_step=1,
                 alpha=0.5, beta=0.5, black_ridges=True):
    """
    Frangi vesselness filter — detects tubular structures by analysing
    the eigenvalues of the Hessian matrix at multiple scales.

    Parameters:
        image      : grayscale uint8 numpy array
        scale_min  : minimum vessel width scale in pixels (default 1)
        scale_max  : maximum vessel width scale in pixels (default 5)
        scale_step : step between scales (default 1)
        alpha      : sensitivity to plate-like structures (default 0.5)
        beta       : sensitivity to blob-like structures (default 0.5)
        black_ridges: True for dark vessels on bright background (default True)

    Returns:
        Vesselness map as float32 numpy array in [0, 1]
    """
    sigmas = np.arange(scale_min, scale_max + scale_step, scale_step)
    result = frangi(image.astype(np.float32) / 255.0,
                    sigmas=sigmas,
                    alpha=alpha,
                    beta=beta,
                    black_ridges=black_ridges)
    return result.astype(np.float32)

def apply_sato(image, scale_min=1, scale_max=5, scale_step=1, black_ridges=True):
    """
    Sato tubeness filter — uses only the largest Hessian eigenvalue,
    making it more robust to noise than Frangi at the cost of some specificity.

    Parameters:
        image      : grayscale uint8 numpy array
        scale_min  : minimum vessel width scale in pixels (default 1)
        scale_max  : maximum vessel width scale in pixels (default 5)
        scale_step : step between scales (default 1)
        black_ridges: True for dark vessels on bright background (default True)

    Returns:
        Tubeness map as float32 numpy array in [0, 1]
    """
    sigmas = np.arange(scale_min, scale_max + scale_step, scale_step)
    result = sato(image.astype(np.float32) / 255.0,
                  sigmas=sigmas,
                  black_ridges=black_ridges)
    # Normalise to [0, 1]
    r_max = result.max()
    return (result / r_max).astype(np.float32) if r_max > 0 else result.astype(np.float32)

def apply_meijering(image, scale_min=1, scale_max=5, scale_step=1, black_ridges=True):
    """
    Meijering neuriteness filter — variant of Frangi optimised for thin,
    elongated structures. Often better than Frangi for very fine vessels.

    Parameters:
        image      : grayscale uint8 numpy array
        scale_min  : minimum vessel width scale in pixels (default 1)
        scale_max  : maximum vessel width scale in pixels (default 5)
        scale_step : step between scales (default 1)
        black_ridges: True for dark vessels on bright background (default True)

    Returns:
        Neuriteness map as float32 numpy array in [0, 1]
    """
    sigmas = np.arange(scale_min, scale_max + scale_step, scale_step)
    result = meijering(image.astype(np.float32) / 255.0,
                       sigmas=sigmas,
                       black_ridges=black_ridges)
    r_max = result.max()
    return (result / r_max).astype(np.float32) if r_max > 0 else result.astype(np.float32)

# ─────────────────────────────────────────────────────────────
# Filter dispatcher
# ─────────────────────────────────────────────────────────────
FILTER_MAP = {
    "frangi": apply_frangi,
    "sato": apply_sato,
    "meijering": apply_meijering,
}

# ─────────────────────────────────────────────────────────────
# Interactive grid
# ─────────────────────────────────────────────────────────────
def browse_ridge_grid(series_dict,
                      kernel_size=41, h=10, clip_limit=2.0,
                      n=9, seed=42):
    """
    Interactive 3×3 grid for exploring ridge filters on preprocessed images.

    Preprocessing (BG sub → NLM → CLAHE) is fixed at load time using the
    provided parameters. Only the filter type and scale range are interactive.

    Sliders / dropdowns:
        - Filter     : choose between frangi, sato, meijering
        - Scale min  : smallest vessel width searched (pixels)
        - Scale max  : largest vessel width searched (pixels)
        - black_ridges: toggle for dark vs bright vessels

    Parameters:
        series_dict : output of group_paths_by_serie
        kernel_size : BG subtraction kernel (fixed, set manually)
        h           : NLM filter strength (fixed, set manually)
        clip_limit  : CLAHE clip limit (fixed, set manually)
        n           : number of patients in the grid (default 9)
        seed        : random seed — use same as other browse functions
    """
    # Preprocess once — filter changes are the only variable
    raw     = _sample_patient_frames(series_dict, n=n, seed=seed)
    loaded  = [(pid, preprocess_bs_nlm_clahe(img,
                                              kernel_size=kernel_size,
                                              h=h,
                                              clip_limit=clip_limit))
               for pid, img in raw]

    def update(filter_type, scale_min, scale_max, black_ridges):
        if scale_min >= scale_max:
            print("scale_min must be less than scale_max")
            return

        filter_fn = FILTER_MAP[filter_type]
        rows = int(np.ceil(len(loaded) / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(3 * 3.5, rows * 3.5))
        axes = axes.flatten()

        for i, (pid, proc) in enumerate(loaded):
            result = filter_fn(proc,
                               scale_min=scale_min,
                               scale_max=scale_max,
                               black_ridges=black_ridges)
            axes[i].imshow(result, cmap='gray')
            axes[i].set_title(f"P{pid}", fontsize=8)
            axes[i].axis('off')

        for j in range(len(loaded), len(axes)):
            axes[j].axis('off')

        plt.suptitle(
            f"{filter_type.capitalize()}  |  scales {scale_min}–{scale_max}  |  "
            f"black_ridges={black_ridges}  |  "
            f"preproc: k={kernel_size}, h={h}, clip={clip_limit}",
            fontsize=10, y=1.01
        )
        plt.tight_layout()
        plt.show()

    slider_style  = {'description_width': 'initial'}
    slider_layout = {'width': '450px'}

    interact(
        update,
        filter_type=Dropdown(
            options=["frangi", "sato", "meijering"],
            value="frangi",
            description="Filter",
            style=slider_style,
            layout=slider_layout
        ),
        scale_min=IntSlider(min=1, max=10, step=1, value=4,
                            description='Scale min (px)',
                            style=slider_style, layout=slider_layout),
        scale_max=IntSlider(min=2, max=30, step=1, value=15,
                            description='Scale max (px)',
                            style=slider_style, layout=slider_layout),
        black_ridges=Dropdown(
            options=[True, False],
            value=True,
            description='Black ridges',
            style=slider_style,
            layout=slider_layout
        ),
        continuous_update=False
    )

# ─────────────────────────────────────────────────────────────
# Hysteresis Thresholding & Visualization
# ─────────────────────────────────────────────────────────────

def apply_hysteresis(frangi_image, low_thresh=0.05, high_thresh=0.25):
    """
    Applies a dual-threshold hysteresis operation to binarize the vesselness map.
    Pixels above high_thresh are certain vessels. Connected pixels above low_thresh
    are preserved to bridge gaps.
    """
    mask = apply_hysteresis_threshold(frangi_image, low=low_thresh, high=high_thresh)
    return mask

def browse_hysteresis_grid(series_dict,
                           kernel_size=81, h=8, clip_limit=4.0,
                           scale_min=4, scale_max=15,
                           n=9, seed=42):
    """
    Interactive 3×3 grid for tuning Hysteresis thresholds on a 9-patient panel.
    Precomputes the preprocessing and Frangi stages once for smooth widget response.
    """
    print("Precomputing Frangi outputs for the grid panel... please wait.")
    raw = _sample_patient_frames(series_dict, n=n, seed=seed)

    # Precompute up to the Frangi filter step so slider updates are instantaneous
    loaded_frangi = []
    for pid, img in raw:
        proc = preprocess_bs_nlm_clahe(img, kernel_size=kernel_size, h=h, clip_limit=clip_limit)
        frangi_map = apply_frangi(proc, scale_min=scale_min, scale_max=scale_max, black_ridges=True)
        loaded_frangi.append((pid, frangi_map))

    print("Precomputation finished! Loading grid...")

    def update(low_thresh, high_thresh):
        if low_thresh >= high_thresh:
            print("⚠️ Error: Low threshold must be strictly less than High threshold.")
            return

        rows = int(np.ceil(len(loaded_frangi) / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(3 * 3.5, rows * 3.5))
        axes = axes.flatten()

        for i, (pid, frangi_img) in enumerate(loaded_frangi):
            binary_mask = apply_hysteresis(frangi_img, low_thresh=low_thresh, high_thresh=high_thresh)

            # Displaying as binary (black and white)
            axes[i].imshow(binary_mask, cmap='gray')
            axes[i].set_title(f"P{pid}", fontsize=8)
            axes[i].axis('off')

        for j in range(len(loaded_frangi), len(axes)):
            axes[j].axis('off')

        plt.suptitle(
            f"Hysteresis Mask  |  Low={low_thresh:.2f}, High={high_thresh:.2f}\n"
            f"Fixed Frangi scales: {scale_min}–{scale_max} px",
            fontsize=11, y=1.02
        )
        plt.tight_layout()
        plt.show()

    slider_style = {'description_width': 'initial'}
    slider_layout = {'width': '450px'}

    interact(
        update,
        high_thresh=FloatSlider(min=0.05, max=0.60, step=0.01, value=0.20,
                                description='High Thresh (Seeds)',
                                style=slider_style, layout=slider_layout),
        low_thresh=FloatSlider(min=0.01, max=0.20, step=0.01, value=0.05,
                               description='Low Thresh (Bridges)',
                               style=slider_style, layout=slider_layout),
        continuous_update=False
    )

# ─────────────────────────────────────────────────────────────
# Clea small objects after hysteresis thresholding
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# Clean Hysteresis with Area Filtering
# ─────────────────────────────────────────────────────────────

def apply_cleaned_hysteresis(frangi_image, low_thresh=0.05, high_thresh=0.25, min_size=200):
    """
    Applies hysteresis thresholding and automatically deletes any isolated
    binary structures/blobs containing fewer than `min_size` pixels.
    """
    # 1. Standard dual-threshold binarization
    mask = apply_hysteresis_threshold(frangi_image, low=low_thresh, high=high_thresh)

    # 2. Size filtering to eliminate background noise components
    cleaned_mask = remove_small_objects(mask, min_size=min_size)

    return cleaned_mask


def browse_clean_hysteresis_grid(series_dict,
                                 kernel_size=81, h=8, clip_limit=4.0,
                                 scale_min=4, scale_max=15,
                                 n=9, seed=42):
    """
    Interactive 3×3 grid for tuning Hysteresis thresholds combined with
    a connected component minimum size noise filter.
    """
    print("Precomputing Frangi outputs for the grid panel... please wait.")
    raw = _sample_patient_frames(series_dict, n=n, seed=seed)

    loaded_frangi = []
    for pid, img in raw:
        proc = preprocess_bs_nlm_clahe(img, kernel_size=kernel_size, h=h, clip_limit=clip_limit)
        frangi_map = apply_frangi(proc, scale_min=scale_min, scale_max=scale_max, black_ridges=True)
        loaded_frangi.append((pid, frangi_map))

    print("Precomputation finished! Loading grid...")

    def update(low_thresh, high_thresh, min_size):
        if low_thresh >= high_thresh:
            print("⚠️ Error: Low threshold must be strictly less than High threshold.")
            return

        rows = int(np.ceil(len(loaded_frangi) / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(3 * 3.5, rows * 3.5))
        axes = axes.flatten()

        for i, (pid, frangi_img) in enumerate(loaded_frangi):
            # Apply our noise-filtering binarization pipeline
            binary_mask = apply_cleaned_hysteresis(
                frangi_img,
                low_thresh=low_thresh,
                high_thresh=high_thresh,
                min_size=min_size
            )

            axes[i].imshow(binary_mask, cmap='gray')
            axes[i].set_title(f"P{pid}", fontsize=8)
            axes[i].axis('off')

        for j in range(len(loaded_frangi), len(axes)):
            axes[j].axis('off')

        plt.suptitle(
            f"Low={low_thresh:.2f}, High={high_thresh:.2f} | Noise Filter (min_size={min_size}px)",
            fontsize=11, y=1.02
        )
        plt.tight_layout()
        plt.show()

    slider_style = {'description_width': 'initial'}
    slider_layout = {'width': '450px'}

    interact(
        update,
        high_thresh=FloatSlider(min=0.05, max=0.60, step=0.01, value=0.18,
                                description='High Thresh (Seeds)',
                                style=slider_style, layout=slider_layout),
        low_thresh=FloatSlider(min=0.01, max=0.20, step=0.01, value=0.02,
                               description='Low Thresh (Bridges)',
                               style=slider_style, layout=slider_layout),
        min_size=IntSlider(min=10, max=1000, step=20, value=200,
                           description='Min Object Size (px)',
                           style=slider_style, layout=slider_layout),
        continuous_update=False
    )