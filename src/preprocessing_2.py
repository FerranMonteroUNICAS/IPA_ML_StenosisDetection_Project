import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider, VBox, HBox, Output, widgets
from IPython.display import display
from scipy.ndimage import gaussian_filter

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

#######################################################
# BACKGROUND REMOVAL FOR ILUMINATION NORMALIZATION
#######################################################
def apply_background_subtraction(image, sigma=35):
    """
    High-pass filtering via Gaussian background subtraction.
    Removes low-frequency illumination gradients, large structures (ribs, detector artifacts).

    Input:  uint8 or float32 image
    Output: uint8 image (same dtype/range as input)
    """
    was_uint8 = image.dtype == np.uint8

    img = image.astype(np.float32)
    if was_uint8:
        img /= 255.0

    background = gaussian_filter(img, sigma=sigma)
    result = img - background

    # renormalize to [0, 1]
    result -= result.min()
    result /= (result.max() + 1e-8)

    if was_uint8:
        return (result * 255).astype(np.uint8)
    return result


def browse_background_sub_interactive(serie_paths, title="Background Subtraction Explorer"):
    """
    Interactive widget to explore background subtraction (+ optional top-hat)
    before CLAHE, on a single serie.
    """
    print(f"Caching frames for: {title}")
    cache = [load_image(p) for p in serie_paths]
    print(f"Done. {len(cache)} frames loaded.")

    _cache_processed = {}

    def update(frame_idx, sigma_bg, clip_limit):
        frame_idx = min(frame_idx, len(cache) - 1)
        img = cache[frame_idx]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        fname = Path(list(serie_paths)[frame_idx]).name
        key = (frame_idx, sigma_bg, clip_limit)

        if key not in _cache_processed:
            # Step 0 — Background subtraction
            background = gaussian_filter(img, sigma=sigma_bg)
            bg_sub = img - background
            bg_sub -= bg_sub.min()
            bg_sub /= (bg_sub.max() + 1e-8)


            # Step 1 — CLAHE
            from skimage.exposure import equalize_adapthist
            img_uint8 = (bg_sub * 255).astype(np.uint8)
            clahe_out = equalize_adapthist(img_uint8, clip_limit=clip_limit / 100.0)

            _cache_processed[key] = (bg_sub, clahe_out)

        bg_sub, clahe_out = _cache_processed[key]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"{title}  |  {fname}\n"
            f"BG sub σ={sigma_bg} →  CLAHE clip={clip_limit}",
            fontsize=11
        )

        axes[0].imshow(img,      cmap="gray"); axes[0].set_title("① Original");            axes[0].axis("off")
        axes[1].imshow(bg_sub, cmap="gray"); axes[1].set_title("② BG sub");                axes[1].axis("off")
        axes[2].imshow(clahe_out,cmap="gray"); axes[2].set_title("③ CLAHE output");        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    interact(
        update,
        frame_idx=IntSlider(min=0, max=len(cache)-1, step=1, value=0,
                            description="Frame",          continuous_update=False),
        sigma_bg=FloatSlider(min=5, max=60, step=5, value=30,
                             description="σ BG",          continuous_update=False),
        clip_limit=FloatSlider(min=0.5, max=5.0, step=0.5, value=2.0,
                               description="CLAHE clip",  continuous_update=False),
    )



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


def browse_clahe_interactive(series_paths_or_frames, title="Serie"):
    """
    Two-slider widget: One for the frame index and one for the CLAHE intensity.
    Accepts either a list of file paths or a list of already-loaded numpy arrays.
    """
    # ← handle both paths and already-loaded arrays
    is_arrays = isinstance(series_paths_or_frames[0], np.ndarray)

    if is_arrays:
        cache = series_paths_or_frames
    else:
        cache = [load_image(p) for p in series_paths_or_frames]

    def update(frame_idx, clip_val):
        frame_idx = min(frame_idx, len(cache) - 1)
        frame = cache[frame_idx]

        if frame is None:
            return

        # filename only available if we have paths
        fname = "frame_" + str(frame_idx) if is_arrays else Path(series_paths_or_frames[frame_idx]).name

        enhanced = apply_clahe(frame, clip_limit=clip_val)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(title, fontsize=12)

        axes[0].imshow(frame,    cmap='gray'); axes[0].set_title(f"Input: {fname}")
        axes[1].imshow(enhanced, cmap='gray'); axes[1].set_title(f"CLAHE Enhanced (clip={clip_val:.1f})")

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    interact(
        update,
        frame_idx=IntSlider(min=0, max=len(cache)-1, step=1,
                            value=len(cache) // 2, description='Frame'),
        clip_val=FloatSlider(min=0.1, max=10.0, step=0.1,
                             value=2.0, description='Clip Limit'),
        continuous_update=False
    )

# ─────────────────────────────────────────────────────────────
# STEP 2 — DENOISING
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# APPROACH A — BILATERAL FILTERING
# ─────────────────────────────────────────────────────────────
def apply_bilateral(image, d=9, sigma_color=75, sigma_space=75):
    """Bilateral filter - preserves edges, faster"""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
def browse_bilateral_interactive(clahe_frames, title="Serie"):

    def update(frame_idx, d, sigma_color, sigma_space):
        clahe_frame = clahe_frames[frame_idx]
        filtered = apply_bilateral(clahe_frame, d=d,
                                   sigma_color=sigma_color,
                                   sigma_space=sigma_space)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].imshow(clahe_frame, cmap='gray')
        axes[0].set_title(f"CLAHE input (frame {frame_idx})")

        axes[1].imshow(filtered, cmap='gray')
        axes[1].set_title(f"Bilateral  d={d}  σ_color={sigma_color}  σ_space={sigma_space}")

        sharpness = cv2.Laplacian(filtered, cv2.CV_64F).var()
        fig.suptitle(f"Sharpness: {sharpness:.1f}")

        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.show()

    interact(update,
        frame_idx   = IntSlider(min=0, max=len(clahe_frames)-1, step=1,
                                value=len(clahe_frames)//2, description='Frame'),
        d           = IntSlider(min=1, max=25, step=2, value=9,
                                description='d (diameter)', style={'description_width': 'initial'}),
        sigma_color = IntSlider(min=10, max=200, step=5, value=75,
                                description='σ color',     style={'description_width': 'initial'}),
        sigma_space = IntSlider(min=10, max=200, step=5, value=75,
                                description='σ space',     style={'description_width': 'initial'}),
        continuous_update=False)

# ─────────────────────────────────────────────────────────────
# APPROACH b — NON LOCAL MEANS FILTERING
# ─────────────────────────────────────────────────────────────
def apply_nlmeans(image, h=10, template_window=7, search_window=21):
    """Non-Local Means denoising - better quality, slower"""
    return cv2.fastNlMeansDenoising(image, None, h, template_window, search_window)

def browse_nlmeans_interactive(clahe_frames, title="Serie"):

    def update(frame_idx, h, template_window_size, search_window_size):
        clahe_frame = clahe_frames[frame_idx]
        denoised = apply_nlmeans(clahe_frame, h=h,
                                 template_window=template_window_size,
                                 search_window=search_window_size)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].imshow(clahe_frame, cmap='gray')
        axes[0].set_title(f"CLAHE input (frame {frame_idx})")

        axes[1].imshow(denoised, cmap='gray')
        axes[1].set_title(f"NL-Means  h={h}  tpl={template_window_size}  srch={search_window_size}")

        sharpness = cv2.Laplacian(denoised, cv2.CV_64F).var()
        fig.suptitle(f"Sharpness: {sharpness:.1f}")

        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.show()

    interact(update,
        frame_idx            = IntSlider(min=0, max=len(clahe_frames)-1, step=1,
                                         value=len(clahe_frames)//2, description='Frame'),
        h                    = IntSlider(min=1, max=30, step=1, value=10,
                                         description='h (strength)', style={'description_width': 'initial'}),
        template_window_size = IntSlider(min=3, max=15, step=2, value=7,
                                         description='Template win', style={'description_width': 'initial'}),
        search_window_size   = IntSlider(min=11, max=41, step=2, value=21,
                                         description='Search win',   style={'description_width': 'initial'}),
        continuous_update=False)
