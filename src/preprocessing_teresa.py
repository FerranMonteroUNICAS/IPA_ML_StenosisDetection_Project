import numpy as np
import cv2
from scipy.stats import entropy as scipy_entropy
from pathlib import Path
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Dropdown
import random
from src.utils import *
from src.preprocessing import *
from collections import defaultdict
from src.preprocessing import _sample_patient_frames
from src.preprocessing import subtract_background
from src.preprocessing import (
    _sample_patient_frames,
    subtract_background,
    histogram_match_realtime
)

from skimage.filters import frangi
from skimage.filters import sato
from skimage.morphology import skeletonize
from scipy.ndimage import binary_fill_holes
from skimage import exposure

from ipywidgets import interact
from ipywidgets import IntSlider
from ipywidgets import FloatSlider

from src.utils import load_image

# ─────────────────────────────────────────────────────────────
# Quality metrics
# ─────────────────────────────────────────────────────────────

def shannon_entropy(image):
    """
    Shannon entropy of the pixel intensity histogram.

    Measures how spread the histogram is across the 256 bins.
    A perfectly flat histogram (maximum information) = log2(256) ≈ 8 bits.
    A narrow, peaked histogram (low contrast) → low entropy.

    Parameters:
        image : grayscale uint8 numpy array

    Returns:
        float — entropy in bits (0–8)
    """
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    hist = hist[hist > 0].astype(np.float64)
    prob = hist / hist.sum()
    return float(-np.sum(prob * np.log2(prob)))


def laplacian_variance(image):
    """
    Variance of the Laplacian — a fast sharpness / focus measure.

    The Laplacian highlights regions of rapid intensity change (edges).
    High variance → image has many strong edges → well-focused, sharp.
    Low variance → image is blurry or over-smoothed.

    Use this alongside entropy to avoid selecting a clip_limit that
    boosts contrast but blurs fine vessel structure.

    Parameters:
        image : grayscale uint8 numpy array

    Returns:
        float — Laplacian variance (higher = sharper)
    """
    lap = cv2.Laplacian(image, cv2.CV_64F)
    return float(lap.var())


def tenengrad(image, ksize=3):
    """
    Tenengrad focus measure based on the Sobel gradient magnitude.

    More robust to noise than Laplacian variance because it uses
    directional gradients (Gx, Gy) rather than a second derivative.
    Useful as an alternative sharpness metric.

    Parameters:
        image : grayscale uint8 numpy array
        ksize : Sobel kernel size (default 3)

    Returns:
        float — mean squared gradient magnitude (higher = sharper)
    """
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    return float(np.mean(gx**2 + gy**2))


def composite_score(image, alpha=0.6, beta=0.4):
    """
    Weighted combination of normalised entropy and normalised sharpness
    (Tenengrad). Balances contrast enhancement against detail preservation.

    Score = alpha * H_norm + beta * T_norm

    where H_norm and T_norm are each independently min-max normalised
    across the clip_limit sweep for a single image (see find_optimal_clip).

    Parameters:
        image : grayscale uint8 numpy array
        alpha : weight for entropy   (default 0.6)
        beta  : weight for tenengrad (default 0.4)

    Returns:
        tuple (entropy_val, tenengrad_val) — raw values before normalisation.
        Normalisation happens inside find_optimal_clip.
    """
    return shannon_entropy(image), tenengrad(image)


# ─────────────────────────────────────────────────────────────
# Core: find optimal clip limit for a single image
# ─────────────────────────────────────────────────────────────

METRIC_FUNCTIONS = {
    "entropy":    shannon_entropy,
    "sharpness":  laplacian_variance,
    "tenengrad":  tenengrad,
    "composite":  None,           # handled separately
}

def find_optimal_clip(
    image,
    clip_range=(0.5, 4.0),
    n_steps=20,
    grid_size=(8, 8),
    metric="composite",
    alpha=0.6,
    beta=0.4,
):
    """
    Sweeps clip_limit values and selects the one that maximises the
    chosen quality metric after CLAHE is applied to `image`.

    Parameters:
        image      : grayscale uint8 numpy array (typically the middle frame
                     of a serie after background subtraction)
        clip_range : (min, max) of the clip_limit sweep
        n_steps    : number of candidate values to evaluate
        grid_size  : CLAHE tile grid size (default (8, 8))
        metric     : one of "entropy", "sharpness", "tenengrad", "composite"
        alpha      : composite weight for entropy   (ignored for other metrics)
        beta       : composite weight for tenengrad (ignored for other metrics)

    Returns:
        dict with keys:
            "optimal_clip"   : float — best clip_limit found
            "clips"          : list of clip values evaluated
            "scores"         : list of metric scores (same order as clips)
            "metric"         : name of metric used
    """
    clips  = np.linspace(clip_range[0], clip_range[1], n_steps)
    scores = []

    if metric == "composite":
        raw_entropy   = []
        raw_tenengrad = []
        for c in clips:
            clahe = cv2.createCLAHE(clipLimit=float(c), tileGridSize=grid_size)
            enhanced = clahe.apply(image)
            raw_entropy.append(shannon_entropy(enhanced))
            raw_tenengrad.append(tenengrad(enhanced))

        # Min-max normalise each component independently
        e_arr = np.array(raw_entropy)
        t_arr = np.array(raw_tenengrad)
        e_norm = (e_arr - e_arr.min()) / (np.ptp(e_arr) + 1e-9)
        t_norm = (t_arr - t_arr.min()) / (np.ptp(t_arr) + 1e-9)
        scores = list(alpha * e_norm + beta * t_norm)

    else:
        fn = METRIC_FUNCTIONS[metric]
        for c in clips:
            clahe = cv2.createCLAHE(clipLimit=float(c), tileGridSize=grid_size)
            enhanced = clahe.apply(image)
            scores.append(fn(enhanced))

    best_idx = int(np.argmax(scores))
    return {
        "optimal_clip": float(clips[best_idx]),
        "clips":        list(clips),
        "scores":       scores,
        "metric":       metric,
    }


# ─────────────────────────────────────────────────────────────
# Per-serie optimal clip limit
# ─────────────────────────────────────────────────────────────

def compute_optimal_clips_for_series(
    series_dict,
    load_fn,
    bg_subtract_fn,
    kernel_size=41,
    clip_range=(0.5, 8.0),
    n_steps=20,
    grid_size=(8, 8),
    metric="composite",
    alpha=0.6,
    beta=0.4,
    verbose=True,
):
    """
    Computes one optimal clip_limit per serie by evaluating the middle frame.

    Workflow per serie:
        1. Load the middle frame
        2. Apply background subtraction (fixed kernel_size)
        3. Sweep clip_limit values, score each with the chosen metric
        4. Return the best clip_limit

    Parameters:
        series_dict    : output of group_paths_by_serie
        load_fn        : callable(path) → grayscale uint8 array or None
        bg_subtract_fn : callable(image, kernel_size) → uint8 array
        kernel_size    : Gaussian kernel for BG subtraction
        clip_range     : (min, max) clip_limit sweep range
        n_steps        : number of candidate values
        grid_size      : CLAHE tile grid size
        metric         : quality metric (see find_optimal_clip)
        alpha, beta    : composite weights
        verbose        : print progress

    Returns:
        dict mapping serie_key → float (optimal clip_limit)
        Failed series (load error) are skipped and not included.
    """
    results = {}
    total = len(series_dict)
    for i, (key, paths) in enumerate(series_dict.items()):
        mid_path = paths[len(paths) // 2]
        img = load_fn(mid_path)
        if img is None:
            if verbose:
                print(f"  [SKIP] {key} — could not load middle frame")
            continue

        bg = bg_subtract_fn(img, kernel_size)
        opt = find_optimal_clip(
            bg,
            clip_range=clip_range,
            n_steps=n_steps,
            grid_size=grid_size,
            metric=metric,
            alpha=alpha,
            beta=beta,
        )
        results[key] = opt["optimal_clip"]
        if verbose and (i % 20 == 0 or i == total - 1):
            print(f"  [{i+1}/{total}] {key} → clip={opt['optimal_clip']:.2f}")

    return results


def apply_adaptive_clahe(image, serie_key, clip_table, grid_size=(8, 8), fallback=2.0):
    """
    Applies CLAHE to a single image using the precomputed optimal clip_limit
    for its serie. Falls back to `fallback` if the serie is not in the table.

    Parameters:
        image      : grayscale uint8 numpy array
        serie_key  : str matching a key in clip_table (e.g. "P001/S01")
        clip_table : dict from compute_optimal_clips_for_series
        grid_size  : CLAHE tile grid size
        fallback   : clip_limit if serie_key is missing from table

    Returns:
        CLAHE-enhanced grayscale uint8 numpy array
    """
    clip = clip_table.get(serie_key, fallback)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=grid_size)
    return clahe.apply(image)


# ─────────────────────────────────────────────────────────────
# Diagnostic: metric curve for one image
# ─────────────────────────────────────────────────────────────

def plot_metric_curve(image, clip_range=(0.5, 8.0), n_steps=30, grid_size=(8, 8)):
    """
    Plots all four quality metrics as a function of clip_limit for a single
    image, with each metric normalised to [0, 1] for visual comparison.

    Useful to understand how the metrics behave on your data before
    committing to one.

    Parameters:
        image      : grayscale uint8 numpy array (post BG subtraction)
        clip_range : sweep range
        n_steps    : resolution of the sweep
        grid_size  : CLAHE tile grid size
    """
    clips = np.linspace(clip_range[0], clip_range[1], n_steps)
    raw = {m: [] for m in ["entropy", "sharpness", "tenengrad"]}

    for c in clips:
        clahe = cv2.createCLAHE(clipLimit=float(c), tileGridSize=grid_size)
        enh = clahe.apply(image)
        raw["entropy"].append(shannon_entropy(enh))
        raw["sharpness"].append(laplacian_variance(enh))
        raw["tenengrad"].append(tenengrad(enh))

    # normalise
    def norm(arr):
        a = np.array(arr)
        return (a - a.min()) / (np.ptp(a) + 1e-9)

    e_n = norm(raw["entropy"])
    s_n = norm(raw["sharpness"])
    t_n = norm(raw["tenengrad"])
    composite = 0.6 * e_n + 0.4 * t_n

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(clips, e_n,      label="Entropy",    color="#185FA5", lw=1.8)
    ax.plot(clips, s_n,      label="Sharpness",  color="#0F6E56", lw=1.8)
    ax.plot(clips, t_n,      label="Tenengrad",  color="#993C1D", lw=1.8)
    ax.plot(clips, composite,label="Composite",  color="#533AB7", lw=2.2, ls="--")

    best_idx = int(np.argmax(composite))
    ax.axvline(clips[best_idx], color="#533AB7", lw=1.2, ls=":", alpha=0.7,
               label=f"Best clip = {clips[best_idx]:.2f}")

    ax.set_xlabel("clip_limit", fontsize=12)
    ax.set_ylabel("Normalised score", fontsize=12)
    ax.set_title("Quality metrics vs clip_limit (normalised)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# Browse: adaptive vs fixed clip comparison grid
# ─────────────────────────────────────────────────────────────

def browse_adaptive_vs_fixed_grid(
    series_dict,
    clip_table,
    load_fn,
    bg_subtract_fn,
    kernel_size=41,
    n=9,
    seed=42,
):
    """
    Side-by-side 3-column grid: original | fixed CLAHE | adaptive CLAHE.

    The fixed clip_limit is controlled by a slider. The adaptive result
    uses the per-serie optimal from clip_table. This makes it easy to see
    whether adaptive outperforms a single hand-picked value.

    Parameters:
        series_dict    : output of group_paths_by_serie
        clip_table     : output of compute_optimal_clips_for_series
        load_fn        : callable(path) → uint8 array or None
        bg_subtract_fn : callable(image, kernel_size) → uint8 array
        kernel_size    : fixed BG subtraction kernel
        n              : number of series to sample
        seed           : random seed
    """
    rng = random.Random(seed)

    # One middle frame per serie (not per patient — we need the serie key)
    keys = sorted(series_dict.keys())
    selected_keys = rng.sample(keys, min(n, len(keys)))

    loaded = []
    for key in selected_keys:
        paths = series_dict[key]
        mid = paths[len(paths) // 2]
        img = load_fn(mid)
        if img is None:
            continue
        bg = bg_subtract_fn(img, kernel_size)
        opt_clip = clip_table.get(key, 2.0)
        loaded.append((key, bg, opt_clip))

    def update(fixed_clip):
        rows = len(loaded)
        fig, axes = plt.subplots(rows, 3, figsize=(13, rows * 3.2))
        if rows == 1:
            axes = axes[np.newaxis, :]
        for i, (key, bg, opt_clip) in enumerate(loaded):
            clahe_fixed  = cv2.createCLAHE(clipLimit=float(fixed_clip), tileGridSize=(8, 8))
            clahe_adapt  = cv2.createCLAHE(clipLimit=float(opt_clip),   tileGridSize=(8, 8))
            axes[i, 0].imshow(bg,                        cmap="gray"); axes[i, 0].set_ylabel(key, fontsize=7, rotation=0, labelpad=60)
            axes[i, 1].imshow(clahe_fixed.apply(bg),    cmap="gray")
            axes[i, 2].imshow(clahe_adapt.apply(bg),    cmap="gray")
            axes[i, 2].set_title(f"adaptive clip={opt_clip:.2f}", fontsize=8)
            for ax in axes[i]:
                ax.axis("off")

        axes[0, 0].set_title("BG subtracted",         fontsize=9)
        axes[0, 1].set_title(f"Fixed clip={fixed_clip:.1f}", fontsize=9)
        axes[0, 2].set_title("Adaptive clip",         fontsize=9)
        plt.suptitle("Fixed vs adaptive CLAHE", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.show()

    interact(
        update,
        fixed_clip=FloatSlider(min=0.5, max=8.0, step=0.5, value=2.0,
                               description="Fixed clip",
                               style={"description_width": "initial"},
                               layout={"width": "450px"}),
        continuous_update=False,
    )


# ─────────────────────────────────────────────────────────────
# Distribution: histogram of optimal clip limits across dataset
# ─────────────────────────────────────────────────────────────

def plot_clip_distribution(clip_table, bins=30):
    """
    Histogram of the optimal clip_limit values across all series.

    Useful sanity checks:
    - Are most series clustering around 2–3 (expected for angiography)?
    - Are some series getting very high values (≥ 6)? Those may have
      unusual illumination or be worth manual review.
    - Is the distribution unimodal? Bimodal might hint at two distinct
      acquisition protocols.

    Parameters:
        clip_table : output of compute_optimal_clips_for_series
        bins       : histogram bin count
    """
    values = list(clip_table.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=bins, color="#185FA5", alpha=0.75, edgecolor="white")
    ax.axvline(np.mean(values), color="#993C1D", lw=1.8,
               label=f"Mean = {np.mean(values):.2f}")
    ax.axvline(np.median(values), color="#0F6E56", lw=1.8, ls="--",
               label=f"Median = {np.median(values):.2f}")
    ax.set_xlabel("Optimal clip_limit", fontsize=12)
    ax.set_ylabel("Number of series",  fontsize=12)
    ax.set_title("Distribution of per-serie optimal clip limits", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()
    print(f"Series evaluated : {len(values)}")
    print(f"Mean clip         : {np.mean(values):.3f}")
    print(f"Median clip       : {np.median(values):.3f}")
    print(f"Std               : {np.std(values):.3f}")
    print(f"Min / Max         : {np.min(values):.2f} / {np.max(values):.2f}")











# ─────────────────────────────────────────────────────────────
# Gamma correction
# ─────────────────────────────────────────────────────────────
def apply_gamma(image, gamma=1.0):
    """
    Applies gamma correction using a lookup table.

    Gamma < 1:
        Brightens darker regions more aggressively.

    Gamma > 1:
        Darkens the image and compresses brighter regions.

    Parameters:
        image : grayscale uint8 numpy array
        gamma : gamma exponent (>0)

    Returns:
        Gamma-corrected uint8 image
    """
    gamma = max(gamma, 0.01)

    # Build LUT
    lut = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype(np.uint8)

    return cv2.LUT(image, lut)


def browse_gamma_grid(series_dict, n=9, seed=42):
    """
    3×3 grid with one gamma slider.
    Shows gamma-corrected result for n randomly selected patients.
    """
    loaded = _sample_patient_frames(series_dict, n, seed)

    def update(gamma):
        rows = int(np.ceil(len(loaded) / 3))

        fig, axes = plt.subplots(rows, 3, figsize=(3 * 3.5, rows * 3.5))
        axes = axes.flatten()

        for i, (pid, img) in enumerate(loaded):
            axes[i].imshow(apply_gamma(img, gamma=gamma), cmap='gray')
            axes[i].set_title(f"P{pid}", fontsize=8)
            axes[i].axis('off')

        for j in range(len(loaded), len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"Gamma correction grid — γ={gamma:.2f}",
                     fontsize=13, y=1.01)

        plt.tight_layout()
        plt.show()

    interact(
        update,
        gamma=FloatSlider(
            min=0.1,
            max=3.0,
            step=0.1,
            value=1.0,
            description='Gamma',
            style={'description_width': 'initial'},
            layout={'width': '500px'}
        ),
        continuous_update=False
    )


def browse_background_vs_gamma(series_dict, n=9, seed=42):
    """
    Interactive comparison between:
        - Background subtraction
        - Gamma correction

    Uses two sliders:
        - kernel_size for BG subtraction
        - gamma for gamma correction

    Each patient is shown as:
        Original | BG subtraction | Gamma correction
    """
    loaded = _sample_patient_frames(series_dict, n, seed)

    def update(kernel_size, gamma):

        rows = len(loaded)

        fig, axes = plt.subplots(
            rows, 3,
            figsize=(10, rows * 3)
        )

        if rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, (pid, img) in enumerate(loaded):

            bg_img = subtract_background(
                img,
                kernel_size=kernel_size
            )

            gamma_img = apply_gamma(
                img,
                gamma=gamma
            )

            # Original
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f"P{pid} — Original", fontsize=8)
            axes[i, 0].axis('off')

            # BG subtraction
            axes[i, 1].imshow(bg_img, cmap='gray')
            axes[i, 1].set_title(f"BG sub (k={kernel_size})", fontsize=8)
            axes[i, 1].axis('off')

            # Gamma
            axes[i, 2].imshow(gamma_img, cmap='gray')
            axes[i, 2].set_title(f"Gamma (γ={gamma:.2f})", fontsize=8)
            axes[i, 2].axis('off')

        plt.suptitle(
            "Background subtraction vs Gamma correction",
            fontsize=14,
            y=1.01
        )

        plt.tight_layout()
        plt.show()

    interact(
        update,

        kernel_size=IntSlider(
            min=1,
            max=201,
            step=10,
            value=21,
            description='BG kernel size',
            style={'description_width': 'initial'},
            layout={'width': '500px'}
        ),

        gamma=FloatSlider(
            min=0.1,
            max=3.0,
            step=0.1,
            value=1.0,
            description='Gamma',
            style={'description_width': 'initial'},
            layout={'width': '500px'}
        ),

        continuous_update=False
    )

# ─────────────────────────────────────────────────────────────
# Background subtraction vs Histogram Matching
# ─────────────────────────────────────────────────────────────
def browse_background_vs_histmatch(
    series_dict,
    global_reference,
    n=9,
    seed=42
):
    """
    Interactive comparison between:
        - Background subtraction
        - Histogram matching to a fixed global reference

    Each patient is shown as:
        Original | BG subtraction | Histogram matching
    """

    loaded = _sample_patient_frames(series_dict, n, seed)

    def update(kernel_size):

        rows = len(loaded)

        fig, axes = plt.subplots(
            rows,
            3,
            figsize=(10, rows * 3)
        )

        if rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, (pid, img) in enumerate(loaded):

            # Background subtraction
            bg_img = subtract_background(
                img,
                kernel_size=kernel_size
            )

            # Histogram matching
            hist_img = histogram_match_realtime(
                img,
                global_reference
            )

            # Original
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(
                f"P{pid} — Original",
                fontsize=8
            )
            axes[i, 0].axis('off')

            # BG subtraction
            axes[i, 1].imshow(bg_img, cmap='gray')
            axes[i, 1].set_title(
                f"BG sub (k={kernel_size})",
                fontsize=8
            )
            axes[i, 1].axis('off')

            # Histogram matching
            axes[i, 2].imshow(hist_img, cmap='gray')
            axes[i, 2].set_title(
                "Histogram matching",
                fontsize=8
            )
            axes[i, 2].axis('off')

        plt.suptitle(
            "Background subtraction vs Histogram matching",
            fontsize=14,
            y=1.01
        )

        plt.tight_layout()
        plt.show()

    interact(
        update,

        kernel_size=IntSlider(
            min=1,
            max=201,
            step=10,
            value=21,
            description='BG kernel size',
            style={'description_width': 'initial'},
            layout={'width': '500px'}
        ),

        continuous_update=False
    )

# ─────────────────────────────────────────────────────────────
# Residual fusion enhancement (smoothed residual version)
# ─────────────────────────────────────────────────────────────
def residual_fusion_enhancement(
    image,
    kernel_size=31,
    alpha=1.0,
    smooth_residual=True,
    bilateral_d=5,
    bilateral_sigma_color=20,
    bilateral_sigma_space=20
):
    """
    Vessel enhancement using:

        1. Background estimation
        2. Residual extraction
        3. Residual smoothing
        4. Residual amplification
        5. Fusion with original image

    Formally:

        B   = G_sigma(I)

        R   = I - B

        R_s = Smooth(R)

        I_enhanced = I + alpha * R_s

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image

    kernel_size : int
        Gaussian kernel size for background estimation

    alpha : float
        Residual amplification factor

    smooth_residual : bool
        Whether to smooth the residual before fusion

    bilateral_d : int
        Bilateral filter neighborhood diameter

    bilateral_sigma_color : float
        Bilateral intensity sigma

    bilateral_sigma_space : float
        Bilateral spatial sigma

    Returns
    -------
    enhanced : np.ndarray
        Enhanced uint8 image
    """

    # Ensure odd kernel size
    ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    # Convert to float32
    img_f = image.astype(np.float32)

    # ---------------------------------------------------------
    # 1. Background estimation
    # ---------------------------------------------------------
    background = cv2.GaussianBlur(
        img_f,
        (ks, ks),
        0
    )

    # ---------------------------------------------------------
    # 2. Residual extraction
    # ---------------------------------------------------------
    residual = background - img_f

    # ---------------------------------------------------------
    # Keep only positive residuals
    # (bright vessel-like structures)
    # ---------------------------------------------------------
    residual = np.maximum(residual, 0)

    # ---------------------------------------------------------
    # 3. Residual smoothing
    # ---------------------------------------------------------
    if smooth_residual:

        residual_uint8 = cv2.normalize(
            residual,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        residual_smooth = cv2.bilateralFilter(
            residual_uint8,
            d=bilateral_d,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space
        ).astype(np.float32)

    else:
        residual_smooth = residual

    # ---------------------------------------------------------
    # 4. Residual amplification + fusion
    # ---------------------------------------------------------
    enhanced = img_f - alpha * residual_smooth

    # ---------------------------------------------------------
    # Clip to valid range
    # ---------------------------------------------------------
    enhanced = np.clip(enhanced, 0, 255)

    return enhanced.astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# Interactive residual fusion grid
# ─────────────────────────────────────────────────────────────
def browse_residual_fusion_grid(
    series_dict,
    n=9,
    seed=42
):
    """
    Interactive comparison grid:

        Original | Residual fusion | Fusion + CLAHE

    Sliders:
        - kernel size
        - alpha
        - CLAHE clip limit
        - residual smoothing strength
    """

    loaded = _sample_patient_frames(series_dict, n, seed)

    def update(
        kernel_size,
        alpha,
        clip_limit,
        bilateral_sigma
    ):

        rows = len(loaded)

        fig, axes = plt.subplots(
            rows,
            3,
            figsize=(11, rows * 3)
        )

        if rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, (pid, img) in enumerate(loaded):

            # -------------------------------------------------
            # Residual fusion enhancement
            # -------------------------------------------------
            enhanced = residual_fusion_enhancement(
                img,
                kernel_size=kernel_size,
                alpha=alpha,
                smooth_residual=True,
                bilateral_d=5,
                bilateral_sigma_color=bilateral_sigma,
                bilateral_sigma_space=bilateral_sigma
            )

            # -------------------------------------------------
            # CLAHE
            # -------------------------------------------------
            clahe_img = apply_clahe(
                enhanced,
                clip_limit=clip_limit
            )

            # -------------------------------------------------
            # Original
            # -------------------------------------------------
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(
                f"P{pid} — Original",
                fontsize=8
            )
            axes[i, 0].axis('off')

            # -------------------------------------------------
            # Residual fusion
            # -------------------------------------------------
            axes[i, 1].imshow(enhanced, cmap='gray')
            axes[i, 1].set_title(
                f"Fusion α={alpha:.2f}",
                fontsize=8
            )
            axes[i, 1].axis('off')

            # -------------------------------------------------
            # Fusion + CLAHE
            # -------------------------------------------------
            axes[i, 2].imshow(clahe_img, cmap='gray')
            axes[i, 2].set_title(
                f"Fusion + CLAHE",
                fontsize=8
            )
            axes[i, 2].axis('off')

        plt.suptitle(
            f"Residual fusion enhancement "
            f"(k={kernel_size}, α={alpha:.2f}, "
            f"bilateral σ={bilateral_sigma}, "
            f"CLAHE={clip_limit:.1f})",
            fontsize=13,
            y=1.01
        )

        plt.tight_layout()
        plt.show()

    slider_style = {'description_width': 'initial'}
    slider_layout = {'width': '450px'}

    interact(
        update,

        kernel_size=IntSlider(
            min=3,
            max=201,
            step=10,
            value=31,
            description='Kernel size',
            style=slider_style,
            layout=slider_layout
        ),

        alpha=FloatSlider(
            min=0.1,
            max=3.0,
            step=0.1,
            value=1.0,
            description='Residual amplification α',
            style=slider_style,
            layout=slider_layout
        ),

        bilateral_sigma=IntSlider(
            min=5,
            max=100,
            step=5,
            value=20,
            description='Residual smoothing σ',
            style=slider_style,
            layout=slider_layout
        ),

        clip_limit=FloatSlider(
            min=0.1,
            max=10.0,
            step=0.5,
            value=2.0,
            description='CLAHE clip limit',
            style=slider_style,
            layout=slider_layout
        ),

        continuous_update=False
    )

def multiscale_residual_enhancement(
    image,
    kernel_sizes=(15, 33, 65),
    alpha=1.0,
    smooth_residual=True,
    bilateral_d=5,
    bilateral_sigma_color=50,
    bilateral_sigma_space=50,
    fusion_mode="max",   # "mean" or "max"
    invert=True
):
    """
    Multi-scale dark vessel enhancement.

    Pipeline
    --------
        1. Multi-scale background estimation
        2. Dark-vessel residual extraction
        3. Residual fusion across scales
        4. Residual smoothing
        5. Residual amplification
        6. Optional inversion

    Formally
    --------
        B_i = G_sigma_i(I)

        R_i = B_i - I

        R_multi = max(R_i)
        or
        R_multi = mean(R_i)

        I_enhanced = alpha * R_multi

    Notes
    -----
    - Dark vessels produce strong positive residuals.
    - The original image is NOT re-injected.
    - Output behaves like a vessel-response map.
    """

    img_f = image.astype(np.float32)

    residuals = []

    # ---------------------------------------------------------
    # 1. Multi-scale residual extraction
    # ---------------------------------------------------------
    for ks in kernel_sizes:

        ks = ks if ks % 2 == 1 else ks + 1

        # Background estimation
        background = cv2.GaussianBlur(
            img_f,
            (ks, ks),
            0
        )

        # Dark vessel residual
        residual = background - img_f

        # Keep only positive responses
        residual = np.maximum(residual, 0)

        residuals.append(residual)

    # ---------------------------------------------------------
    # 2. Multi-scale fusion
    # ---------------------------------------------------------
    if fusion_mode == "max":

        residual_multi = np.maximum.reduce(residuals)

    else:

        residual_multi = np.mean(residuals, axis=0)

    # ---------------------------------------------------------
    # 3. Remove weak responses
    # ---------------------------------------------------------
    threshold = np.percentile(residual_multi, 80)

    residual_multi[residual_multi < threshold] = 0

    # ---------------------------------------------------------
    # 4. Residual smoothing
    # ---------------------------------------------------------
    if smooth_residual:

        residual_uint8 = cv2.normalize(
            residual_multi,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        residual_smooth = cv2.bilateralFilter(
            residual_uint8,
            d=bilateral_d,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space
        ).astype(np.float32)

    else:

        residual_smooth = residual_multi

    # ---------------------------------------------------------
    # 5. Residual amplification only
    # ---------------------------------------------------------
    enhanced = alpha * residual_smooth

    # ---------------------------------------------------------
    # 6. Normalize output
    # ---------------------------------------------------------
    enhanced = cv2.normalize(
        enhanced,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    enhanced = enhanced.astype(np.uint8)

    # ---------------------------------------------------------
    # 7. Optional inversion
    # ---------------------------------------------------------
    if invert:

        enhanced = enhanced

    return enhanced


def browse_multiscale_residual_grid(
    series_dict,
    n=9,
    seed=42
):

    loaded = _sample_patient_frames(series_dict, n, seed)

    def update(
        alpha,
        clip_limit,
        bilateral_sigma,
        fusion_mode
    ):

        rows = len(loaded)

        fig, axes = plt.subplots(
            rows,
            3,
            figsize=(11, rows * 3)
        )

        if rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, (pid, img) in enumerate(loaded):

            # -------------------------------------------------
            # Multi-scale enhancement
            # -------------------------------------------------
            enhanced = multiscale_residual_enhancement(
                img,
                kernel_sizes=(15, 33, 65),
                alpha=alpha,
                smooth_residual=True,
                bilateral_d=5,
                bilateral_sigma_color=bilateral_sigma,
                bilateral_sigma_space=bilateral_sigma,
                fusion_mode=fusion_mode,
                invert=True
            )

            # -------------------------------------------------
            # CLAHE
            # -------------------------------------------------
            clahe_img = apply_clahe(
                enhanced,
                clip_limit=clip_limit
            )

            # -------------------------------------------------
            # Original
            # -------------------------------------------------
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(
                f"P{pid} — Original",
                fontsize=8
            )
            axes[i, 0].axis('off')

            # -------------------------------------------------
            # Multi-scale response
            # -------------------------------------------------
            axes[i, 1].imshow(enhanced, cmap='gray')
            axes[i, 1].set_title(
                f"Residual map ({fusion_mode})",
                fontsize=8
            )
            axes[i, 1].axis('off')

            # -------------------------------------------------
            # Response + CLAHE
            # -------------------------------------------------
            axes[i, 2].imshow(clahe_img, cmap='gray')
            axes[i, 2].set_title(
                "Residual + CLAHE",
                fontsize=8
            )
            axes[i, 2].axis('off')

        plt.suptitle(
            f"Multi-scale residual vessel enhancement "
            f"(fusion={fusion_mode}, α={alpha:.2f})",
            fontsize=13,
            y=1.01
        )

        plt.tight_layout()
        plt.show()

    slider_style = {'description_width': 'initial'}
    slider_layout = {'width': '450px'}

    interact(

        update,

        alpha=FloatSlider(
            min=0.1,
            max=3.0,
            step=0.1,
            value=1.0,
            description='Residual amplification α',
            style=slider_style,
            layout=slider_layout
        ),

        bilateral_sigma=IntSlider(
            min=5,
            max=120,
            step=5,
            value=80,
            description='Residual smoothing σ',
            style=slider_style,
            layout=slider_layout
        ),

        clip_limit=FloatSlider(
            min=0.1,
            max=5.0,
            step=0.1,
            value=1.0,
            description='CLAHE clip limit',
            style=slider_style,
            layout=slider_layout
        ),

        fusion_mode=Dropdown(
            options=["max", "mean"],
            value="max",
            description="Fusion mode",
            style=slider_style,
            layout=slider_layout
        ),

        continuous_update=False
    )





# ─────────────────────────────────────────────────────────────
# preprocessing_segmentation_teresa
# ─────────────────────────────────────────────────────────────

# ============================================================
# TEMPORAL BACKGROUND ESTIMATION
# ============================================================

def compute_temporal_median_background(frame_paths):
    """
    Computes temporal median background from all frames
    in a session.
    """
    frames = []

    for p in frame_paths:

        img = load_image(p)

        if img is not None:
            frames.append(img.astype(np.float32))

    if len(frames) == 0:
        raise ValueError("No valid frames.")

    stack = np.stack(frames, axis=0)

    return np.median(stack, axis=0)


def compute_temporal_mean_background(frame_paths):
    """
    Computes temporal mean background from all frames
    in a session.
    """
    frames = []

    for p in frame_paths:

        img = load_image(p)

        if img is not None:
            frames.append(img.astype(np.float32))

    if len(frames) == 0:
        raise ValueError("No valid frames.")

    stack = np.stack(frames, axis=0)

    return np.mean(stack, axis=0)



# ============================================================
# STATIC SUBTRACTION
# ============================================================

def subtract_static_background(image, background):
    """
    Static subtraction:

        R = I - B
    """

    img_f = image.astype(np.float32)

    residual = img_f - background

    residual = cv2.normalize(
        residual,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return residual.astype(np.uint8)


def session_median_subtraction(image, session_paths):
    """
    Median-based temporal subtraction.
    """

    background = compute_temporal_median_background(
        session_paths
    )

    return subtract_static_background(
        image,
        background
    )


def session_mean_subtraction(image, session_paths):
    """
    Mean-based temporal subtraction.
    """

    background = compute_temporal_mean_background(
        session_paths
    )

    return subtract_static_background(
        image,
        background
    )


# ============================================================
# DENOISING
# ============================================================

def apply_bilateral_filter(
    image,
    d=5,
    sigma_color=50,
    sigma_space=50
):
    """
    Edge-preserving bilateral denoising.
    """

    return cv2.bilateralFilter(
        image,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )


def apply_nlm_filter(
    image,
    h=10,
    template_window_size=7,
    search_window_size=21
):
    """
    Non-local means denoising.
    """

    return cv2.fastNlMeansDenoising(
        image,
        None,
        h=h,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size
    )



# ============================================================
# CLAHE
# ============================================================

def apply_clahe(
    image,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    CLAHE contrast enhancement.
    """

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(image)



# ============================================================
# FRANGI VESSELNESS
# ============================================================

def apply_frangi_filter(
    image,
    sigmas=(1, 2, 3),
    black_ridges=True
):
    """
    Vessel enhancement using Sato filter.

    NOTE:
    Function name is kept as apply_frangi_filter
    so the existing grids continue working.
    """

    img_f = image.astype(np.float32) / 255.0

    vesselness = sato(
        img_f,
        sigmas=sigmas,
        black_ridges=black_ridges
    )

    vesselness = cv2.normalize(
        vesselness,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return vesselness.astype(np.uint8)


# ============================================================
# THRESHOLDING
# ============================================================

def apply_otsu_threshold(image):
    """
    Otsu vessel segmentation.
    """

    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary



# ============================================================
# SKELETONIZATION
# ============================================================

def apply_skeletonization(binary_image):
    """
    Skeletonizes binary vessel mask.
    """

    skeleton = skeletonize(binary_image > 0)

    return (skeleton * 255).astype(np.uint8)




# ============================================================
# VESSEL MASK REFINEMENT
# ============================================================

def refine_vessel_mask(
    binary_image,
    kernel_size=5,
    min_component_size=100
):
    """
    Refines vessel segmentation mask.

    Steps:
    - Morphological closing
    - Hole filling
    - Small component removal
    """

    # --------------------------------------------------------
    # Morphological closing
    # --------------------------------------------------------
    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    closed = cv2.morphologyEx(
        binary_image,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # Fill internal holes
    # --------------------------------------------------------
    filled = binary_fill_holes(
        closed > 0
    )

    filled = (
        filled.astype(np.uint8) * 255
    )

    # --------------------------------------------------------
    # Remove small components
    # --------------------------------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        filled,
        connectivity=8
    )

    cleaned = np.zeros_like(filled)

    for label in range(1, num_labels):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= min_component_size:

            cleaned[labels == label] = 255

    return cleaned




# ============================================================
# Grid 1
# Original | Median subtraction | Mean subtraction
# ============================================================
def browse_static_subtraction_grid(
    series_dict,
    n=4,
    seed=42
):

    rng = random.Random(seed)

    selected = rng.sample(
        list(series_dict.items()),
        min(n, len(series_dict))
    )

    fig, axes = plt.subplots(
        len(selected),
        3,
        figsize=(10, len(selected) * 3)
    )

    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (key, session_paths) in enumerate(selected):

        frame_path = rng.choice(session_paths)

        img = load_image(frame_path)

        median_sub = session_median_subtraction(
            img,
            session_paths
        )

        mean_sub = session_mean_subtraction(
            img,
            session_paths
        )

        images = [
            ("Original", img),
            ("Median subtraction", median_sub),
            ("Mean subtraction", mean_sub),
        ]

        for j, (title, im) in enumerate(images):

            axes[i, j].imshow(im, cmap="gray")
            axes[i, j].set_title(title, fontsize=8)
            axes[i, j].axis("off")

    plt.suptitle(
        "Static background subtraction",
        fontsize=13
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# Grid 2
# Original | Mean | Mean+NLM | Mean+bilateral
# ============================================================
def browse_mean_denoising_grid(
    series_dict,
    n=4,
    seed=42
):

    rng = random.Random(seed)

    selected = rng.sample(
        list(series_dict.items()),
        min(n, len(series_dict))
    )

    loaded = []

    for key, session_paths in selected:

        frame_path = rng.choice(session_paths)

        img = load_image(frame_path)

        mean_sub = session_mean_subtraction(
            img,
            session_paths
        )

        loaded.append((img, mean_sub))

    def update(
        bilateral_sigma,
        nlm_h
    ):

        fig, axes = plt.subplots(
            len(loaded),
            4,
            figsize=(14, len(loaded) * 3)
        )

        if len(loaded) == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, (original, mean_sub) in enumerate(loaded):

            nlm = apply_nlm_filter(
                mean_sub,
                h=nlm_h
            )

            bilateral = apply_bilateral_filter(
                mean_sub,
                d=5,
                sigma_color=bilateral_sigma,
                sigma_space=bilateral_sigma
            )

            images = [
                ("Original", original),
                ("Mean subtraction", mean_sub),
                ("Mean + NLM", nlm),
                ("Mean + Bilateral", bilateral),
            ]

            for j, (title, im) in enumerate(images):

                axes[i, j].imshow(im, cmap="gray")
                axes[i, j].set_title(title, fontsize=8)
                axes[i, j].axis("off")

        plt.suptitle(
            f"Mean subtraction denoising "
            f"(NLM h={nlm_h}, bilateral σ={bilateral_sigma})",
            fontsize=13
        )

        plt.tight_layout()
        plt.show()

    interact(

        update,

        bilateral_sigma=IntSlider(
            min=5,
            max=100,
            step=5,
            value=50,
            description="Bilateral σ"
        ),

        nlm_h=IntSlider(
            min=1,
            max=30,
            step=1,
            value=10,
            description="NLM h"
        ),

        continuous_update=False
    )


# ============================================================
# Grid 3
# Original | Median | Median+NLM | Median+bilateral
# ============================================================
def browse_median_denoising_grid(
    series_dict,
    n=4,
    seed=42
):

    rng = random.Random(seed)

    selected = rng.sample(
        list(series_dict.items()),
        min(n, len(series_dict))
    )

    loaded = []

    for key, session_paths in selected:

        frame_path = rng.choice(session_paths)

        img = load_image(frame_path)

        median_sub = session_median_subtraction(
            img,
            session_paths
        )

        loaded.append((img, median_sub))

    def update(
        bilateral_sigma,
        nlm_h
    ):

        fig, axes = plt.subplots(
            len(loaded),
            4,
            figsize=(14, len(loaded) * 3)
        )

        if len(loaded) == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, (original, median_sub) in enumerate(loaded):

            nlm = apply_nlm_filter(
                median_sub,
                h=nlm_h
            )

            bilateral = apply_bilateral_filter(
                median_sub,
                d=5,
                sigma_color=bilateral_sigma,
                sigma_space=bilateral_sigma
            )

            images = [
                ("Original", original),
                ("Median subtraction", median_sub),
                ("Median + NLM", nlm),
                ("Median + Bilateral", bilateral),
            ]

            for j, (title, im) in enumerate(images):

                axes[i, j].imshow(im, cmap="gray")
                axes[i, j].set_title(title, fontsize=8)
                axes[i, j].axis("off")

        plt.suptitle(
            f"Median subtraction denoising "
            f"(NLM h={nlm_h}, bilateral σ={bilateral_sigma})",
            fontsize=13
        )

        plt.tight_layout()
        plt.show()

    interact(

        update,

        bilateral_sigma=IntSlider(
            min=5,
            max=100,
            step=5,
            value=50,
            description="Bilateral σ"
        ),

        nlm_h=IntSlider(
            min=1,
            max=30,
            step=1,
            value=10,
            description="NLM h"
        ),

        continuous_update=False
    )


# ============================================================
# Grid 4
# Original | Mean+NLM+CLAHE | Mean+bilateral+CLAHE
# ============================================================
def browse_mean_clahe_grid(
    series_dict,
    n=4,
    seed=42
):

    rng = random.Random(seed)

    selected = rng.sample(
        list(series_dict.items()),
        min(n, len(series_dict))
    )

    loaded = []

    for key, session_paths in selected:

        frame_path = rng.choice(session_paths)

        img = load_image(frame_path)

        mean_sub = session_mean_subtraction(
            img,
            session_paths
        )

        loaded.append((img, mean_sub))

    def update(
        bilateral_sigma,
        nlm_h,
        clip_limit
    ):

        fig, axes = plt.subplots(
            len(loaded),
            3,
            figsize=(11, len(loaded) * 3)
        )

        if len(loaded) == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, (original, mean_sub) in enumerate(loaded):

            nlm = apply_nlm_filter(
                mean_sub,
                h=nlm_h
            )

            nlm_clahe = apply_clahe(
                nlm,
                clip_limit=clip_limit
            )

            bilateral = apply_bilateral_filter(
                mean_sub,
                d=5,
                sigma_color=bilateral_sigma,
                sigma_space=bilateral_sigma
            )

            bilateral_clahe = apply_clahe(
                bilateral,
                clip_limit=clip_limit
            )

            images = [
                ("Original", original),
                ("Mean + NLM + CLAHE", nlm_clahe),
                ("Mean + Bilateral + CLAHE", bilateral_clahe),
            ]

            for j, (title, im) in enumerate(images):

                axes[i, j].imshow(im, cmap="gray")
                axes[i, j].set_title(title, fontsize=8)
                axes[i, j].axis("off")

        plt.suptitle(
            f"Mean subtraction + CLAHE "
            f"(clip={clip_limit:.1f})",
            fontsize=13
        )

        plt.tight_layout()
        plt.show()

    interact(

        update,

        bilateral_sigma=IntSlider(
            min=5,
            max=100,
            step=5,
            value=50,
            description="Bilateral σ"
        ),

        nlm_h=IntSlider(
            min=1,
            max=30,
            step=1,
            value=10,
            description="NLM h"
        ),

        clip_limit=FloatSlider(
            min=0.1,
            max=5.0,
            step=0.1,
            value=2.0,
            description="CLAHE clip"
        ),

        continuous_update=False
    )


# ============================================================
# Grid 5
# Original | Median+NLM+CLAHE | Median+bilateral+CLAHE
# ============================================================
def browse_median_clahe_grid(
    series_dict,
    n=4,
    seed=42
):

    rng = random.Random(seed)

    selected = rng.sample(
        list(series_dict.items()),
        min(n, len(series_dict))
    )

    loaded = []

    for key, session_paths in selected:

        frame_path = rng.choice(session_paths)

        img = load_image(frame_path)

        median_sub = session_median_subtraction(
            img,
            session_paths
        )

        loaded.append((img, median_sub))

    def update(
        bilateral_sigma,
        nlm_h,
        clip_limit
    ):

        fig, axes = plt.subplots(
            len(loaded),
            3,
            figsize=(11, len(loaded) * 3)
        )

        if len(loaded) == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, (original, median_sub) in enumerate(loaded):

            nlm = apply_nlm_filter(
                median_sub,
                h=nlm_h
            )

            nlm_clahe = apply_clahe(
                nlm,
                clip_limit=clip_limit
            )

            bilateral = apply_bilateral_filter(
                median_sub,
                d=5,
                sigma_color=bilateral_sigma,
                sigma_space=bilateral_sigma
            )

            bilateral_clahe = apply_clahe(
                bilateral,
                clip_limit=clip_limit
            )

            images = [
                ("Original", original),
                ("Median + NLM + CLAHE", nlm_clahe),
                ("Median + Bilateral + CLAHE", bilateral_clahe),
            ]

            for j, (title, im) in enumerate(images):

                axes[i, j].imshow(im, cmap="gray")
                axes[i, j].set_title(title, fontsize=8)
                axes[i, j].axis("off")

        plt.suptitle(
            f"Median subtraction + CLAHE "
            f"(clip={clip_limit:.1f})",
            fontsize=13
        )

        plt.tight_layout()
        plt.show()

    interact(

        update,

        bilateral_sigma=IntSlider(
            min=5,
            max=100,
            step=5,
            value=50,
            description="Bilateral σ"
        ),

        nlm_h=IntSlider(
            min=1,
            max=30,
            step=1,
            value=10,
            description="NLM h"
        ),

        clip_limit=FloatSlider(
            min=0.1,
            max=5.0,
            step=0.1,
            value=2.0,
            description="CLAHE clip"
        ),

        continuous_update=False
    )


# ============================================================
# Grid 6
# MEDIAN + NLM + CLAHE + FRANGI
# ============================================================

def browse_median_frangi_grid(
    series_dict,
    n=4,
    seed=42
):

    rng = random.Random(seed)

    selected = rng.sample(
        list(series_dict.items()),
        min(n, len(series_dict))
    )

    loaded = []

    for key, session_paths in selected:

        frame_path = rng.choice(session_paths)

        img = load_image(frame_path)

        # ----------------------------------------------------
        # Median subtraction
        # ----------------------------------------------------
        median_sub = session_median_subtraction(
            img,
            session_paths
        )

        # ----------------------------------------------------
        # NLM
        # ----------------------------------------------------
        nlm = apply_nlm_filter(
            median_sub,
            h=10
        )

        # ----------------------------------------------------
        # CLAHE
        # ----------------------------------------------------
        clahe = apply_clahe(
            nlm,
            clip_limit=2
        )

        loaded.append((img, clahe))

    def update(

        sigma_1,
        sigma_2,
        sigma_3

    ):

        sigmas = (
            sigma_1,
            sigma_2,
            sigma_3
        )

        fig, axes = plt.subplots(
            len(loaded),
            5,
            figsize=(18, len(loaded) * 3)
        )

        if len(loaded) == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, (original, clahe) in enumerate(loaded):

            # ------------------------------------------------
            # Frangi
            # ------------------------------------------------
            vesselness = apply_frangi_filter(
                clahe,
                sigmas=sigmas,
                black_ridges=False
            )

            # ------------------------------------------------
            # Threshold
            # ------------------------------------------------
            binary = apply_otsu_threshold(
                vesselness
            )

            binary = refine_vessel_mask(
                binary,
                kernel_size=5,
                min_component_size=100
            )

            # ------------------------------------------------
            # Skeleton
            # ------------------------------------------------
            skeleton = apply_skeletonization(
                binary
            )

            images = [

                ("Original", original),

                ("Median + NLM + CLAHE", clahe),

                ("Frangi", vesselness),

                ("Binary", binary),

                ("Skeleton", skeleton),

            ]

            for j, (title, im) in enumerate(images):

                axes[i, j].imshow(
                    im,
                    cmap="gray"
                )

                axes[i, j].set_title(
                    title,
                    fontsize=8
                )

                axes[i, j].axis("off")

        plt.suptitle(
            f"Median pipeline + Frangi "
            f"(sigmas={sigmas})",
            fontsize=13
        )

        plt.tight_layout()
        plt.show()

    interact(

        update,

        sigma_1=FloatSlider(
            min=0.5,
            max=5,
            step=0.5,
            value=1,
            description="Sigma 1"
        ),

        sigma_2=FloatSlider(
            min=0.5,
            max=8,
            step=0.5,
            value=2,
            description="Sigma 2"
        ),

        sigma_3=FloatSlider(
            min=0.5,
            max=12,
            step=0.5,
            value=3,
            description="Sigma 3"
        ),

        continuous_update=False
    )





# ============================================================
# Grid 7
# MEAN + NLM + CLAHE + FRANGI
# ============================================================

def browse_mean_frangi_grid(
    series_dict,
    n=4,
    seed=42
):

    rng = random.Random(seed)

    selected = rng.sample(
        list(series_dict.items()),
        min(n, len(series_dict))
    )

    loaded = []

    for key, session_paths in selected:

        frame_path = rng.choice(session_paths)

        img = load_image(frame_path)

        # ----------------------------------------------------
        # Mean subtraction
        # ----------------------------------------------------
        mean_sub = session_mean_subtraction(
            img,
            session_paths
        )

        # ----------------------------------------------------
        # NLM
        # ----------------------------------------------------
        nlm = apply_nlm_filter(
            mean_sub,
            h=10
        )

        # ----------------------------------------------------
        # CLAHE
        # ----------------------------------------------------
        clahe = apply_clahe(
            nlm,
            clip_limit=2
        )

        loaded.append((img, clahe))

    def update(

        sigma_1,
        sigma_2,
        sigma_3

    ):

        sigmas = (
            sigma_1,
            sigma_2,
            sigma_3
        )

        fig, axes = plt.subplots(
            len(loaded),
            5,
            figsize=(18, len(loaded) * 3)
        )

        if len(loaded) == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, (original, clahe) in enumerate(loaded):

            # ------------------------------------------------
            # Frangi
            # ------------------------------------------------
            vesselness = apply_frangi_filter(
                clahe,
                sigmas=sigmas,
                black_ridges=False
            )

            # ------------------------------------------------
            # Threshold
            # ------------------------------------------------
            binary = apply_otsu_threshold(
                vesselness
            )

            binary = refine_vessel_mask(
                binary,
                kernel_size=5,
                min_component_size=100
            )

            # ------------------------------------------------
            # Skeleton
            # ------------------------------------------------
            skeleton = apply_skeletonization(
                binary
            )

            images = [

                ("Original", original),

                ("Mean + NLM + CLAHE", clahe),

                ("Frangi", vesselness),

                ("Binary", binary),

                ("Skeleton", skeleton),

            ]

            for j, (title, im) in enumerate(images):

                axes[i, j].imshow(
                    im,
                    cmap="gray"
                )

                axes[i, j].set_title(
                    title,
                    fontsize=8
                )

                axes[i, j].axis("off")

        plt.suptitle(
            f"Mean pipeline + Frangi "
            f"(sigmas={sigmas})",
            fontsize=13
        )

        plt.tight_layout()
        plt.show()

    interact(

        update,

        sigma_1=FloatSlider(
            min=0.5,
            max=5,
            step=0.5,
            value=1,
            description="Sigma 1"
        ),

        sigma_2=FloatSlider(
            min=0.5,
            max=8,
            step=0.5,
            value=2,
            description="Sigma 2"
        ),

        sigma_3=FloatSlider(
            min=0.5,
            max=12,
            step=0.5,
            value=3,
            description="Sigma 3"
        ),

        continuous_update=False
    )