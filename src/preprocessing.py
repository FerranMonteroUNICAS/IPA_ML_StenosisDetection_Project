import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider, VBox, HBox
import random
from src.utils import *
from src.utils import _sample_patient_frames

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

def browse_bg_clahe_grid(series_dict, kernel_size=41, n=9, seed=42):
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

# ─────────────────────────────────────────────────────────────
# Image Pre-Processing functions:
# ─────────────────────────────────────────────────────────────
def preprocess_bs_clahe_nlm(img, kernel_size=81, clip_limit=4.0, h=8):
    return apply_nlm(apply_clahe(subtract_background(img, kernel_size=kernel_size), clip_limit=clip_limit), h=h)

def preprocess_bs_nlm_clahe(img, kernel_size=81, h=8, clip_limit=4.0):
    return apply_clahe(apply_nlm(subtract_background(img, kernel_size=kernel_size), h=h), clip_limit=clip_limit)

# ─────────────────────────────────────────────────────────────
# Morphological Bottom-Hat Pipeline (Alternative to Gaussian BS) --> NOT USED
# ─────────────────────────────────────────────────────────────
def apply_bottom_hat(image, radius=15):
    """
    Applies a Morphological Bottom-Hat (Black Top-Hat) transform using a
    circular structuring element. This isolates dark features (vessels) from
    a light background and outputs them as bright structures on a dark background.

    Parameters:
        image  : grayscale uint8 numpy array
        radius : radius of the circular structuring element (disk) in pixels
    """
    # Create a circular (ellipse) structuring element of size (2r+1) x (2r+1)
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    # Bottom-Hat = Closing(Image) - Image
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

def browse_bottomhat_nlm_clahe_grid(series_dict, n=9, seed=42):
    """
    Interactive 3×3 grid browser for the morphological pipeline:
        Bottom-Hat Transform → NLM Denoising → CLAHE Enhancement

    Sliders:
        - Radius (r)   : Radius of the disk structuring element. Must be slightly
                         larger than the thickest vessel you wish to keep.
        - h            : NLM filter strength.
        - Clip limit   : CLAHE local contrast amplification limit.
    """
    # Load raw images once
    loaded = _sample_patient_frames(series_dict, n, seed)

    def update(radius, h, clip_val):
        rows = int(np.ceil(len(loaded) / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(3 * 3.5, rows * 3.5))
        axes = axes.flatten()

        for i, (pid, img) in enumerate(loaded):
            # 1. Bottom-Hat isolates dark vessels and converts them to bright features
            bhat = apply_bottom_hat(img, radius=radius)
            # 2. Denoise the bright feature space
            denoised = apply_nlm(bhat, h=h)
            # 3. Enhance local contrast
            result = apply_clahe(denoised, clip_limit=clip_val)

            axes[i].imshow(result, cmap='gray')
            axes[i].set_title(f"P{pid}", fontsize=8)
            axes[i].axis('off')

        for j in range(len(loaded), len(axes)):
            axes[j].axis('off')

        plt.suptitle(
            f"Bottom-Hat (r={radius}px) → NLM (h={h}) → CLAHE (clip={clip_val:.1f})",
            fontsize=12, y=1.01
        )
        plt.tight_layout()
        plt.show()

    slider_style = {'description_width': 'initial'}
    slider_layout = {'width': '450px'}

    interact(
        update,
        radius=IntSlider(min=3, max=45, step=2, value=15,
                         description='Disk Radius (r)',
                         style=slider_style, layout=slider_layout),
        h=IntSlider(min=1, max=30, step=1, value=8,
                    description='NLM strength (h)',
                    style=slider_style, layout=slider_layout),
        clip_val=FloatSlider(min=0.1, max=10.0, step=0.1, value=4.0,
                             description='Clip limit',
                             style=slider_style, layout=slider_layout),
        continuous_update=False
    )

def preprocess_bottomhat_nlm_clahe(img, radius=15, h=8, clip_limit=4.0):
    """
    Production pipeline utilizing morphological bottom-hat background elimination.
    Outputs bright vessels over a dark background.
    """
    bhat = apply_bottom_hat(img, radius=radius)
    denoised = apply_nlm(bhat, h=h)
    return apply_clahe(denoised, clip_limit=clip_limit)
