import os
import json
import random
import warnings
from pathlib import Path

import numpy as np
import cv2
from ipywidgets import IntSlider, FloatSlider, interact, Dropdown
from matplotlib import pyplot as plt
from skimage import filters, morphology
from skimage.filters import frangi
from src.utils import load_image


# import pandas as pd
# from tqdm import tqdm


# ─────────────────────────────────────────────
# STAGE 3 — Vesselness (Frangi)
# ─────────────────────────────────────────────

def frangi_vesselness(preprocessed: np.ndarray,
                      sigma_min: float = 1.0,
                      sigma_max: float = 10.0,
                      sigma_steps: int = 6,
                      beta: float = 0.5,
                      gamma: float = 15.0) -> np.ndarray:
    """
    Apply Frangi vesselness filter to a single preprocessed float32 frame.

    IMPORTANT: angiography vessels are DARK → invert before Frangi,
    then use black_ridges=False (skimage ≥ 0.19) which detects bright ridges
    on the inverted image == dark vessels on the original.

    Returns a float64 vesselness map in [0, 1].
    """
    # Invert: dark vessels → bright ridges
    inverted = 1.0 - preprocessed

    sigmas = np.linspace(sigma_min, sigma_max, sigma_steps)

    vessel_map = frangi(inverted,
                        sigmas=sigmas,
                        beta=beta,
                        gamma=gamma,
                        black_ridges=False)   # bright ridges on inverted img

    # Normalise to [0, 1] for display / thresholding
    v_min, v_max = vessel_map.min(), vessel_map.max()
    if v_max > v_min:
        vessel_map = (vessel_map - v_min) / (v_max - v_min)

    return vessel_map


# ─────────────────────────────────────────────
# INTERACTIVE EXPLORER  (Jupyter / ipywidgets)
# ─────────────────────────────────────────────

# def browse_frangi_interactive(series_dict, title="Frangi explorer"):
#     """
#     Single interactive widget that lets you switch between series
#     via a dropdown — avoids the broken for-loop pattern.
#     """
#     # Pre-cache all series
#     print("Loading preprocessed frames…")
#     cache = {
#         key: [load_image(p) for p in paths]
#         for key, paths in series_dict.items()
#     }
#     print("Done. Use sliders to explore Frangi parameters.")
#
#     def update(serie_key, frame_idx, sigma_min, sigma_max, sigma_steps, beta, gamma):
#         try:
#             frames = cache[serie_key]
#             frame_idx = min(frame_idx, len(frames) - 1)
#             pre = frames[frame_idx]
#             if pre.dtype == np.uint8:
#                 pre = pre.astype(np.float32) / 255.0
#
#             # Debug: confirm what load_image returned
#             print(f"pre type: {type(pre)}, dtype: {getattr(pre, 'dtype', 'N/A')}, shape: {getattr(pre, 'shape', 'N/A')}")
#
#             if pre is None:
#                 print("❌ load_image returned None — check your path/loader")
#                 return
#             fname = Path(list(series_dict[serie_key])[frame_idx]).name
#
#             if sigma_min >= sigma_max:
#                 sigma_max = sigma_min + 0.5
#
#             vessel = frangi_vesselness(pre,
#                                        sigma_min=sigma_min,
#                                        sigma_max=sigma_max,
#                                        sigma_steps=int(sigma_steps),
#                                        beta=beta,
#                                        gamma=gamma)
#         except Exception as e:
#             import traceback
#             print(f"❌ Error in update(): {e}")
#             traceback.print_exc()
#
#         from skimage.filters import threshold_otsu
#         otsu_val = threshold_otsu(vessel)
#         p95 = np.percentile(vessel, 95)
#         p99 = np.percentile(vessel, 99)
#
#         fig, axes = plt.subplots(1, 3, figsize=(20, 7))
#         fig.suptitle(
#             f"{serie_key}  |  {fname}\n"
#             f"σ=[{sigma_min:.1f}–{sigma_max:.1f}, {int(sigma_steps)} steps]  "
#             f"β={beta:.2f}  γ={gamma:.1f}",
#             fontsize=11,
#         )
#
#         axes[0].imshow(pre, cmap="gray")
#         axes[0].set_title("① Preprocessed (CLAHE + NLM)")
#         axes[0].axis("off")
#
#         axes[1].imshow(vessel, cmap="hot")
#         axes[1].set_title("② Vesselness map (Frangi)")
#         axes[1].axis("off")
#         sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(vmin=0, vmax=1))
#         plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
#
#         axes[2].hist(vessel.ravel(), bins=256, color="steelblue", log=True)
#         axes[2].set_title("③ Vesselness histogram (log scale)")
#         axes[2].set_xlabel("Vesselness value")
#         axes[2].set_ylabel("Pixel count (log)")
#         axes[2].axvline(otsu_val, color="red", linestyle="--",
#                         label=f"Otsu = {otsu_val:.3f}")
#         axes[2].axvline(p95, color="orange", linestyle="--",
#                         label=f"p95  = {p95:.3f}")
#         axes[2].axvline(p99, color="green", linestyle="--",
#                         label=f"p99  = {p99:.3f}")
#         axes[2].legend(fontsize=8)
#
#         plt.tight_layout()
#         plt.show()
#
#     # Max frames across all series for the slider upper bound
#     max_frames = max(len(v) for v in series_dict.values())
#
#     serie_dropdown = Dropdown(options=list(series_dict.keys()),
#                               description="Serie")
#     frame_slider = IntSlider(min=0, max=max_frames - 1, step=1,
#                              value=0, description="Frame")
#     sigma_min_slider = FloatSlider(min=0.5, max=5.0, step=0.5, value=1.0, description="σ min")
#     sigma_max_slider = FloatSlider(min=2.0, max=20.0, step=0.5, value=10.0, description="σ max")
#     sigma_steps_slider = IntSlider(min=2, max=20, step=1, value=6, description="σ steps")
#     beta_slider = FloatSlider(min=0.1, max=2.0, step=0.05, value=0.5, description="beta (β)")
#     gamma_slider = FloatSlider(min=1.0, max=50.0, step=0.5, value=15.0, description="gamma (γ)")
#
#     interact(
#         update,
#         serie_key=serie_dropdown,
#         frame_idx=frame_slider,
#         sigma_min=sigma_min_slider,
#         sigma_max=sigma_max_slider,
#         sigma_steps=sigma_steps_slider,
#         beta=beta_slider,
#         gamma=gamma_slider,
#         continuous_update=False
#     )

def browse_frangi_interactive(series_dict, title="Frangi explorer"):
    print("Loading preprocessed frames…")
    cache = {
        key: [load_image(p) for p in paths]
        for key, paths in series_dict.items()
    }
    print("Done. Use sliders to explore Frangi parameters.")

    _frangi_cache = {}  # ← CAMBIO 1: cache

    def update(serie_key, frame_idx, sigma_min, sigma_max, sigma_steps, beta, gamma):
        frames = cache[serie_key]
        frame_idx = min(frame_idx, len(frames) - 1)
        pre = frames[frame_idx]
        if pre.dtype == np.uint8:
            pre = pre.astype(np.float32) / 255.0
        fname = Path(list(series_dict[serie_key])[frame_idx]).name

        if sigma_min >= sigma_max:
            sigma_max = sigma_min + 0.5

        # ← CAMBIO 1: usar cache
        fkey = (serie_key, frame_idx, sigma_min, sigma_max, int(sigma_steps), beta, gamma)
        if fkey not in _frangi_cache:
            _frangi_cache[fkey] = frangi_vesselness(pre,
                                                     sigma_min=sigma_min,
                                                     sigma_max=sigma_max,
                                                     sigma_steps=int(sigma_steps),
                                                     beta=beta,
                                                     gamma=gamma)
        vessel = _frangi_cache[fkey]

        # ← CAMBIO 2: quitado threshold_otsu (no se usa en el plot)
        p95 = np.percentile(vessel, 95)
        p99 = np.percentile(vessel, 99)

        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(
            f"{serie_key}  |  {fname}\n"
            f"σ=[{sigma_min:.1f}–{sigma_max:.1f}, {int(sigma_steps)} steps]  "
            f"β={beta:.2f}  γ={gamma:.1f}",
            fontsize=11,
        )

        axes[0].imshow(pre, cmap="gray")
        axes[0].set_title("① Preprocessed (CLAHE + NLM)")
        axes[0].axis("off")

        axes[1].imshow(vessel, cmap="gray")
        axes[1].set_title("② Vesselness map (Frangi)")
        axes[1].axis("off")
        sm = plt.cm.ScalarMappable(cmap="gray", norm=plt.Normalize(vmin=0, vmax=1))
        plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].hist(vessel.ravel(), bins=256, color="steelblue", log=True)
        axes[2].set_title("③ Vesselness histogram (log scale)")
        axes[2].set_xlabel("Vesselness value")
        axes[2].set_ylabel("Pixel count (log)")
        axes[2].axvline(p95, color="orange", linestyle="--", label=f"p95 = {p95:.3f}")
        axes[2].axvline(p99, color="green",  linestyle="--", label=f"p99 = {p99:.3f}")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        plt.show()

    max_frames = max(len(v) for v in series_dict.values())

    interact(
        update,
        serie_key=Dropdown(options=list(series_dict.keys()), description="Serie"),
        frame_idx=IntSlider(min=0, max=max_frames - 1, step=1, value=0, description="Frame"),
        sigma_min=FloatSlider(min=0.5, max=5.0,  step=0.5, value=1.0, description="σ min"),
        sigma_max=FloatSlider(min=2.0, max=20.0, step=0.5, value=10.0, description="σ max"),
        sigma_steps=IntSlider(min=2,   max=20,   step=1,   value=6,    description="σ steps"),
        beta=FloatSlider(min=0.1,  max=2.0,  step=0.05, value=0.5,  description="beta (β)"),
        gamma=FloatSlider(min=1.0, max=50.0, step=0.5,  value=15.0, description="gamma (γ)"),
        continuous_update=False
    )

def browse_frangi_interactive2(series_dict, title="Frangi explorer"):
    """
    series_dict: {serie_key: [np.ndarray, ...]}  ← already loaded frames
    """
    print("Frames already loaded. Use sliders to explore Frangi parameters.")

    # No load_image needed — frames are already numpy arrays
    cache = series_dict  # ← direct reference, no loading

    _frangi_cache = {}

    def update(serie_key, frame_idx, sigma_min, sigma_max, sigma_steps, beta, gamma):
        frames = cache[serie_key]
        frame_idx = min(frame_idx, len(frames) - 1)
        pre = frames[frame_idx]

        if pre.dtype == np.uint8:
            pre = pre.astype(np.float32) / 255.0

        # No path available, use index as label
        fname = f"frame_{frame_idx:03d}"

        if sigma_min >= sigma_max:
            sigma_max = sigma_min + 0.5

        fkey = (serie_key, frame_idx, sigma_min, sigma_max, int(sigma_steps), beta, gamma)
        if fkey not in _frangi_cache:
            _frangi_cache[fkey] = frangi_vesselness(
                pre,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sigma_steps=int(sigma_steps),
                beta=beta,
                gamma=gamma,
            )
        vessel = _frangi_cache[fkey]

        p95 = np.percentile(vessel, 95)
        p99 = np.percentile(vessel, 99)

        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(
            f"{serie_key}  |  {fname}\n"
            f"σ=[{sigma_min:.1f}–{sigma_max:.1f}, {int(sigma_steps)} steps]  "
            f"β={beta:.2f}  γ={gamma:.1f}",
            fontsize=11,
        )

        axes[0].imshow(pre, cmap="gray")
        axes[0].set_title("① Preprocessed (CLAHE + NLM)")
        axes[0].axis("off")

        axes[1].imshow(vessel, cmap="gray")
        axes[1].set_title("② Vesselness map (Frangi)")
        axes[1].axis("off")
        sm = plt.cm.ScalarMappable(cmap="gray", norm=plt.Normalize(vmin=0, vmax=1))
        plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].hist(vessel.ravel(), bins=256, color="steelblue", log=True)
        axes[2].set_title("③ Vesselness histogram (log scale)")
        axes[2].set_xlabel("Vesselness value")
        axes[2].set_ylabel("Pixel count (log)")
        axes[2].axvline(p95, color="orange", linestyle="--", label=f"p95 = {p95:.3f}")
        axes[2].axvline(p99, color="green",  linestyle="--", label=f"p99 = {p99:.3f}")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        plt.show()

    max_frames = max(len(v) for v in series_dict.values())

    interact(
        update,
        serie_key=Dropdown(options=list(series_dict.keys()), description="Serie"),
        frame_idx=IntSlider(min=0, max=max_frames - 1, step=1, value=0, description="Frame"),
        sigma_min=FloatSlider(min=0.5, max=5.0,  step=0.5, value=1.0,  description="σ min"),
        sigma_max=FloatSlider(min=2.0, max=20.0, step=0.5, value=10.0, description="σ max"),
        sigma_steps=IntSlider(min=2,   max=20,   step=1,   value=6,    description="σ steps"),
        beta=FloatSlider(min=0.1,  max=2.0,  step=0.05, value=0.5,  description="beta (β)"),
        gamma=FloatSlider(min=1.0, max=50.0, step=0.5,  value=15.0, description="gamma (γ)"),
        continuous_update=False,
    )

def binarize_vessels(vesselness: np.ndarray,
                     method: str = "otsu") -> np.ndarray:
    v_u8 = (vesselness * 255).astype(np.uint8)

    if method == "otsu":
        _, binary = cv2.threshold(v_u8, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "local":
        thresh = filters.threshold_sauvola(vesselness, window_size=25)
        binary = (vesselness > thresh).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    # morphological clean-up: remove tiny speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return (binary > 0).astype(np.uint8)


def skeletonise(binary_mask: np.ndarray) -> np.ndarray:
    """
    Medial-axis skeletonisation → single-pixel-wide centreline.
    """
    skel = morphology.skeletonize(binary_mask.astype(bool))
    return skel.astype(np.uint8)


def load_annotations(annot_path: Path):
    """
    Load ground-truth bounding boxes from a JSON file.
    Expected format: {"boxes": [[x, y, w, h], ...]}
    Returns list of (x, y, w, h) tuples or [] if file missing.
    """
    if not annot_path.exists():
        return []
    with open(annot_path) as f:
        data = json.load(f)
    return [tuple(b) for b in data.get("boxes", [])]


def point_in_boxes(cx: int, cy: int, boxes) -> bool:
    """Return True if (cx, cy) falls inside any bounding box."""
    for (x, y, w, h) in boxes:
        if x <= cx <= x + w and y <= cy <= y + h:
            return True
    return False


def min_dist_to_boxes(cx: int, cy: int, boxes) -> float:
    """Minimum distance from (cx, cy) to the nearest bounding box border."""
    if not boxes:
        return float("inf")
    dists = []
    for (x, y, w, h) in boxes:
        # clamp point to box, measure distance
        nx = np.clip(cx, x, x + w)
        ny = np.clip(cy, y, y + h)
        dists.append(np.hypot(cx - nx, cy - ny))
    return min(dists)


def extract_patch(img: np.ndarray,
                  cx: int, cy: int,
                  size: int) -> np.ndarray | None:
    """
    Extract a square patch centred on (cx, cy).
    Returns None if the patch would go out of bounds.
    """
    half = size // 2
    h, w = img.shape[:2]
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return img[y0:y1, x0:x1].copy()

#
# def extract_rois(preprocessed_img: np.ndarray,
#                  skeleton: np.ndarray,
#                  boxes,
#                  patient: str,
#                  session: str,
#                  frame_stem: str,
#                  metadata: list,
#                  roi_size: int = ROI_SIZE):
#     """
#     Monte-Carlo sampling along the skeleton centreline.
#     Positive ROIs  — centre inside a GT bounding box.
#     Negative ROIs  — centre ≥ MIN_DIST_NEG px from every bbox.
#     """
#     ys, xs = np.where(skeleton > 0)
#     n_pixels = len(xs)
#     if n_pixels == 0:
#         return
#
#     # sample a fraction of skeleton pixels
#     n_sample = max(1, int(n_pixels * MONTECARLO_FRAC))
#     indices = np.random.choice(n_pixels, size=n_sample, replace=False)
#
#     positives, negatives = [], []
#
#     for idx in indices:
#         cx, cy = int(xs[idx]), int(ys[idx])
#         patch = extract_patch(preprocessed_img, cx, cy, roi_size)
#         if patch is None:
#             continue
#
#         if point_in_boxes(cx, cy, boxes):
#             positives.append((cx, cy, patch))
#         elif min_dist_to_boxes(cx, cy, boxes) >= MIN_DIST_NEG:
#             negatives.append((cx, cy, patch))
#
#     # ── class balancing: cap negatives ─────────────────────────────────────
#     max_neg = max(1, int(len(positives) * MAX_NEG_RATIO))
#     if len(negatives) > max_neg:
#         negatives = random.sample(negatives, max_neg)
#
#     # ── save patches ────────────────────────────────────────────────────────
#     base = f"{patient}_{session}_{frame_stem}"
#
#     for i, (cx, cy, patch) in enumerate(positives):
#         fname = f"{base}_roi{i:04d}.png"
#         cv2.imwrite(str(OUTPUT_DIR / "positive" / fname),
#                     (patch * 255).astype(np.uint8))
#         metadata.append(dict(label=1, patient=patient, session=session,
#                              frame=frame_stem, cx=cx, cy=cy, file=fname))
#
#     n_pos = len(positives)
#     for i, (cx, cy, patch) in enumerate(negatives):
#         fname = f"{base}_roi{n_pos + i:04d}.png"
#         cv2.imwrite(str(OUTPUT_DIR / "negative" / fname),
#                     (patch * 255).astype(np.uint8))
#         metadata.append(dict(label=0, patient=patient, session=session,
#                              frame=frame_stem, cx=cx, cy=cy, file=fname))