"""
preprocessing.py
────────────────────────────────────────────────────────────────
Stage 2 — Preprocessing pipeline for coronary angiography frames.

Pipeline per frame:
    1. Histogram matching  → inter-series intensity normalisation
    2. CLAHE               → local contrast enhancement
    3. Bilateral filter    → edge-preserving noise reduction

Reference strategy:
    Instead of a mean image (blurry, wrong) or a single middle frame
    (series-specific), we build the reference from the MEDIAN CDF of a
    random sample of frames.  The median CDF is then inverted back into
    a synthetic 1-D intensity mapping that all frames are matched against.
    This is robust, reproducible, and truly represents a "typical" frame
    from the dataset without any spatial blurring artefacts.

Usage:
    ref = build_median_cdf_reference(all_image_paths)   # once offline
    for path in series_paths:
        img = load_image(path)
        out = preprocess_frame(img, ref)
"""



from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import cv2
from tqdm.notebook import tqdm
from src.utils import load_image  # your existing loader


# ─────────────────────────────────────────────────────────────
# 0.  Path utilities
# ─────────────────────────────────────────────────────────────

def group_paths_by_serie(image_paths: List[str]) -> dict:
    """
    Groups image paths by their serieID.
    Folder layout expected:  .../patientID/serieID/frame.png

    Uses parent-relative names instead of fixed index offsets, so it is
    robust to any root depth.
    """
    groups = defaultdict(list)
    for path in image_paths:
        p = Path(path)
        # p.parent.name  → serieID
        # p.parents[1].name → patientID
        serie_key = f"{p.parents[1].name}/{p.parent.name}"
        groups[serie_key].append(path)
    return {k: sorted(v) for k, v in groups.items()}


def get_middle_frame(serie_paths: List[str]) -> str:
    return serie_paths[len(serie_paths) // 2]


# ─────────────────────────────────────────────────────────────
# 1.  Reference construction — median CDF strategy
# ─────────────────────────────────────────────────────────────

def _compute_cdf_from_path(path):
    """Worker function — must be top-level for pickle-ability."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    cdf  = hist.cumsum()
    return cdf / cdf[-1]


def build_median_cdf_reference_parallel(
    image_paths,
    n_samples=100,
    seed=42,
    n_workers=8,        # tune to your CPU core count
):
    rng     = np.random.default_rng(seed)
    sampled = rng.choice(image_paths,
                         size=min(n_samples, len(image_paths)),
                         replace=False).tolist()
    cdfs = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_compute_cdf_from_path, p): p for p in sampled}
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Building reference CDF"):
            result = future.result()
            if result is not None:
                cdfs.append(result)

    cdf_stack  = np.stack(cdfs, axis=0)
    median_cdf = np.median(cdf_stack, axis=0)
    print(f"Reference CDF built from {len(cdfs)} frames.")
    return median_cdf

    """
    Builds a global reference as the PIXEL-WISE MEDIAN of sampled CDFs.

    Why median CDF instead of a mean image?
    ────────────────────────────────────────
    • A mean image averages spatial content → produces a blurry phantom
      that is not a valid intensity distribution.
    • The median CDF captures the "typical" tonal curve of the dataset
      without any spatial averaging.  It is a proper 1-D histogram target
      and is robust to outlier frames (over/under-exposed acquisitions).

    Returns:
        ref_cdf : (256,) float64 array — the median normalised CDF.
                  Pass this directly to match_to_reference_cdf().
    """
    rng = np.random.default_rng(seed)
    sampled = rng.choice(image_paths,
                         size=min(n_samples, len(image_paths)),
                         replace=False)
    cdfs = []
    for path in sampled:
        img = load_image(str(path))
        if img is not None:
            cdfs.append(_compute_cdf(img))

    if not cdfs:
        raise ValueError("No images could be loaded for reference computation.")

    cdf_stack = np.stack(cdfs, axis=0)  # (N, 256)
    median_cdf = np.median(cdf_stack, axis=0)  # (256,)
    print(f"[reference] median CDF built from {len(cdfs)} frames.")
    return median_cdf


# ─────────────────────────────────────────────────────────────
# 2.  Histogram matching
# ─────────────────────────────────────────────────────────────

def _build_lut(src_cdf: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    """
    Builds a uint8 look-up table that maps source intensities to reference
    intensities by aligning their CDFs.

    Fully vectorised — no Python loop over 256 values.
    """
    # diff[i, j] = |ref_cdf[i] - src_cdf[j]|
    diff = np.abs(ref_cdf[:, None] - src_cdf[None, :])  # (256, 256)
    lut = np.argmin(diff, axis=0).astype(np.uint8)  # (256,)
    return lut


def match_to_reference_cdf(
        image: np.ndarray,
        ref_cdf: np.ndarray,
) -> np.ndarray:
    """
    Matches a single grayscale frame to the reference CDF.

    Parameters:
        image   : uint8 grayscale numpy array
        ref_cdf : (256,) normalised CDF from build_median_cdf_reference()

    Returns:
        uint8 grayscale numpy array with matched histogram
    """
    src_cdf = _compute_cdf(image)
    lut = _build_lut(src_cdf, ref_cdf)
    return cv2.LUT(image, lut)


# ─────────────────────────────────────────────────────────────
# 3.  CLAHE
# ─────────────────────────────────────────────────────────────

def apply_clahe(
        image: np.ndarray,
        clip_limit: float = 2.0,
        grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Applies CLAHE with a guard: tile size is clamped so that at least one
    full tile fits in both dimensions.  Without this guard OpenCV raises
    an obscure assertion error on small images.
    """
    h, w = image.shape[:2]
    safe_grid = (
        min(grid_size[0], w),
        min(grid_size[1], h),
    )
    clahe = cv2.createCLAHE(
        clipLimit=max(0.01, clip_limit),
        tileGridSize=safe_grid,
    )
    return clahe.apply(image)


# ─────────────────────────────────────────────────────────────
# 4.  Bilateral filter
# ─────────────────────────────────────────────────────────────

def apply_bilateral(
        image: np.ndarray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75,
) -> np.ndarray:
    """
    Edge-preserving smoothing.  d=9 is a good default for angiography:
    large enough to kill acquisition noise, small enough not to blur vessel
    edges.  Increase sigma_color if speckle remains after CLAHE.
    """
    return cv2.bilateralFilter(image, d=d,
                               sigmaColor=sigma_color,
                               sigmaSpace=sigma_space)


# ─────────────────────────────────────────────────────────────
# 5.  Full preprocessing pipeline (single frame)
# ─────────────────────────────────────────────────────────────

def preprocess_frame(
        image: np.ndarray,
        ref_cdf: np.ndarray,
        clahe_clip: float = 2.0,
        clahe_grid: Tuple[int, int] = (8, 8),
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75,
        bilateral_sigma_space: float = 75,
) -> np.ndarray:
    """
    Full Stage-2 preprocessing for a single angiography frame.

    Steps:
        1. Histogram matching  (inter-series normalisation via median CDF)
        2. CLAHE               (local contrast enhancement)
        3. Bilateral filter    (edge-preserving denoising)

    Parameters:
        image    : raw uint8 grayscale frame
        ref_cdf  : precomputed median CDF from build_median_cdf_reference()

    Returns:
        preprocessed uint8 grayscale numpy array
    """
    matched = match_to_reference_cdf(image, ref_cdf)
    enhanced = apply_clahe(matched,
                           clip_limit=clahe_clip,
                           grid_size=clahe_grid)
    denoised = apply_bilateral(enhanced,
                               d=bilateral_d,
                               sigmaColor=bilateral_sigma_color,
                               sigmaSpace=bilateral_sigma_space)
    return denoised


# ─────────────────────────────────────────────────────────────
# 6.  Batch runner — processes an entire serie
# ─────────────────────────────────────────────────────────────

def preprocess_serie(
        serie_paths: List[str],
        ref_cdf: np.ndarray,
        **kwargs,
) -> List[dict]:
    """
    Runs preprocess_frame() on every frame of a serie.

    Returns:
        List of dicts, one per successfully loaded frame:
            'path'      : original file path
            'original'  : raw uint8 array
            'processed' : preprocessed uint8 array
    """
    results = []
    for path in serie_paths:
        image = load_image(path)
        if image is None:
            print(f"  [WARN] Could not load: {path}")
            continue
        processed = preprocess_frame(image, ref_cdf, **kwargs)
        results.append({
            'path': path,
            'original': image,
            'processed': processed,
        })
    return results
