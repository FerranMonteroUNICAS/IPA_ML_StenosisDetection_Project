"""
Parallelized Advanced Pipeline Dataset Generator
================================================
Uses concurrent multi-processing to distribute patient series across all
available CPU cores, significantly accelerating dataset generation.

Source Layout:
    dataset_subtracted/
    └── <patientID>/
        └── <serieID>/
            ├── <frame>.png
            └── <frame>.xml

Destination Layout:
    dataset_processed/
    └── <patientID>/
        └── <serieID>/
            ├── <frame>_processed.png
            ├── <frame>_mask.png
            ├── <frame>_skeleton.png
            └── <frame>_bbox.xml
"""

import os
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from skimage.measure import label
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# Dynamic Import of Project Modules from 'src'
# ─────────────────────────────────────────────────────────────
ROOT_DIR = r"C:\Users\ferra\MIC\1r_any_UNICAS\2n_Semestre\Image_Processing_and_Analysis\project\MIC_project\Proposal_StenosisDetection"
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
try:
    from src.preprocessing import preprocess_bs_nlm_clahe
    from src.vessel_segmentation_ferran import apply_frangi, apply_cleaned_hysteresis
except ImportError as e:
    print(f"[CRITICAL ERROR] Failed to import core functions from 'src': {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# Hyperparameters Configuration
# ─────────────────────────────────────────────────────────────
KERNEL_SIZE = 81
H = 8
CLIP_LIMIT = 4.0
SCALE_MIN = 4
SCALE_MAX = 15
LOW_THRESH = 0.02
HIGH_THRESH = 0.18
MIN_SIZE = 200
DISTANCIA_MAX_UNION = 25.0

INPUT_DIR = Path(os.path.join(ROOT_DIR, "dataset_subtracted"))
OUTPUT_DIR = Path(os.path.join(ROOT_DIR, "dataset_processed"))


# ─────────────────────────────────────────────────────────────
# Morphological Optimization Block
# ─────────────────────────────────────────────────────────────

def optimize_skeleton(bool_mask):
    """
    Extracts geometric medial axes, analyzes endpoints, and bridges
    disconnected gaps within the specified geometric tolerance.
    """
    skeleton_raw = skeletonize(bool_mask)
    labeled_skel, num_features = label(skeleton_raw, return_num=True)

    if num_features > 0:
        pixel_counts = np.bincount(labeled_skel.ravel())
        pixel_counts[0] = 0
        sorted_indices = np.argsort(pixel_counts)
        idx_primero = sorted_indices[-1]
        pixels_primero = pixel_counts[idx_primero]

        if num_features >= 2:
            idx_segundo = sorted_indices[-2]
            pixels_segundo = pixel_counts[idx_segundo]

            if pixels_segundo > 0.2 * pixels_primero:
                indices_validos = [idx_primero, idx_segundo]
            else:
                indices_validos = [idx_primero]
        else:
            indices_validos = [idx_primero]

        skeleton_orig = np.isin(labeled_skel, indices_validos)
    else:
        skeleton_orig = skeleton_raw.copy()

    kernel_vecinos = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], dtype=np.uint8)

    conteo_vecinos = cv2.filter2D(skeleton_orig.astype(np.uint8), -1, kernel_vecinos)
    endpoints_mask = (conteo_vecinos == 1) & skeleton_orig

    y_ends, x_ends = np.where(endpoints_mask)
    lista_endpoints = list(zip(x_ends, y_ends))

    lineas_union = np.zeros_like(skeleton_orig, dtype=np.uint8)

    for i in range(len(lista_endpoints)):
        for j in range(i + 1, len(lista_endpoints)):
            pt1 = lista_endpoints[i]
            pt2 = lista_endpoints[j]

            dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

            if 2.0 < dist <= DISTANCIA_MAX_UNION:
                cv2.line(lineas_union, pt1, pt2, color=1, thickness=1)

    skeleton_final = (skeleton_orig | (lineas_union > 0))

    plt.imshow(skeleton_final)

    return skeleton_final


# ─────────────────────────────────────────────────────────────
# Worker Function for Parallel Processing
# ─────────────────────────────────────────────────────────────

def process_single_serie(serie_key, path_list_strings):
    """
    Worker function executed by individual CPU processes.
    Takes string paths to avoid complex serialization issues across processes.
    """
    patient_id, serie_id = serie_key.split("/")
    target_destination_dir = OUTPUT_DIR / patient_id / serie_id
    target_destination_dir.mkdir(parents=True, exist_ok=True)

    local_img_count = 0
    local_xml_count = 0

    for path_str in path_list_strings:
        frame_path = Path(path_str)
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        base_stem = frame_path.stem

        try:
            # Running the sequential processing steps
            proc = preprocess_bs_nlm_clahe(img, kernel_size=KERNEL_SIZE, h=H, clip_limit=CLIP_LIMIT)
            frangi_map = apply_frangi(proc, scale_min=SCALE_MIN, scale_max=SCALE_MAX, black_ridges=True)
            bool_mask = apply_cleaned_hysteresis(frangi_map, low_thresh=LOW_THRESH, high_thresh=HIGH_THRESH,
                                                 min_size=MIN_SIZE)
            skeleton = optimize_skeleton(bool_mask)

            mask_uint8 = (bool_mask * 255).astype(np.uint8)
            skeleton_uint8 = (skeleton * 255).astype(np.uint8)

            # File Outputs
            cv2.imwrite(str(target_destination_dir / f"{base_stem}_processed.png"), proc)
            cv2.imwrite(str(target_destination_dir / f"{base_stem}_mask.png"), mask_uint8)
            cv2.imwrite(str(target_destination_dir / f"{base_stem}_skeleton.png"), skeleton_uint8)
            local_img_count += 1

        except Exception as e:
            print(f"\n  [ERROR] Processing failure at {frame_path.name}: {e}")
            continue

        # Check and clone matching XML structures
        xml_source_file = frame_path.with_suffix(".xml")
        if xml_source_file.exists():
            xml_destination_file = target_destination_dir / f"{base_stem}_bbox.xml"
            shutil.copy2(xml_source_file, xml_destination_file)
            local_xml_count += 1

    return serie_key, local_img_count, local_xml_count


# ─────────────────────────────────────────────────────────────
# Orchestration Handler
# ─────────────────────────────────────────────────────────────

def collect_series_groups(input_dir: Path) -> dict:
    groups = defaultdict(list)
    for file_path in sorted(input_dir.rglob("*.png")):
        key = f"{file_path.parts[-3]}/{file_path.parts[-2]}"
        groups[key].append(str(file_path))  # Storing strings for cleaner parallel execution workers
    return {k: sorted(v) for k, v in sorted(groups.items())}


def execute_pipeline():
    if not INPUT_DIR.exists():
        print(f"[CRITICAL ERROR] Target input directory not detected: {INPUT_DIR}")
        sys.exit(1)

    series_map = collect_series_groups(INPUT_DIR)
    total_series = len(series_map)

    # os.cpu_count() auto-detects all available cores/threads on your workstation
    #max_workers = os.cpu_count()
    max_workers = 4
    print(f"Loaded {total_series} series from: {INPUT_DIR}")
    print(f"Spawning ProcessPool Executor using {max_workers} parallel CPU cores...\n")

    images_processed_count = 0
    xml_copied_count = 0
    completed_series = 0

    # Initialize concurrent process framework
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the pool
        future_to_serie = {
            executor.submit(process_single_serie, key, paths): key
            for key, paths in series_map.items()
        }

        # Monitor completed tasks dynamically
        for future in as_completed(future_to_serie):
            serie_key = future_to_serie[future]
            try:
                key, img_count, xml_count = future.result()
                images_processed_count += img_count
                xml_copied_count += xml_count
                completed_series += 1

                # Dynamic terminal counter tracking progress
                print(f"[{completed_series:>3}/{total_series}] Finished Node: {key} (+{img_count} frames processed)")
            except Exception as e:
                print(f"[CRITICAL ENGINE FAULT] Serie {serie_key} generated an execution error: {e}")

    print("\n" + "=" * 50)
    print(" PARALLEL PIPELINE RUN COMPLETE")
    print("=" * 50)
    print(f" Total Multi-modal Frame Triplets Exported : {images_processed_count}")
    print(f" Total Ground-truth XML Files Cloned       : {xml_copied_count}")
    print(f" Target Output Location                    : {OUTPUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Windows requires the main guard protection block to spawn sub-processes safely
    execute_pipeline()