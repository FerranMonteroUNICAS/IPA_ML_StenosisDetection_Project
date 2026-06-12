"""
Temporal Background Subtraction Dataset Generator
==================================================
For each serie in the organized dataset, computes the mean image across
all frames and subtracts it from every individual frame. Static structures
(ribs, spine, diaphragm) that are present in every frame cancel out,
leaving only the time-varying contrast agent signal.

Input structure:
    organized_dataset/
    └── <patientID>/
        └── <serieID>/
            ├── 14_<patientID>_<serieID>_<frame>.png
            └── 14_<patientID>_<serieID>_<frame>.xml

Output structure (identical hierarchy):
    dataset_subtracted/
    └── <patientID>/
        └── <serieID>/
            ├── 14_<patientID>_<serieID>_<frame>.png   ← temporally subtracted
            └── 14_<patientID>_<serieID>_<frame>.xml   ← copied as-is

Usage:
    python subtraction_dataset.py
    python subtraction_dataset.py --dry-run
"""

import sys
import shutil
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────
# Paths  — edit these to match your setup
# ─────────────────────────────────────────────

INPUT_DIR = Path(
    r"C:\Users\ferra\MIC\1r_any_UNICAS\2n_Semestre\Image_Processing_and_Analysis\project\MIC_project\Proposal_StenosisDetection\organized_dataset")
OUTPUT_DIR = Path(
    r"C:\Users\ferra\MIC\1r_any_UNICAS\2n_Semestre\Image_Processing_and_Analysis\project\MIC_project\Proposal_StenosisDetection\dataset_subtracted")


# ─────────────────────────────────────────────
# Serie grouping
# ─────────────────────────────────────────────

def collect_series(input_dir: Path) -> dict:
    """
    Recursively scans input_dir for .png files and groups them by
    their patientID/serieID folder, returning a sorted dict:
        'patientID/serieID' → sorted list of Path objects
    """
    groups = defaultdict(list)
    for f in sorted(input_dir.rglob("*.png")):
        serie_key = f"{f.parts[-3]}/{f.parts[-2]}"
        groups[serie_key].append(f)
    return {k: sorted(v) for k, v in sorted(groups.items())}


# ─────────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────────

def compute_serie_mean(serie_paths: list) -> np.ndarray:
    """
    Loads all frames of a serie and returns their mean as a float32 array.
    Skips unreadable files with a warning.
    """
    stack = []
    for path in serie_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            stack.append(img.astype(np.float32))
        else:
            print(f"  [WARN] Could not load: {path.name}")

    if not stack:
        raise ValueError("No frames could be loaded for this serie.")

    return np.mean(stack, axis=0)


def temporal_subtract(image: np.ndarray, mean_image: np.ndarray) -> np.ndarray:
    """
    Subtracts the serie mean from a single frame and rescales to [0, 255].
    """
    residual = image.astype(np.float32) - mean_image
    r_min, r_max = residual.min(), residual.max()
    if r_max > r_min:
        return ((residual - r_min) / (r_max - r_min) * 255).astype(np.uint8)
    return np.zeros_like(image)


# ─────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────

def build_subtracted_dataset(input_dir: Path, output_dir: Path, dry_run: bool = False) -> dict:
    """
    Iterates over all series in input_dir, applies temporal mean subtraction
    to every frame, saves results to output_dir preserving folder structure,
    and copies paired XML annotations.
    """
    series = collect_series(input_dir)
    total = len(series)
    summary = {"series_ok": 0, "series_skipped": 0,
               "images_saved": 0, "xmls_copied": 0, "errors": 0}

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing {total} series from: {input_dir}\n")

    for i, (serie_key, serie_paths) in enumerate(series.items(), 1):
        patient_id, serie_id = serie_key.split("/")
        dest_dir = output_dir / patient_id / serie_id

        print(f"[{i:>3}/{total}] {serie_key}  ({len(serie_paths)} frames)", end=" ... ")

        # Compute mean (skip serie if it fails)
        try:
            mean_img = compute_serie_mean(serie_paths)
        except ValueError as e:
            print(f"SKIP — {e}")
            summary["series_skipped"] += 1
            continue

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        for path in serie_paths:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                summary["errors"] += 1
                continue

            # Save subtracted PNG
            dest_png = dest_dir / path.name
            if not dry_run:
                try:
                    subtracted = temporal_subtract(img, mean_img)
                    cv2.imwrite(str(dest_png), subtracted)
                    summary["images_saved"] += 1
                except Exception as e:
                    print(f"\n  [ERROR] {path.name}: {e}")
                    summary["errors"] += 1
            else:
                summary["images_saved"] += 1

            # Copy paired XML if it exists
            xml_src = path.with_suffix(".xml")
            if xml_src.exists():
                dest_xml = dest_dir / xml_src.name
                if not dry_run:
                    shutil.copy2(xml_src, dest_xml)
                summary["xmls_copied"] += 1

        summary["series_ok"] += 1
        print("done" if not dry_run else "ok (dry run)")

    return summary


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

def print_summary(summary: dict, dry_run: bool):
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{'─' * 50}")
    print(f"{prefix}Done!")
    print(f"  Series processed : {summary['series_ok']}")
    if summary["series_skipped"]:
        print(f"  Series skipped   : {summary['series_skipped']}")
    print(f"  Images saved     : {summary['images_saved']}")
    print(f"  XMLs copied      : {summary['xmls_copied']}")
    if summary["errors"]:
        print(f"  Errors           : {summary['errors']}  ← check output above")
    print(f"{'─' * 50}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply temporal mean subtraction to the organized dataset."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be done without writing any files."
    )
    args = parser.parse_args()

    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory does not exist: {INPUT_DIR}")
        sys.exit(1)

    if INPUT_DIR == OUTPUT_DIR:
        print("[ERROR] Input and output directories must be different.")
        sys.exit(1)

    summary = build_subtracted_dataset(INPUT_DIR, OUTPUT_DIR, dry_run=args.dry_run)
    print_summary(summary, dry_run=args.dry_run)


if __name__ == "__main__":
    main()