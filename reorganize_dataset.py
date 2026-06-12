"""
Dataset Reorganizer for Stenosis Detection
==========================================
Converts flat dataset with naming pattern:
    14_<patientID>_<serieID>_<frame>.bmp / .xml
Into structured hierarchy:
    output_root/
    └── <patientID>/
        └── <serieID>/
            ├── 14_<patientID>_<serieID>_<frame>.png
            └── 14_<patientID>_<serieID>_<frame>.xml

Usage:
    python reorganize_dataset.py --input /path/to/flat/dataset --output /path/to/output
    python reorganize_dataset.py --input ./data --output ./dataset_structured --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image
except ImportError:
    print("[ERROR] Pillow is not installed. Run: pip install Pillow")
    sys.exit(1)


# ─────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────

def parse_filename(stem: str) -> tuple[str, str, str] | None:
    """
    Parses a filename stem of the form:  14_patientID_serieID_frame
    Returns (patient_id, serie_id, frame) or None if pattern doesn't match.
    Handles patientID / serieID / frame that may contain underscores.
    Strategy: split on '_', take [0] as prefix, [1] as patient, [2] as serie, [3:] as frame.
    """
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    # parts[0] = "14" (prefix), parts[1] = patientID, parts[2] = serieID, rest = frame
    patient_id = parts[1]
    serie_id   = parts[2]
    frame      = "_".join(parts[3:])
    return patient_id, serie_id, frame


# ─────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────

def collect_files(input_dir: Path) -> dict:
    """
    Scans input_dir (non-recursively) for .bmp and .xml files.
    Returns a dict keyed by stem → {'bmp': Path|None, 'xml': Path|None, 'parsed': tuple|None}
    """
    records = defaultdict(lambda: {"bmp": None, "xml": None, "parsed": None})

    for f in input_dir.iterdir():
        if not f.is_file():
            continue
        suffix = f.suffix.lower()
        if suffix not in (".bmp", ".xml"):
            continue

        stem = f.stem
        parsed = parse_filename(stem)
        if parsed is None:
            print(f"  [SKIP] Could not parse filename: {f.name}")
            continue

        records[stem]["parsed"] = parsed
        if suffix == ".bmp":
            records[stem]["bmp"] = f
        elif suffix == ".xml":
            records[stem]["xml"] = f

    return records


def reorganize(input_dir: Path, output_dir: Path, dry_run: bool = False) -> dict:
    """
    Main reorganization function.
    Returns a summary dict with counts.
    """
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Scanning: {input_dir}")
    records = collect_files(input_dir)

    if not records:
        print("  No matching files found.")
        return {}

    summary = {"patients": set(), "series": set(), "converted": 0, "xml_copied": 0,
                "skipped_bmp": 0, "skipped_xml": 0, "errors": 0}

    for stem, info in sorted(records.items()):
        patient_id, serie_id, frame = info["parsed"]
        summary["patients"].add(patient_id)
        summary["series"].add(f"{patient_id}/{serie_id}")

        # Build destination folder
        dest_dir = output_dir / patient_id / serie_id
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        # ── BMP → PNG conversion ──────────────────────────────
        if info["bmp"]:
            png_name = stem + ".png"
            dest_png = dest_dir / png_name
            if not dry_run:
                try:
                    with Image.open(info["bmp"]) as img:
                        img.save(dest_png, format="PNG")
                    summary["converted"] += 1
                except Exception as e:
                    print(f"  [ERROR] Converting {info['bmp'].name}: {e}")
                    summary["errors"] += 1
            else:
                print(f"  [DRY] {info['bmp'].name} → {dest_png.relative_to(output_dir)}")
                summary["converted"] += 1
        else:
            print(f"  [WARN] No BMP for stem: {stem}")
            summary["skipped_bmp"] += 1

        # ── XML copy ─────────────────────────────────────────
        if info["xml"]:
            dest_xml = dest_dir / info["xml"].name
            if not dry_run:
                try:
                    shutil.copy2(info["xml"], dest_xml)
                    summary["xml_copied"] += 1
                except Exception as e:
                    print(f"  [ERROR] Copying {info['xml'].name}: {e}")
                    summary["errors"] += 1
            else:
                print(f"  [DRY] {info['xml'].name} → {dest_xml.relative_to(output_dir)}")
                summary["xml_copied"] += 1
        else:
            print(f"  [WARN] No XML for stem: {stem}")
            summary["skipped_xml"] += 1

    return summary


def print_summary(summary: dict):
    print(f"\n{'─'*50}")
    print(f"{prefix}Done!")
    print(f"  Patients found   : {len(summary.get('patients', []))}")
    print(f"  Series found     : {len(summary.get('series', []))}")
    print(f"  BMPs converted   : {summary.get('converted', 0)}")
    print(f"  XMLs copied      : {summary.get('xml_copied', 0)}")
    if summary.get("skipped_bmp"):
        print(f"  Stems w/o BMP    : {summary['skipped_bmp']}")
    if summary.get("skipped_xml"):
        print(f"  Stems w/o XML    : {summary['skipped_xml']}")
    if summary.get("errors"):
        print(f"  Errors           : {summary['errors']}  ← check output above")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():

    input_dir  = Path(r"C:\Users\ferra\MIC\1r_any_UNICAS\2n_Semestre\Image_Processing_and_Analysis\project\MIC_project\Proposal_StenosisDetection\dataset").resolve()
    output_dir = Path(r"C:\Users\ferra\MIC\1r_any_UNICAS\2n_Semestre\Image_Processing_and_Analysis\project\MIC_project\Proposal_StenosisDetection\organized_dataset").resolve()

    if not input_dir.exists():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)

    if input_dir == output_dir:
        print("[ERROR] Input and output directories must be different.")
        sys.exit(1)

    summary = reorganize(input_dir, output_dir)
    if summary:
        print_summary(summary)


if __name__ == "__main__":
    main()
