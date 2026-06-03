"""Verify all dataset CSV/label files: image existence, MOS ranges, name consistency."""
import os
import sys
import csv

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")

def check_image_exists(base_dir, img_path):
    """Check if an image file exists, trying multiple possible base dirs."""
    full = os.path.join(base_dir, img_path).replace('\\', '/')
    if os.path.exists(full):
        return True
    # Try flat filename lookup in SR_images subdir
    flat_base = os.path.join(base_dir, "SR_images")
    flat = os.path.join(flat_base, os.path.basename(img_path))
    if os.path.exists(flat):
        return True
    # Try with .png fallback
    png = full.rsplit('.', 1)[0] + '.png' if '.' in full else full + '.png'
    if os.path.exists(png):
        return True
    return False


def verify_csv(path, img_cols, base_dir, label="", delim=',', skip_header=True, mos_col=0):
    """Generic CSV verifier. img_cols is list of column indices containing image paths."""
    errors = []
    missing = []
    mos_values = []
    total = 0

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if skip_header and i == 0:
                continue
            if not row or all(c.strip() == '' for c in row):
                continue
            total += 1
            # Check image columns
            for col in img_cols:
                if col < len(row) and row[col].strip():
                    img_path = row[col].strip()
                    if not check_image_exists(base_dir, img_path):
                        missing.append((i + 1, img_path))
            # Check MOS
            if mos_col < len(row) and row[mos_col].strip():
                try:
                    mos = float(row[mos_col].strip())
                    mos_values.append(mos)
                except ValueError:
                    errors.append(f"  Line {i+1}: invalid MOS '{row[mos_col]}'")

    print(f"\n{'='*60}")
    print(f"Dataset: {label}")
    print(f"  File: {path}")
    print(f"  Total entries: {total}")
    print(f"  MOS range: [{min(mos_values):.4f}, {max(mos_values):.4f}]" if mos_values else "  No MOS values")
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(e)
    else:
        print(f"  MOS parsing: OK")
    if missing:
        print(f"  MISSING IMAGES ({len(missing)}):")
        for m in missing[:15]:
            print(f"    Line {m[0]}: {m[1]}")
        if len(missing) > 15:
            print(f"    ... and {len(missing) - 15} more")
    else:
        print(f"  Image existence: ALL OK")
    return total, len(missing), len(errors)


def verify_sisar_txt(path, base_dir):
    """Verify SISAR MOS_with_name.txt (format: filename#MOS)."""
    missing = []
    mos_values = []
    total = 0
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total += 1
            parts = line.split('#')
            if len(parts) >= 2:
                img_name = parts[0].strip()
                try:
                    mos = float(parts[1].strip())
                    mos_values.append(mos)
                except ValueError:
                    pass
                if not check_image_exists(base_dir, img_name):
                    missing.append((i + 1, img_name))

    print(f"\n{'='*60}")
    print(f"Dataset: SISAR (MOS_with_name.txt)")
    print(f"  File: {path}")
    print(f"  Total entries: {total}")
    print(f"  MOS range: [{min(mos_values):.4f}, {max(mos_values):.4f}]" if mos_values else "  No MOS")
    if missing:
        print(f"  MISSING IMAGES ({len(missing)}):")
        for m in missing[:15]:
            print(f"    Line {m[0]}: {m[1]}")
        if len(missing) > 15:
            print(f"    ... and {len(missing) - 15} more")
    else:
        print(f"  Image existence: ALL OK")
    return total, len(missing), 0


def verify_label_txt(path, base_dir, label, delim='\t'):
    """Verify datasets/*.txt label files (format: img_path MOS ...)."""
    missing = []
    mos_values = []
    total = 0
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            total += 1
            fields = line.split(delim)
            if len(fields) >= 2:
                img_name = fields[0].strip()
                try:
                    mos = float(fields[1].strip())
                    mos_values.append(mos)
                except ValueError:
                    pass
                if not check_image_exists(base_dir, img_name):
                    missing.append((i + 1, img_name))

    print(f"\n{'='*60}")
    print(f"Dataset: {label}")
    print(f"  File: {path}")
    print(f"  Total entries: {total}")
    print(f"  MOS range: [{min(mos_values):.4f}, {max(mos_values):.4f}]" if mos_values else "  No MOS")
    if missing:
        print(f"  MISSING IMAGES ({len(missing)}):")
        for m in missing[:15]:
            print(f"    Line {m[0]}: {m[1]}")
        if len(missing) > 15:
            print(f"    ... and {len(missing) - 15} more")
    else:
        print(f"  Image existence: ALL OK")
    return total, len(missing), 0


def main():
    print("Verifying all dataset CSV/label files...")
    all_ok = True

    # ---- CVIU17 ----
    verify_csv(
        os.path.join(DATA_DIR, "cviu17", "mos_with_names.csv"),
        img_cols=[1], base_dir=os.path.join(DATA_DIR, "cviu17", "SRimages"),
        label="CVIU17", mos_col=0
    )
    verify_label_txt(
        os.path.join(DATASETS_DIR, "cviu17_all_clip_sr.txt"),
        base_dir=os.path.join(DATA_DIR, "cviu17", "SRimages"),
        label="CVIU17 (label txt)"
    )

    # ---- QADS ----
    verify_csv(
        os.path.join(DATA_DIR, "QADS", "mos_with_names.csv"),
        img_cols=[1, 2], base_dir=os.path.join(DATA_DIR, "QADS"),
        label="QADS", mos_col=0
    )
    verify_label_txt(
        os.path.join(DATASETS_DIR, "qads_all_clip_sr.txt"),
        base_dir=os.path.join(DATA_DIR, "QADS"),
        label="QADS (label txt)"
    )

    # ---- SISAR ----
    verify_sisar_txt(
        os.path.join(DATA_DIR, "SISAR", "MOS_with_name.txt"),
        base_dir=os.path.join(DATA_DIR, "SISAR"),
    )
    verify_label_txt(
        os.path.join(DATASETS_DIR, "sisar_all_clip_sr.txt"),
        base_dir=os.path.join(DATA_DIR, "SISAR"),
        label="SISAR (label txt)"
    )

    # ---- RealSRQ ----
    verify_csv(
        os.path.join(DATA_DIR, "RealSRQ", "mos_with_name.csv"),
        img_cols=[1, 2, 3], base_dir=os.path.join(DATA_DIR, "RealSRQ"),
        label="RealSRQ", mos_col=0
    )

    # ---- Waterloo15 ----
    verify_csv(
        os.path.join(DATA_DIR, "Waterloo15", "mos_with_names.csv"),
        img_cols=[0, 2], base_dir=os.path.join(DATA_DIR, "Waterloo15", "WIND_all"),
        label="Waterloo15", mos_col=1
    )


if __name__ == "__main__":
    main()
