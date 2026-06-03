"""Quick test to verify SISAR and RealSRQ datasets are usable."""
import os
import sys

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

import csv
from PIL import Image

DATA_DIR = os.path.join(PROJ_ROOT, "data")


def test_sisar():
    """Verify SISAR dataset: all images in label file exist and load correctly."""
    print("=" * 60)
    print("Testing SISAR dataset...")

    label_path = os.path.join(PROJ_ROOT, "datasets", "sisar_all_clip_sr.txt")
    with open(label_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    errors = []
    loaded = 0
    for i, line in enumerate(lines):
        fields = line.split('\t')
        if len(fields) < 2:
            continue
        img_name = fields[0]
        mos = float(fields[1])

        # Try to find image
        found = False
        # Check SR_images flat dir
        flat_path = os.path.join(DATA_DIR, "SISAR", "SR_images", img_name)
        if os.path.exists(flat_path):
            try:
                img = Image.open(flat_path).convert("RGB")
                img.verify()
                loaded += 1
                found = True
            except Exception as e:
                errors.append(f"Line {i+1}: {img_name} - load error: {e}")
            continue

        # Check SR_Images subdirs
        for root, dirs, files in os.walk(os.path.join(DATA_DIR, "SISAR")):
            if img_name in files:
                full = os.path.join(root, img_name)
                try:
                    img = Image.open(full).convert("RGB")
                    img.verify()
                    loaded += 1
                    found = True
                except Exception as e:
                    errors.append(f"Line {i+1}: {img_name} - load error: {e}")
                break
            if found:
                break

        if not found:
            errors.append(f"Line {i+1}: {img_name} - FILE NOT FOUND")

    print(f"  Total entries: {len(lines)}")
    print(f"  Successfully loaded: {loaded}")
    print(f"  Errors: {len(errors)}")
    if errors:
        for e in errors[:10]:
            print(f"    {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")
        return False
    else:
        print("  SISAR: ALL OK - dataset is usable!")
        return True


def test_realsrq():
    """Verify RealSRQ dataset: all images in CSV exist and load correctly."""
    print("\n" + "=" * 60)
    print("Testing RealSRQ dataset...")

    csv_path = os.path.join(DATA_DIR, "RealSRQ", "mos_with_name.csv")
    base = os.path.join(DATA_DIR, "RealSRQ")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    errors = []
    loaded = 0
    for i, row in enumerate(rows):
        if len(row) < 4:
            continue
        mos = float(row[0].strip())
        for j, col_name in [(1, "LR"), (2, "SR"), (3, "HR")]:
            img_path_rel = row[j].strip()
            full = os.path.join(base, img_path_rel).replace('\\', '/')
            if os.path.exists(full):
                try:
                    img = Image.open(full).convert("RGB")
                    img.verify()
                    loaded += 1
                except Exception as e:
                    errors.append(f"Line {i+2}: {img_path_rel} - load error: {e}")
            else:
                errors.append(f"Line {i+2}: {img_path_rel} - FILE NOT FOUND")

    print(f"  Total CSV rows: {len(rows)}")
    print(f"  Total image references: {len(rows) * 3}")
    print(f"  Successfully loaded: {loaded}")
    print(f"  Errors: {len(errors)}")
    if errors:
        for e in errors[:10]:
            print(f"    {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")
        return False
    else:
        print("  RealSRQ: ALL OK - dataset is usable!")
        return True


if __name__ == "__main__":
    s_ok = test_sisar()
    r_ok = test_realsrq()
    print("\n" + "=" * 60)
    if s_ok and r_ok:
        print("Both datasets verified: ALL OK!")
    else:
        status = []
        if not s_ok:
            status.append("SISAR: FAILED")
        if not r_ok:
            status.append("RealSRQ: FAILED")
        print(f"Some checks failed: {', '.join(status)}")
        sys.exit(1)
