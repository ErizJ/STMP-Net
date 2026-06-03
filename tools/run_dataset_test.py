"""Load SISAR and RealSRQ datasets through actual project infrastructure."""
import os
import sys

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from PIL import Image
import csv
import numpy as np

# Load config and imports
from config import get_config
import argparse

DATA_DIR = os.path.join(PROJ_ROOT, "data")


def test_sisar_dataloader():
    """Test SISAR through the actual dataset class."""
    print("=" * 60)
    print("SISAR: Testing through SISARDATASET_clip...")

    from datasets.iqa_dataset_clip import SISARDATASET_clip, transfer

    root = os.path.join(DATA_DIR, "SISAR", "sr_images_flat")
    index = list(range(8428))  # All images
    patch_num = 1

    try:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset = SISARDATASET_clip(
            root=root,
            index=index,
            patch_num=patch_num,
            transform=transform,
        )

        print(f"  Dataset length: {len(dataset)}")
        assert len(dataset) == 8428 * patch_num, f"Expected {8428 * patch_num}, got {len(dataset)}"

        # Test loading first 10 samples
        mos_values = []
        for i in range(min(10, len(dataset))):
            sample, mos, scene, texture, structure, distortion = dataset[i]
            mos_values.append(mos)
            assert sample.shape == (3, 224, 224), f"Bad shape: {sample.shape}"

        print(f"  Sample shape: {sample.shape}")
        print(f"  MOS sample: {mos_values[:5]}")
        print(f"  Scene: {scene}, Texture: {texture}, Structure: {structure}, Dist: {distortion}")
        print("  SISAR dataloader: ALL OK!")
        return True

    except Exception as e:
        print(f"  SISAR ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_realsrq_files():
    """Verify all RealSRQ images via the CSV paths."""
    print("\n" + "=" * 60)
    print("RealSRQ: Verifying all images...")

    csv_path = os.path.join(DATA_DIR, "RealSRQ", "mos_with_name.csv")
    base = os.path.join(DATA_DIR, "RealSRQ")

    total = 0
    errors = []
    mos_values = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            mos = float(row[0].strip())
            mos_values.append(mos)
            for j in [1, 2, 3]:
                rel_path = row[j].strip()
                full = os.path.join(base, rel_path).replace('\\', '/')
                if not os.path.exists(full):
                    errors.append(f"{rel_path}")
                    continue
                try:
                    img = Image.open(full)
                    img.verify()
                    total += 1
                except Exception as e:
                    errors.append(f"{rel_path}: {e}")

    print(f"  Entries: {len(mos_values)}")
    print(f"  Images verified: {total}/{len(mos_values) * 3}")
    print(f"  MOS range: [{min(mos_values):.4f}, {max(mos_values):.4f}]")
    if errors:
        print(f"  ERRORS: {len(errors)}")
        for e in errors[:5]:
            print(f"    {e}")
        return False
    else:
        print("  RealSRQ: ALL OK!")
        return True


def test_sisar_raw():
    """Also verify SISAR MOS vs label consistency and mat availability."""
    print("\n" + "=" * 60)
    print("SISAR MOS-label cross-check...")

    mos_path = os.path.join(DATA_DIR, "SISAR", "MOS_with_name.txt")
    label_path = os.path.join(PROJ_ROOT, "datasets", "sisar_all_clip_sr.txt")
    sr_dir = os.path.join(DATA_DIR, "SISAR", "sr_images_flat")

    mos_data = {}
    with open(mos_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('#')
            if len(parts) >= 2:
                mos_data[parts[0]] = float(parts[1])

    label_data = {}
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            fields = line.split('\t')
            if len(fields) >= 2:
                label_data[fields[0]] = float(fields[1])

    common = set(mos_data.keys()) & set(label_data.keys())

    # Verify MOS correspondence
    diffs = [abs(mos_data[k] - label_data[k]) for k in common]
    max_diff = max(diffs) if diffs else 1.0
    big_diffs = [d for d in diffs if d > 0.01]

    print(f"  MOS entries: {len(mos_data)}")
    print(f"  Label entries: {len(label_data)}")
    print(f"  Common: {len(common)}")
    print(f"  Max MOS diff: {max_diff:.6f}")
    print(f"  Diffs > 0.01: {len(big_diffs)}")

    # Verify all label images exist
    missing = [k for k in label_data if not os.path.exists(os.path.join(sr_dir, k))]
    print(f"  Label images missing: {len(missing)}/{len(label_data)}")

    if max_diff > 0.1:
        print("  WARNING: Large MOS discrepancies found!")
        return False
    elif missing:
        print("  WARNING: Missing images!")
        return False
    else:
        print("  SISAR MOS-label: ALL OK!")
        return True


if __name__ == "__main__":
    r1 = test_sisar_raw()
    r2 = test_sisar_dataloader()
    r3 = test_realsrq_files()

    print("\n" + "=" * 60)
    results = {"SISAR MOS": r1, "SISAR Loader": r2, "RealSRQ": r3}
    passed = sum(1 for v in results.values() if v)
    print(f"Results: {passed}/{len(results)} tests passed")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}")
