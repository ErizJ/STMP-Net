"""Comprehensive check: image-score correspondence, mat files, data integrity across all datasets."""
import os
import sys
import csv
import scipy.io as sio
import numpy as np
from PIL import Image
from collections import defaultdict

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_ROOT, "data")
DS_DIR = os.path.join(PROJ_ROOT, "datasets")


def load_mat(path):
    """Load a .mat file and return dict of arrays."""
    try:
        data = sio.loadmat(path)
        return {k: v for k, v in data.items() if not k.startswith('__')}
    except Exception as e:
        print(f"  ERROR loading mat: {e}")
        return {}


def check_cviu17():
    """Check CVIU17: CSV, mat file, label txt consistency."""
    print("=" * 60)
    print("CVIU17")
    print("=" * 60)

    csv_path = os.path.join(DATA_DIR, "cviu17", "mos_with_names.csv")
    mat_path = os.path.join(DATA_DIR, "cviu17", "sr_metric_data.mat")
    mat_path2 = os.path.join(DATA_DIR, "cviu17", "sr_metric_data", "sr_metric_data.mat")
    label_path = os.path.join(DS_DIR, "cviu17_all_clip_sr.txt")
    img_base = os.path.join(DATA_DIR, "cviu17", "SRimages")

    # 1. Read CSV
    csv_data = {}  # img_name -> MOS
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 2 and row[1].strip():
                name = row[1].strip()
                mos = float(row[0].strip())
                csv_data[name] = mos

    print(f"  CSV entries: {len(csv_data)}")
    print(f"  MOS range: [{min(csv_data.values()):.4f}, {max(csv_data.values()):.4f}]")

    # 2. Read label txt
    label_data = {}
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            fields = line.split('\t')
            if len(fields) >= 2:
                name = fields[0].strip()
                mos = float(fields[1].strip())
                label_data[name] = mos

    # 3. Compare CSV vs label
    csv_names = set(csv_data.keys())
    label_names = set(label_data.keys())
    common = csv_names & label_names
    only_csv = csv_names - label_names
    only_label = label_names - csv_names

    print(f"  Label txt entries: {len(label_data)}")
    print(f"  Common names: {len(common)}")
    if only_csv:
        print(f"  Only in CSV: {len(only_csv)}")
        for n in list(only_csv)[:5]:
            print(f"    {n}")
    if only_label:
        print(f"  Only in label txt: {len(only_label)}")
        for n in list(only_label)[:5]:
            print(f"    {n}")

    # Check MOS consistency for common entries
    mos_diffs = []
    for n in common:
        if abs(csv_data[n] - label_data[n]) > 0.01:
            mos_diffs.append((n, csv_data[n], label_data[n]))
    if mos_diffs:
        print(f"  MOS MISMATCH: {len(mos_diffs)} entries")
        for n, c, l in mos_diffs[:5]:
            print(f"    {n}: CSV={c:.4f}, Label={l:.4f}")
    else:
        print(f"  MOS consistency CSV vs Label: OK")

    # 4. Load mat file
    mat_found = False
    for mp in [mat_path, mat_path2]:
        if os.path.exists(mp):
            mat = load_mat(mp)
            print(f"  Mat file: {mp}")
            print(f"  Mat keys: {list(mat.keys())}")
            mat_found = True
            # Check if mat has MOS data
            for key in mat.keys():
                arr = mat[key]
                if arr.size > 0:
                    try:
                        mn, mx = float(arr.min()), float(arr.max())
                        print(f"    {key}: shape={arr.shape}, range=[{mn:.4f}, {mx:.4f}]")
                    except (TypeError, ValueError):
                        print(f"    {key}: shape={arr.shape}, dtype={arr.dtype} (non-numeric)")
            break
    if not mat_found:
        print(f"  Mat file: NOT FOUND")

    # 5. Verify images exist
    missing = []
    ok = 0
    for name in list(label_data.keys())[:100]:  # Sample 100
        full = os.path.join(img_base, name)
        if os.path.exists(full):
            ok += 1
        else:
            missing.append(name)
    print(f"  Image spot-check ({ok}/100 found)")
    if missing:
        for m in missing[:5]:
            print(f"    Missing: {m}")

    return True


def check_qads():
    """Check QADS: CSV, label txt consistency."""
    print("\n" + "=" * 60)
    print("QADS")
    print("=" * 60)

    csv_path = os.path.join(DATA_DIR, "QADS", "mos_with_names.csv")
    label_path = os.path.join(DS_DIR, "qads_all_clip_sr.txt")

    # CSV
    csv_data = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 3:
                sr_name = row[2].strip()  # SR image name
                mos = float(row[0].strip())
                csv_data[sr_name] = mos

    print(f"  CSV entries: {len(csv_data)}")
    print(f"  MOS range: [{min(csv_data.values()):.4f}, {max(csv_data.values()):.4f}]")

    # Label txt
    label_data = {}
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            fields = line.split('\t')
            if len(fields) >= 2:
                name = fields[0].strip()
                mos = float(fields[1].strip())
                label_data[name] = mos

    # Compare
    csv_names = set(csv_data.keys())
    label_names = set(label_data.keys())
    common = csv_names & label_names
    only_csv = csv_names - label_names
    only_label = label_names - csv_names

    print(f"  Label txt entries: {len(label_data)}")
    print(f"  Common names: {len(common)}")
    if only_csv:
        print(f"  Only in CSV: {len(only_csv)}")
    if only_label:
        print(f"  Only in label txt: {len(only_label)}")

    # MOS consistency
    mos_diffs = []
    for n in common:
        if abs(csv_data[n] - label_data[n]) > 0.01:
            mos_diffs.append((n, csv_data[n], label_data[n]))
    if mos_diffs:
        print(f"  MOS MISMATCH: {len(mos_diffs)}")
        for n, c, l in mos_diffs[:5]:
            print(f"    {n}: CSV={c:.4f}, Label={l:.4f}")
    else:
        print(f"  MOS consistency: OK")

    # Check image existence
    sr_img_dir = os.path.join(DATA_DIR, "QADS", "super-resolved_images")
    missing = []
    ok = 0
    for name in list(label_data.keys()):
        # QADS SR images are organized by source image subdirs
        parts = name.split('_')
        if len(parts) >= 2:
            subdir = parts[0]  # e.g., img01
            full = os.path.join(sr_img_dir, subdir, name)
            if os.path.exists(full):
                ok += 1
            else:
                # Try flat
                flat = os.path.join(sr_img_dir, name)
                if os.path.exists(flat):
                    ok += 1
                else:
                    missing.append(name)

    print(f"  Image check: {ok}/{len(label_data)} found, {len(missing)} missing")
    if missing:
        for m in missing[:5]:
            print(f"    Missing: {m}")

    return True


def check_sisar():
    """Check SISAR: MOS txt, label txt consistency."""
    print("\n" + "=" * 60)
    print("SISAR")
    print("=" * 60)

    mos_path = os.path.join(DATA_DIR, "SISAR", "MOS_with_name.txt")
    label_path = os.path.join(DS_DIR, "sisar_all_clip_sr.txt")

    # MOS txt (format: filename#MOS)
    mos_data = {}
    with open(mos_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('#')
            if len(parts) >= 2:
                try:
                    mos_data[parts[0]] = float(parts[1])
                except ValueError:
                    pass

    print(f"  MOS txt entries: {len(mos_data)}")
    print(f"  MOS range: [{min(mos_data.values()):.4f}, {max(mos_data.values()):.4f}]")

    # Label txt
    label_data = {}
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            fields = line.split('\t')
            if len(fields) >= 2:
                label_data[fields[0].strip()] = float(fields[1].strip())

    print(f"  Label txt entries: {len(label_data)}")
    print(f"  Label MOS range: [{min(label_data.values()):.4f}, {max(label_data.values()):.4f}]")

    # Compare
    mos_names = set(mos_data.keys())
    label_names = set(label_data.keys())
    common = mos_names & label_names
    only_mos = mos_names - label_names
    only_label = label_names - mos_names

    print(f"  Common names: {len(common)}")
    print(f"  Only in MOS: {len(only_mos)}")
    print(f"  Only in label: {len(only_label)}")

    # MOS consistency
    mos_diffs = []
    perfect_in_label = 0
    for n in common:
        if abs(mos_data[n] - label_data[n]) > 0.01:
            mos_diffs.append((n, mos_data[n], label_data[n]))
        if abs(label_data[n] - 1.0) < 0.001:
            perfect_in_label += 1

    if mos_diffs:
        print(f"  MOS MISMATCH: {len(mos_diffs)}")
        for n, c, l in mos_diffs[:5]:
            print(f"    {n}: MOS={c:.4f}, Label={l:.4f}")
    else:
        print(f"  MOS consistency: OK")
    print(f"  MOS=1.0 entries (reference images): {perfect_in_label}")

    # Image existence check
    sr_dir = os.path.join(DATA_DIR, "SISAR", "SR_images")
    missing = []
    for name in label_data.keys():
        full = os.path.join(sr_dir, name)
        if not os.path.exists(full):
            missing.append(name)

    print(f"  Label image check: {len(label_data) - len(missing)}/{len(label_data)} found")
    if missing:
        print(f"  Missing: {len(missing)}")
        for m in missing[:5]:
            print(f"    {m}")

    return True


def check_realsrq():
    """Check RealSRQ: CSV, mat files consistency."""
    print("\n" + "=" * 60)
    print("RealSRQ")
    print("=" * 60)

    csv_path = os.path.join(DATA_DIR, "RealSRQ", "mos_with_name.csv")
    bt_path = os.path.join(DATA_DIR, "RealSRQ", "subj_BT_score.mat")
    votes_path = os.path.join(DATA_DIR, "RealSRQ", "subj_votes.mat")

    # CSV
    csv_data = []
    sr_names = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 4:
                mos = float(row[0].strip())
                sr = row[2].strip()  # SR image path
                csv_data.append({'mos': mos, 'lr': row[1], 'sr': sr, 'hr': row[3]})
                sr_names.add(sr)

    print(f"  CSV entries: {len(csv_data)}")
    mos_vals = [r['mos'] for r in csv_data]
    print(f"  MOS range: [{min(mos_vals):.4f}, {max(mos_vals):.4f}]")
    print(f"  Unique SR images: {len(sr_names)}")

    # Check mat files
    for name, path in [("subj_BT_score", bt_path), ("subj_votes", votes_path)]:
        if os.path.exists(path):
            mat = load_mat(path)
            print(f"  {name}.mat: keys={list(mat.keys())}")
            for key in mat.keys():
                arr = mat[key]
                if arr.size > 0:
                    print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}, range=[{arr.min():.4f}, {arr.max():.4f}]")

                    # If it's a score matrix, check dimensions match CSV
                    if arr.ndim == 2:
                        if arr.shape[0] == len(csv_data):
                            print(f"    Row count ({arr.shape[0]}) matches CSV entries ({len(csv_data)}) OK")
                        elif arr.shape[1] == len(csv_data):
                            print(f"    Col count ({arr.shape[1]}) matches CSV entries ({len(csv_data)}) OK")
        else:
            print(f"  {name}.mat: NOT FOUND")

    # Image existence
    base = os.path.join(DATA_DIR, "RealSRQ")
    missing = []
    ok = 0
    for rec in csv_data[:500]:  # Sample
        for key in ['lr', 'sr', 'hr']:
            rel = rec[key]
            full = os.path.join(base, rel).replace('\\', '/')
            if os.path.exists(full):
                ok += 1
            else:
                missing.append(rel)

    print(f"  Image spot-check ({ok}/{min(500, len(csv_data)) * 3} found)")
    if missing:
        print(f"  Missing: {len(missing)}")
        for m in missing[:5]:
            print(f"    {m}")

    return True


def check_waterloo15():
    """Brief check of Waterloo15."""
    print("\n" + "=" * 60)
    print("Waterloo15")
    print("=" * 60)

    csv_path = os.path.join(DATA_DIR, "Waterloo15", "mos_with_names.csv")
    label_path = os.path.join(DATA_DIR, "Waterloo15", "WINDALLLabel.txt")
    img_base = os.path.join(DATA_DIR, "Waterloo15", "WIND_all")

    csv_data = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 3:
                csv_data[row[0].strip()] = float(row[1].strip())

    print(f"  CSV entries: {len(csv_data)}")
    print(f"  MOS range: [{min(csv_data.values()):.4f}, {max(csv_data.values()):.4f}]")

    label_data = {}
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('#')
            if len(parts) >= 2:
                label_data[parts[0]] = float(parts[1])

    print(f"  Label entries: {len(label_data)}")

    # Compare
    common = set(csv_data.keys()) & set(label_data.keys())
    only_csv = set(csv_data.keys()) - set(label_data.keys())
    only_label = set(label_data.keys()) - set(csv_data.keys())
    print(f"  Common: {len(common)}, Only CSV: {len(only_csv)}, Only Label: {len(only_label)}")

    mos_diffs = []
    for n in common:
        if abs(csv_data[n] - label_data[n]) > 0.01:
            mos_diffs.append((n, csv_data[n], label_data[n]))
    print(f"  MOS mismatch: {len(mos_diffs)}")

    # Images
    missing = []
    ok = 0
    for name in label_data.keys():
        # Images might be in WIND_all
        found = False
        for ext in ['.bmp', '.png']:
            full = os.path.join(img_base, name)
            if not name.endswith(ext):
                full = os.path.join(img_base, name.rsplit('.', 1)[0] + ext)
            if os.path.exists(full):
                ok += 1
                found = True
                break
        if not found:
            missing.append(name)

    print(f"  Images: {ok}/{len(label_data)} found, {len(missing)} missing")
    if missing:
        for m in missing[:5]:
            print(f"    Missing: {m}")

    return True


if __name__ == "__main__":
    check_cviu17()
    check_qads()
    check_sisar()
    check_realsrq()
    check_waterloo15()
    print("\n" + "=" * 60)
    print("All datasets checked.")
