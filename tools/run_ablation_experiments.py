# -*- coding: utf-8 -*-
"""
Ablation experiment script - Table 3: Training + Evaluation mode
"""

import argparse
import csv
import glob
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta

import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------------------
# Ablation config definitions
# ------------------------------------------------------------------------------

from ablation_configs import ABLATION_CONFIGS, DISPLAY_NAMES, DATASET_DEFAULTS, ABLATION_ORDER


# ------------------------------------------------------------------------------
# Progress tracker
# ------------------------------------------------------------------------------

class ProgressTracker:
    def __init__(self, total_tasks, logger):
        self.total = total_tasks
        self.done = 0
        self.failed = 0
        self.start_time = time.time()
        self.logger = logger
        self.completed_results = []

    def update(self, dataset, exp_name, plcc, srcc, success=True):
        self.done += 1
        if not success:
            self.failed += 1
        elapsed = time.time() - self.start_time
        avg_per_task = elapsed / self.done
        remaining = (self.total - self.done) * avg_per_task
        eta = str(timedelta(seconds=int(remaining)))
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        status = "OK" if success else "FAIL"
        result_str = f"PLCC={plcc:.4f} SRCC={srcc:.4f}" if (plcc is not None and srcc is not None) else "FAILED"
        self.completed_results.append((dataset, exp_name, plcc, srcc))

        bar_filled = int(20 * self.done / self.total)
        bar = "#" * bar_filled + "." * (20 - bar_filled)

        self.logger.info("")
        self.logger.info("=" * 65)
        self.logger.info(f"  Progress [{bar}] {self.done}/{self.total}")
        self.logger.info(f"  {status} [{dataset}] {DISPLAY_NAMES.get(exp_name, exp_name)}: {result_str}")
        self.logger.info(f"  Elapsed: {elapsed_str}  |  ETA: {eta}")
        self.logger.info(f"  Success: {self.done - self.failed}  Failed: {self.failed}")
        self.logger.info("=" * 65)
        self.logger.info("")

    def print_interim_table(self, all_dataset_results, datasets_done):
        if not datasets_done:
            return
        self.logger.info("")
        self.logger.info("-" * 65)
        self.logger.info("  Current results:")
        self.logger.info("-" * 65)
        _print_table(all_dataset_results, datasets_done, self.logger)
        self.logger.info("-" * 65)
        self.logger.info("")


# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'ablation_experiments.log')
    logger = logging.getLogger('ablation_main')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def create_ablation_config(base_config_path, overrides, output_path):
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config.update(overrides)
    # Windows 上 NUM_WORKERS>0 容易导致多进程死锁，强制设为 0
    if os.name == 'nt' and 'DATA' in config:
        config['DATA']['NUM_WORKERS'] = 0
    # Windows 上 cuDNN half-precision Conv 容易触发 CUDNN_STATUS_INTERNAL_ERROR，
    # 禁用 AMP 改用 float32 训练（速度略慢但稳定）
    if os.name == 'nt':
        config['AMP_ENABLE'] = False
    # 消融实验统一 15 epoch
    if 'STAGE1' in config:
        config['STAGE1']['EPOCHS'] = 8
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def find_best_checkpoint_recursive(root_dir):
    pattern = os.path.join(root_dir, '**', 'ckpt_epoch_*.pth')
    ckpts = glob.glob(pattern, recursive=True)
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getmtime)


def run_training(config_path, output_dir, logger, exp_label=""):
    cmd = [
        sys.executable, os.path.join(_PROJECT_ROOT, 'train.py'),
        '--cfg', config_path,
        '--output', output_dir,
        '--tag', 'ablation',
        '--rnum', '1',
    ]
    logger.info(f"Training cmd: {' '.join(cmd)}")
    logger.info(f"Output dir: {output_dir}")

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
        env=env,
    )

    output_lines = []
    for line in process.stdout:
        line = line.rstrip()
        output_lines.append(line)
        if any(kw in line for kw in ['Epoch', 'stage 1', 'Loss:', 'PLCC', 'SRCC', 'best', 'Error', 'error', 'Traceback', 'Exception']):
            logger.info(f"  [train] {line}")

    process.wait()

    if process.returncode != 0:
        logger.error(f"Training process returned: {process.returncode}")
        logger.error("Last 20 lines:")
        for l in output_lines[-20:]:
            logger.error(f"  {l}")
        return None

    ckpt = find_best_checkpoint_recursive(output_dir)
    if ckpt:
        logger.info(f"Best checkpoint: {ckpt}")
    else:
        logger.error(f"Training done but no checkpoint found in: {output_dir}")
    return ckpt


def run_evaluation(config_path, checkpoint_path, output_dir, logger):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(_PROJECT_ROOT, 'tools', 'ablation_eval.py'),
        '--cfg', config_path,
        '--resume', checkpoint_path,
        '--output', output_dir,
    ]
    logger.info(f"Eval cmd: {' '.join(cmd)}")

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding='utf-8', errors='replace',
        env=env,
    )
    combined = result.stdout + result.stderr
    if result.stdout.strip():
        logger.info(f"Eval output:\n{result.stdout.strip()}")
    if result.returncode != 0:
        logger.error(f"Eval process returned: {result.returncode}")
        if result.stderr.strip():
            logger.error(f"Eval stderr:\n{result.stderr.strip()[-1000:]}")

    match = re.search(
        r'ABLATION_RESULT\s+PLCC=([\d.nan]+)\s+SRCC=([\d.nan]+)',
        combined, re.IGNORECASE
    )
    if match:
        plcc = float(match.group(1))
        srcc = float(match.group(2))
        logger.info(f"  -> PLCC={plcc:.4f}, SRCC={srcc:.4f}")
        return plcc, srcc
    else:
        logger.error("Could not parse ABLATION_RESULT. Eval output:")
        logger.error(combined[-500:])
        return None, None


def run_dataset_ablation(dataset_name, config_path, exp_dir, experiments, logger, progress):
    logger.info(f"\n{'#'*65}")
    logger.info(f"  Dataset: {dataset_name.upper()}")
    logger.info(f"  Base Config: {config_path}")
    logger.info(f"{'#'*65}")

    results = {}
    for idx, exp_name in enumerate(experiments):
        if exp_name not in ABLATION_CONFIGS:
            logger.warning(f"Unknown experiment: {exp_name}, skipping")
            continue

        display = DISPLAY_NAMES.get(exp_name, exp_name)
        overrides = ABLATION_CONFIGS[exp_name]

        logger.info(f"\n{'='*65}")
        logger.info(f"  Start: [{dataset_name}] {display}")
        logger.info(f"  Overrides: {overrides if overrides else '(none, full model)'}")
        logger.info(f"{'='*65}")

        cfg_out = os.path.join(exp_dir, f'{dataset_name}_{exp_name}.yaml')
        create_ablation_config(config_path, overrides, cfg_out)

        train_out = os.path.join(exp_dir, dataset_name, exp_name)
        t0 = time.time()
        ckpt = run_training(cfg_out, train_out, logger, exp_label=f"[{dataset_name}/{exp_name}]")
        train_time = str(timedelta(seconds=int(time.time() - t0)))

        if ckpt is None:
            logger.error(f"  [{dataset_name}] {display} training failed, skipping eval")
            results[exp_name] = (None, None)
            progress.update(dataset_name, exp_name, None, None, success=False)
            continue

        logger.info(f"  Training time: {train_time}")

        eval_out = os.path.join(exp_dir, dataset_name, exp_name, 'eval')
        plcc, srcc = run_evaluation(cfg_out, ckpt, eval_out, logger)
        results[exp_name] = (plcc, srcc)
        progress.update(dataset_name, exp_name, plcc, srcc, success=(plcc is not None))

    return results


def _compute_averages(all_dataset_results, datasets, experiments):
    averages = {}
    for exp_name in experiments:
        plcc_vals = []
        srcc_vals = []
        for ds in datasets:
            plcc, srcc = all_dataset_results.get(ds, {}).get(exp_name, (None, None))
            if plcc is not None:
                plcc_vals.append(plcc)
            if srcc is not None:
                srcc_vals.append(srcc)
        avg_plcc = sum(plcc_vals) / len(plcc_vals) if plcc_vals else None
        avg_srcc = sum(srcc_vals) / len(srcc_vals) if srcc_vals else None
        averages[exp_name] = (avg_plcc, avg_srcc)
    return averages


def save_summary(exp_dir, all_dataset_results, datasets, timestamp):
    summary_path = os.path.join(exp_dir, 'ablation_summary.txt')
    csv_path = os.path.join(exp_dir, 'results_summary.csv')
    experiments = list(ABLATION_CONFIGS.keys())
    averages = _compute_averages(all_dataset_results, datasets, experiments)

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("Ablation Study Summary\n")
        f.write(f"Time: {timestamp}\n")
        f.write("=" * 90 + "\n\n")

        all_cols = list(datasets) + ['Average']
        header = f"{'Variant':<26}"
        for col in all_cols:
            header += f"  {col.upper():^17}"
        f.write(header + "\n")

        sub = " " * 26
        for _ in all_cols:
            sub += f"  {'PLCC':>8} {'SRCC':>8}"
        f.write(sub + "\n")
        f.write("-" * (26 + len(all_cols) * 19) + "\n")

        for exp_name in experiments:
            display = DISPLAY_NAMES.get(exp_name, exp_name)
            row = f"{display:<26}"
            for ds in datasets:
                plcc, srcc = all_dataset_results.get(ds, {}).get(exp_name, (None, None))
                ps = f"{plcc:.4f}" if plcc is not None else "N/A"
                ss = f"{srcc:.4f}" if srcc is not None else "N/A"
                row += f"  {ps:>8} {ss:>8}"
            avg_plcc, avg_srcc = averages[exp_name]
            aps = f"{avg_plcc:.4f}" if avg_plcc is not None else "N/A"
            ass_ = f"{avg_srcc:.4f}" if avg_srcc is not None else "N/A"
            row += f"  {aps:>8} {ass_:>8}"
            f.write(row + "\n")

        f.write("\n" + "=" * 90 + "\n")

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        header = ['Variant']
        for ds in datasets:
            header += [f'{ds}_PLCC', f'{ds}_SRCC']
        header += ['Average_PLCC', 'Average_SRCC']
        writer.writerow(header)

        for exp_name in experiments:
            row = [DISPLAY_NAMES.get(exp_name, exp_name)]
            for ds in datasets:
                plcc, srcc = all_dataset_results.get(ds, {}).get(exp_name, (None, None))
                row += [
                    f"{plcc:.4f}" if plcc is not None else "N/A",
                    f"{srcc:.4f}" if srcc is not None else "N/A",
                ]
            avg_plcc, avg_srcc = averages[exp_name]
            row += [
                f"{avg_plcc:.4f}" if avg_plcc is not None else "N/A",
                f"{avg_srcc:.4f}" if avg_srcc is not None else "N/A",
            ]
            writer.writerow(row)

    return summary_path, csv_path


def _print_table(all_dataset_results, datasets, logger):
    experiments = list(ABLATION_CONFIGS.keys())
    averages = _compute_averages(all_dataset_results, datasets, experiments)
    all_cols = list(datasets) + ['Average']

    header = f"{'Variant':<26}"
    for col in all_cols:
        header += f"  {col.upper():^17}"
    logger.info(header)

    sub = " " * 26
    for _ in all_cols:
        sub += f"  {'PLCC':>8} {'SRCC':>8}"
    logger.info(sub)
    logger.info("-" * (26 + len(all_cols) * 19))

    for exp_name in experiments:
        display = DISPLAY_NAMES.get(exp_name, exp_name)
        row = f"{display:<26}"
        for ds in datasets:
            plcc, srcc = all_dataset_results.get(ds, {}).get(exp_name, (None, None))
            ps = f"{plcc:.4f}" if plcc is not None else "  N/A  "
            ss = f"{srcc:.4f}" if srcc is not None else "  N/A  "
            row += f"  {ps:>8} {ss:>8}"
        avg_plcc, avg_srcc = averages[exp_name]
        aps = f"{avg_plcc:.4f}" if avg_plcc is not None else "  N/A  "
        ass = f"{avg_srcc:.4f}" if avg_srcc is not None else "  N/A  "
        row += f"  {aps:>8} {ass:>8}"
        logger.info(row)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Ablation experiments - train + eval')
    parser.add_argument('--datasets', nargs='+',
                        choices=list(DATASET_DEFAULTS.keys()),
                        default=['waterloo15', 'cviu17', 'qads'],
                        help='Datasets to evaluate')
    parser.add_argument('--output_dir', type=str, default='results/ablation',
                        help='Output directory')
    parser.add_argument('--experiments', nargs='+',
                        default=ABLATION_ORDER,
                        help='Experiments to run')
    parser.add_argument('--mode', choices=['standard', 'beta_sweep'], default='standard',
                        help='standard: ablation experiments; beta_sweep: beta value sweep')
    parser.add_argument('--betas', nargs='+', type=float,
                        default=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0],
                        help='Beta values for beta_sweep mode')
    args = parser.parse_args()

    # Beta sweep mode: add beta_* experiments dynamically
    if args.mode == 'beta_sweep':
        for beta_val in args.betas:
            key = f'beta_{beta_val}'
            ABLATION_CONFIGS[key] = {'BETA': beta_val}
            DISPLAY_NAMES[key] = f'Beta={beta_val}'
        args.experiments = [k for k in ABLATION_CONFIGS.keys() if k.startswith('beta_')]

    for ds in args.datasets:
        cfg = DATASET_DEFAULTS[ds]
        if not os.path.exists(cfg):
            print(f"Error: config not found: {cfg}")
            sys.exit(1)

    for f, subdir in [('train.py', ''), ('ablation_eval.py', 'tools')]:
        fpath = os.path.join(_PROJECT_ROOT, subdir, f) if subdir else os.path.join(_PROJECT_ROOT, f)
        if not os.path.exists(fpath):
            print(f"Error: {fpath} not found")
            sys.exit(1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(args.output_dir, f'exp_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)

    logger = setup_logger(exp_dir)

    total_tasks = len(args.datasets) * len(args.experiments)
    progress = ProgressTracker(total_tasks, logger)

    logger.info("=" * 65)
    logger.info("  Ablation experiments start - train + eval mode")
    logger.info(f"  Datasets: {args.datasets}")
    logger.info(f"  Experiments: {args.experiments}")
    logger.info(f"  Total tasks: {total_tasks}")
    logger.info(f"  Output dir: {exp_dir}")
    logger.info("=" * 65)

    all_dataset_results = {}
    datasets_done = []

    for ds_name in args.datasets:
        cfg = DATASET_DEFAULTS[ds_name]
        results = run_dataset_ablation(
            ds_name, cfg, exp_dir, args.experiments, logger, progress
        )
        all_dataset_results[ds_name] = results
        datasets_done.append(ds_name)

        progress.print_interim_table(all_dataset_results, datasets_done)
        save_summary(exp_dir, all_dataset_results, datasets_done, timestamp)
        logger.info(f"  Intermediate results saved to {exp_dir}/ablation_summary.txt")

    logger.info("\n" + "=" * 65)
    logger.info("  All ablation experiments done. Final results:")
    logger.info("=" * 65)
    _print_table(all_dataset_results, datasets_done, logger)

    summary_path, csv_path = save_summary(
        exp_dir, all_dataset_results, datasets_done, timestamp
    )
    total_time = str(timedelta(seconds=int(time.time() - progress.start_time)))
    logger.info(f"\n  Total time: {total_time}")
    logger.info(f"  Summary:    {summary_path}")
    logger.info(f"  CSV:        {csv_path}")
    logger.info("=" * 65)


if __name__ == '__main__':
    main()
