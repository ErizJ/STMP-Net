"""
单一消融实验运行脚本 - 每次只跑一个实验，实时显示训练进度

用法示例:
    # 运行单个实验
    python tools/run_single_ablation.py --dataset qads --experiment full_model

    # 只评估（跳过训练，指定已有 checkpoint）
    python tools/run_single_ablation.py --dataset qads --experiment full_model --eval_only --resume path/to/ckpt.pth

    # 指定输出目录（续接已有实验目录）
    python tools/run_single_ablation.py --dataset qads --experiment wo_scene_prompt --exp_dir results/ablation/exp_20260320_114620

    # 列出所有可用实验
    python tools/run_single_ablation.py --list

实验列表:
    full_model            完整模型
    wo_scene_prompt       移除场景提示
    wo_texture_prompt     移除纹理提示
    wo_structure_prompt   移除结构提示
    wo_distortion_prompt  移除失真提示
    single_scale_window   单尺度窗口
    wo_fidelity_loss      移除保真度损失
    wo_structure_loss     移除结构损失
"""

import argparse
import glob
import logging
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────────────────────
# 消融配置定义
# ──────────────────────────────────────────────────────────────────────────────

from ablation_configs import ABLATION_CONFIGS, DISPLAY_NAMES, DATASET_DEFAULTS


# ──────────────────────────────────────────────────────────────────────────────
# 进度解析
# ──────────────────────────────────────────────────────────────────────────────

# 匹配 train.py 输出的 epoch 进度行，例如:
#   Train: [3/50] ...  或  Epoch [3/50]
EPOCH_RE = re.compile(r'[Ee]poch[:\s\[]*(\d+)[/\s]*(\d+)', re.IGNORECASE)
# 匹配 iter 进度，例如:  [100/1234]
ITER_RE  = re.compile(r'\[(\d+)/(\d+)\]')


def _progress_bar(current, total, width=30):
    """返回 ASCII 进度条字符串"""
    filled = int(width * current / total) if total > 0 else 0
    bar = '#' * filled + '-' * (width - filled)
    pct = 100 * current / total if total > 0 else 0
    return f"[{bar}] {current}/{total} ({pct:.1f}%)"


class ProgressPrinter:
    """
    在子进程输出流中解析 epoch/iter 进度，实时打印到终端。
    同时把所有输出写入日志文件。
    """

    def __init__(self, logger, total_epochs, phase="训练"):
        self.logger = logger
        self.total_epochs = total_epochs
        self.phase = phase
        self.current_epoch = 0
        self.last_progress_line = ""
        self._lock = threading.Lock()

    def feed(self, line: str):
        """处理子进程的一行输出"""
        line = line.rstrip('\n').rstrip('\r')
        if not line.strip():
            return

        # 写入日志（不打印到终端，避免刷屏）
        self.logger.debug(line)

        # 尝试解析 epoch 进度
        m_epoch = EPOCH_RE.search(line)
        if m_epoch:
            cur = int(m_epoch.group(1))
            tot = int(m_epoch.group(2))
            if tot > 1:  # 过滤掉 iter 级别的 [x/y]
                with self._lock:
                    self.current_epoch = cur
                    self.total_epochs = tot
                progress = _progress_bar(cur, tot)
                msg = f"\r  [{self.phase}] Epoch {progress}"
                sys.stdout.write(msg)
                sys.stdout.flush()
                self.last_progress_line = msg
                return

        # 打印关键信息行（loss、PLCC、SRCC、错误等）
        lower = line.lower()
        is_key = any(kw in lower for kw in [
            'loss', 'plcc', 'srcc', 'error', 'traceback', 'exception',
            'warning', 'checkpoint', 'best', 'epoch', 'ablation_result',
            '错误', '警告', '失败', '成功',
        ])
        if is_key:
            # 换行后打印关键信息，保持进度条在最后
            safe_line = line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
                sys.stdout.encoding or "utf-8", errors="replace"
            )
            sys.stdout.write(f"\n  {safe_line}\n")
            sys.stdout.flush()
            # 重新打印进度条
            if self.last_progress_line:
                sys.stdout.write(self.last_progress_line)
                sys.stdout.flush()

    def finish(self):
        """结束时换行"""
        sys.stdout.write("\n")
        sys.stdout.flush()


def stream_subprocess(cmd, logger, total_epochs=50, phase="训练", env=None):
    """
    运行子进程，实时流式读取 stdout+stderr，解析并显示进度。
    返回 (returncode, all_output_str)
    """
    printer = ProgressPrinter(logger, total_epochs, phase)
    output_lines = []

    logger.info(f"执行命令: {' '.join(cmd)}")
    print(f"\n  命令: {' '.join(cmd)}\n")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # 合并 stderr 到 stdout
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,                  # 行缓冲
            env=env,
        )
    except FileNotFoundError as e:
        logger.error(f"无法启动进程: {e}")
        return 1, ""

    for line in proc.stdout:
        output_lines.append(line)
        printer.feed(line)

    proc.wait()
    printer.finish()

    all_output = "".join(output_lines)
    return proc.returncode, all_output


# ──────────────────────────────────────────────────────────────────────────────
# 核心函数
# ──────────────────────────────────────────────────────────────────────────────

def setup_logger(log_dir, exp_name):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{exp_name}.log')
    logger = logging.getLogger(f'ablation.{exp_name}')
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def _safe_text(text: str) -> str:
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


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
    # 消融实验统一 8 epoch
    if 'STAGE1' in config:
        config['STAGE1']['EPOCHS'] = 8
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return config


def find_best_checkpoint(root_dir):
    pattern = os.path.join(root_dir, '**', 'ckpt_epoch_*.pth')
    ckpts = glob.glob(pattern, recursive=True)
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getmtime)


def get_total_epochs(config_path):
    """从 config 文件读取训练 epoch 数"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cfg.get('STAGE1', {}).get('EPOCHS', 50)
    except Exception:
        return 50


def run_training(config_path, output_dir, logger):
    """训练，实时显示进度，返回 checkpoint 路径"""
    total_epochs = get_total_epochs(config_path)
    cmd = [
        sys.executable, os.path.join(_PROJECT_ROOT, 'train.py'),
        '--cfg', config_path,
        '--output', output_dir,
        '--tag', 'ablation',
    ]

    print(f"\n{'─'*60}")
    print(f"  [训练] 共 {total_epochs} 个 epoch，输出目录: {output_dir}")
    print(f"{'─'*60}")

    t0 = time.time()
    returncode, output = stream_subprocess(cmd, logger, total_epochs=total_epochs, phase="训练")
    elapsed = time.time() - t0

    if returncode != 0:
        logger.error(f"训练失败 (returncode={returncode})，耗时 {elapsed:.1f}s")
        logger.error("=== 训练输出（最后100行）===")
        for line in output.splitlines()[-100:]:
            logger.error(_safe_text(line))
        return None

    logger.info(f"训练完成，耗时 {elapsed:.1f}s")
    print(f"  训练完成，耗时 {elapsed/60:.1f} 分钟")

    ckpt = find_best_checkpoint(output_dir)
    if ckpt:
        logger.info(f"找到 checkpoint: {ckpt}")
        print(f"  Checkpoint: {ckpt}")
    else:
        logger.error(f"未找到 checkpoint，目录: {output_dir}")
        print(f"  ✗ 未找到 checkpoint")
    return ckpt


def run_evaluation(config_path, checkpoint_path, output_dir, logger):
    """评估，实时显示输出，返回 (plcc, srcc)"""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(_PROJECT_ROOT, 'tools', 'ablation_eval.py'),
        '--cfg', config_path,
        '--resume', checkpoint_path,
        '--output', output_dir,
    ]

    print(f"\n{'─'*60}")
    print(f"  [评估] checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"{'─'*60}")

    t0 = time.time()
    returncode, output = stream_subprocess(cmd, logger, total_epochs=1, phase="评估")
    elapsed = time.time() - t0

    if returncode != 0:
        logger.error(f"评估失败 (returncode={returncode})，耗时 {elapsed:.1f}s")
        logger.error("=== 评估输出 ===")
        for line in output.splitlines()[-50:]:
            logger.error(line)

    # 解析结果
    match = re.search(
        r'ABLATION_RESULT\s+PLCC=([\d.]+)\s+SRCC=([\d.]+)',
        output, re.IGNORECASE
    )
    if match:
        plcc = float(match.group(1))
        srcc = float(match.group(2))
        logger.info(f"评估结果: PLCC={plcc:.4f}, SRCC={srcc:.4f}")
        return plcc, srcc
    else:
        logger.error("未能解析 ABLATION_RESULT，检查评估输出")
        # 打印完整输出帮助调试
        print("\n  === 评估完整输出（调试用）===")
        for line in output.splitlines()[-30:]:
            print(f"  {line}")
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────

def print_banner(dataset, exp_name, exp_idx, total_exps, exp_dir):
    display = DISPLAY_NAMES.get(exp_name, exp_name)
    overrides = ABLATION_CONFIGS.get(exp_name, {})
    print(f"\n{'═'*60}")
    print(f"  消融实验  [{exp_idx}/{total_exps}]")
    print(f"  数据集  : {dataset.upper()}")
    print(f"  实验    : {display}")
    print(f"  配置覆盖: {overrides if overrides else '(无，完整模型)'}")
    print(f"  输出目录: {exp_dir}")
    print(f"{'═'*60}")


def main():
    parser = argparse.ArgumentParser(
        description='单一消融实验运行器（实时进度显示）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--dataset', choices=list(DATASET_DEFAULTS.keys()),
                        help='数据集名称')
    parser.add_argument('--experiment', choices=list(ABLATION_CONFIGS.keys()),
                        help='实验名称')
    parser.add_argument('--output_dir', default='results/ablation',
                        help='结果根目录 (default: results/ablation)')
    parser.add_argument('--exp_dir', default=None,
                        help='指定已有实验目录（续接），不指定则自动创建新目录')
    parser.add_argument('--eval_only', action='store_true',
                        help='跳过训练，只做评估')
    parser.add_argument('--resume', default=None,
                        help='eval_only 时指定 checkpoint 路径')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用实验后退出')
    args = parser.parse_args()

    # --list
    if args.list:
        print("\n可用数据集:")
        for ds, cfg in DATASET_DEFAULTS.items():
            exists = "✓" if os.path.exists(cfg) else "✗ (config 不存在)"
            print(f"  {ds:<12} {cfg}  {exists}")
        print("\n可用实验:")
        for i, (name, overrides) in enumerate(ABLATION_CONFIGS.items(), 1):
            display = DISPLAY_NAMES[name]
            ov = str(overrides) if overrides else "(无覆盖，完整模型)"
            print(f"  {i}. {name:<24} {display:<28} {ov}")
        print()
        sys.exit(0)

    # 参数校验
    if not args.dataset:
        parser.error("请指定 --dataset")
    if not args.experiment:
        parser.error("请指定 --experiment")

    base_config = DATASET_DEFAULTS[args.dataset]
    if not os.path.exists(base_config):
        print(f"错误: 配置文件不存在: {base_config}")
        sys.exit(1)

    # 确定实验目录
    if args.exp_dir:
        exp_dir = args.exp_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = os.path.join(args.output_dir, f'exp_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)

    exp_name = args.experiment
    all_exps = list(ABLATION_CONFIGS.keys())
    exp_idx = all_exps.index(exp_name) + 1

    # 日志
    log_dir = os.path.join(exp_dir, args.dataset, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir, exp_name)

    print_banner(args.dataset, exp_name, exp_idx, len(all_exps), exp_dir)
    logger.info(f"实验开始: dataset={args.dataset}, experiment={exp_name}")
    logger.info(f"实验目录: {exp_dir}")

    t_start = time.time()

    # 生成消融 config
    cfg_out = os.path.join(exp_dir, f'{args.dataset}_{exp_name}.yaml')
    overrides = ABLATION_CONFIGS[exp_name]
    create_ablation_config(base_config, overrides, cfg_out)
    logger.info(f"消融 config 已生成: {cfg_out}")

    # ── 训练 ──
    if args.eval_only:
        ckpt = args.resume
        if not ckpt:
            # 尝试自动找已有 checkpoint
            train_out = os.path.join(exp_dir, args.dataset, exp_name, 'train')
            ckpt = find_best_checkpoint(train_out)
        if not ckpt or not os.path.exists(ckpt):
            print(f"错误: eval_only 模式下未找到 checkpoint，请用 --resume 指定")
            sys.exit(1)
        print(f"  [跳过训练] 使用 checkpoint: {ckpt}")
        logger.info(f"eval_only 模式，checkpoint: {ckpt}")
    else:
        train_out = os.path.join(exp_dir, args.dataset, exp_name, 'train')
        ckpt = run_training(cfg_out, train_out, logger)
        if ckpt is None:
            print(f"\n  [FAILED] 训练失败，实验终止")
            print(f"  日志文件: {os.path.join(log_dir, exp_name + '.log')}")
            sys.exit(1)

    # ── 评估 ──
    eval_out = os.path.join(exp_dir, args.dataset, exp_name, 'eval')
    plcc, srcc = run_evaluation(cfg_out, ckpt, eval_out, logger)

    # ── 结果汇总 ──
    elapsed = time.time() - t_start
    display = DISPLAY_NAMES.get(exp_name, exp_name)

    print(f"\n{'═'*60}")
    print(f"  实验完成: {display}")
    print(f"  数据集  : {args.dataset.upper()}")
    if plcc is not None:
        print(f"  PLCC    : {plcc:.4f}")
        print(f"  SRCC    : {srcc:.4f}")
    else:
        print(f"  结果    : 解析失败，请查看日志")
    print(f"  总耗时  : {elapsed/60:.1f} 分钟")
    print(f"  日志    : {os.path.join(log_dir, exp_name + '.log')}")
    print(f"{'═'*60}\n")

    logger.info(f"实验完成: PLCC={plcc}, SRCC={srcc}, 耗时={elapsed:.1f}s")

    # 写入结果文件，方便后续汇总
    result_file = os.path.join(log_dir, 'result.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"dataset={args.dataset}\n")
        f.write(f"experiment={exp_name}\n")
        f.write(f"display={display}\n")
        f.write(f"PLCC={plcc}\n")
        f.write(f"SRCC={srcc}\n")
        f.write(f"checkpoint={ckpt}\n")
        f.write(f"elapsed_seconds={elapsed:.1f}\n")
        f.write(f"timestamp={datetime.now().isoformat()}\n")

    sys.exit(0 if plcc is not None else 1)


if __name__ == '__main__':
    main()
