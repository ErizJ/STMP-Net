"""
跨数据集泛化评估脚本
Cross-Dataset Generalization Evaluation

用法：
    python cross_dataset_eval.py \
        --train_dataset cviu17 \
        --test_dataset qads \
        --checkpoint log/cviu17/default/1/ckpt_epoch_20.pth \
        --output_dir results/cross_dataset \
        [--use_tta]
"""

import argparse
import logging
import os
import random
import numpy as np
import torch
import pickle
from scipy import stats

from config import get_config
from models import STMPNet
from eval import get_dataloader


# 各数据集对应的 config 文件
DATASET_CONFIG_MAP = {
    "cviu17":     "configs/Pure/vit_small_pre_coder_cviu17.yaml",
    "qads":       "configs/Pure/vit_small_pre_coder_qads.yaml",
    "waterloo15": "configs/Pure/vit_small_pre_coder_waterloo15.yaml",
    "live":       "configs/Pure/vit_small_pre_coder_live.yaml",
    "csiq":       "configs/Pure/vit_small_pre_coder_csiq.yaml",
    "tid2013":    "configs/Pure/vit_small_pre_coder_tid2013.yaml",
    "kadid":      "configs/Pure/vit_small_pre_coder_kadid.yaml",
    "livec":      "configs/Pure/vit_small_pre_coder_livec.yaml",
    "koniq":      "configs/Pure/vit_small_pre_coder_koniq.yaml",
}


def setup_logger(output_dir, name="cross_eval"):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(name)


def load_model(config, checkpoint_path, logger):
    """加载模型并恢复检查点"""
    model = STMPNet(config)
    model.cuda()

    logger.info(f"加载检查点: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # 兼容不同的检查点格式
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"缺失参数: {missing}")
    if unexpected:
        logger.warning(f"多余参数: {unexpected}")

    model.eval()
    model.float()
    return model


def evaluate(model, val_loader, val_len, patch_num, use_tta, logger, dataset_name):
    """在给定 dataloader 上评估模型，返回 (plcc, srcc)"""
    temp_pred, temp_gt = [], []

    with torch.no_grad():
        for batch_data in val_loader:
            img = batch_data[0].cuda(non_blocking=True).float()
            labels = batch_data[1].cuda(non_blocking=True)

            out = model(img, eval=True)
            preds = out[0] if isinstance(out, (tuple, list)) else out

            if use_tta:
                img_flip = torch.flip(img, dims=[3])
                out_flip = model(img_flip, eval=True)
                preds_flip = out_flip[0] if isinstance(out_flip, (tuple, list)) else out_flip
                preds = (preds + preds_flip) / 2.0

            temp_pred.append(preds.reshape(-1))
            temp_gt.append(labels.reshape(-1))

    pred_scores = torch.cat(temp_pred)[:val_len]
    gt_scores = torch.cat(temp_gt)[:val_len]

    final_preds = pred_scores.view(-1, patch_num).mean(dim=-1).squeeze().cpu().numpy()
    final_gt = gt_scores.view(-1, patch_num).mean(dim=-1).squeeze().cpu().numpy()

    valid = np.isfinite(final_preds) & np.isfinite(final_gt)
    if not np.all(valid):
        logger.info(f"[{dataset_name}] 过滤 {(~valid).sum()} 个异常样本")
        final_preds, final_gt = final_preds[valid], final_gt[valid]

    if len(final_preds) < 2:
        return 0.0, 0.0

    plcc, _ = stats.pearsonr(final_preds, final_gt)
    srcc, _ = stats.spearmanr(final_preds, final_gt)
    return float(plcc), float(srcc)


def main():
    parser = argparse.ArgumentParser(description="跨数据集泛化评估")
    parser.add_argument("--train_dataset", type=str, default=None,
                        help="训练时使用的数据集（用于选择 config）")
    parser.add_argument("--test_dataset", type=str, default=None,
                        help="测试目标数据集")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型检查点路径")
    parser.add_argument("--output_dir", type=str, default="results/cross_dataset",
                        help="结果输出目录")
    parser.add_argument("--use_tta", action="store_true", default=False,
                        help="是否使用 Test Time Augmentation")
    parser.add_argument("--opts", nargs="+", default=None,
                        help="额外的 config 覆盖项，格式: KEY VALUE ...")
    parser.add_argument("--all", action="store_true",
                        help="运行所有标准跨数据集实验对")
    parser.add_argument("--cviu17_checkpoint", type=str,
                        default="log/cviu17/default/1/ckpt_epoch_20.pth",
                        help="CVIU17 checkpoint (for --all mode)")
    parser.add_argument("--qads_checkpoint", type=str,
                        default="log/qads/default/1/ckpt_epoch_90.pth",
                        help="QADS checkpoint (for --all mode)")
    args = parser.parse_args()

    if args.all:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join(args.output_dir, f'exp_{timestamp}')
        experiments = [
            ('cviu17', 'waterloo15', args.cviu17_checkpoint),
            ('cviu17', 'qads', args.cviu17_checkpoint),
            ('qads', 'waterloo15', args.qads_checkpoint),
            ('qads', 'cviu17', args.qads_checkpoint),
        ]
        print("=" * 80)
        print("跨数据集泛化实验 - 批量运行")
        print(f"CVIU17 checkpoint: {args.cviu17_checkpoint}")
        print(f"QADS checkpoint: {args.qads_checkpoint}")
        print(f"Output dir: {args.output_dir}")
        print("=" * 80)
        for train_ds, test_ds, ckpt in experiments:
            print(f"\nRunning: {train_ds} -> {test_ds}")
            single_args = argparse.Namespace(
                train_dataset=train_ds, test_dataset=test_ds,
                checkpoint=ckpt, output_dir=args.output_dir,
                use_tta=args.use_tta, opts=args.opts,
            )
            _run_single(single_args)
        print("\n" + "=" * 80)
        print("All experiments done.")
        summary_file = os.path.join(args.output_dir, "results_summary.txt")
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                print(f.read())
        print("=" * 80)
        return

    if not args.train_dataset or not args.test_dataset or not args.checkpoint:
        parser.error("--train_dataset, --test_dataset, --checkpoint required (or use --all)")
    _run_single(args)


def _run_single(args):
    """Run a single cross-dataset evaluation."""
    import types
    exp_name = f"{args.train_dataset}_to_{args.test_dataset}"
    logger = setup_logger(args.output_dir, name=exp_name)

    logger.info("=" * 60)
    logger.info(f"跨数据集评估: {args.train_dataset.upper()} -> {args.test_dataset.upper()}")
    logger.info(f"检查点: {args.checkpoint}")
    logger.info(f"TTA: {args.use_tta}")
    logger.info("=" * 60)

    if args.train_dataset not in DATASET_CONFIG_MAP:
        raise ValueError(f"未知训练数据集: {args.train_dataset}，请在 DATASET_CONFIG_MAP 中添加")
    config_file = DATASET_CONFIG_MAP[args.train_dataset]
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config 文件不存在: {config_file}")

    cfg_args = types.SimpleNamespace(
        cfg=config_file,
        opts=args.opts,
        batch_size=None, data_path=None, zip=False,
        cache_mode="part", pretrained=None, resume=None,
        alpha=None, beta=None, accumulation_steps=None,
        tensorboard=False, use_checkpoint=False,
        disable_amp=False, amp_opt_level=None,
        output=args.output_dir, tag="cross_eval",
        eval=True, throughput=False, debug=False,
        repeat=False, rnum=1, seed=42, depth=None,
        epoch=None, token=None, prompt=None,
        scene=False, dist=False, texture=False, structure=False,
        gamma=None, delta=None, visual=False,
        data_percent=0.8, print=False,
    )
    config = get_config(cfg_args, local_rank=0)

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    config.defrost()
    sel_num = list(range(0, config.SET.COUNT))
    config.SET.TRAIN_INDEX = sel_num[:int(round(0.8 * len(sel_num)))]
    config.SET.TEST_INDEX = sel_num[int(round(0.8 * len(sel_num))):]
    config.freeze()

    model = load_model(config, args.checkpoint, logger)

    val_loader, val_len = get_dataloader(config, args.test_dataset, logger, cross_dataset=True)
    plcc, srcc = evaluate(model, val_loader, val_len, config.DATA.PATCH_NUM,
                          args.use_tta, logger, args.test_dataset)

    logger.info(f"\n{'='*60}")
    logger.info(f"结果: {args.train_dataset.upper()} -> {args.test_dataset.upper()}")
    logger.info(f"  PLCC: {plcc:.4f}")
    logger.info(f"  SRCC: {srcc:.4f}")
    logger.info(f"{'='*60}")

    result_file = os.path.join(args.output_dir, f"{exp_name}_result.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Train: {args.train_dataset}\n")
        f.write(f"Test:  {args.test_dataset}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"TTA: {args.use_tta}\n")
        f.write(f"PLCC: {plcc:.4f}\n")
        f.write(f"SRCC: {srcc:.4f}\n")
    logger.info(f"结果已保存到: {result_file}")

    summary_file = os.path.join(args.output_dir, "results_summary.txt")
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(f"{args.train_dataset:12s} -> {args.test_dataset:12s}  PLCC={plcc:.4f}  SRCC={srcc:.4f}\n")


if __name__ == "__main__":
    main()
