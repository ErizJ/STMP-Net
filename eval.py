import logging
import os
import time
import random
import torch.cuda
import numpy as np
import torch
import time
from datetime import timedelta
import torch.nn as nn
from timm.utils import AverageMeter  # accuracy
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F

from loss import (SupConLoss, Fidelity_Loss, Fidelity_Loss_distortion, Multi_Fidelity_Loss,
                  loss_quality, CrossEntropyLabelSmooth)
from scipy import stats
from datasets import build_dataloader, get_labels


def get_dataloader(config, dataset_name, logger, cross_dataset=False):
    """
    获取数据加载器。
    - 同数据集评估：使用 config 中固定的 TEST_INDEX（与训练时一致的 80/20 划分）
    - 跨数据集评估：使用全量数据（cross_dataset=True），避免随机划分导致结果不可复现
    """
    # 数据集根目录，相对于项目根目录的 data 文件夹
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    def _full_index(count):
        """跨数据集评估时使用全量数据"""
        return list(range(0, count))

    def _split_index(count):
        """同数据集评估时随机 80/20 划分（仅在无固定 index 时使用）"""
        sel_num = list(range(0, count))
        random.shuffle(sel_num)
        return sel_num[int(round(0.8 * len(sel_num))): len(sel_num)]

    if dataset_name == config.DATA.DATASET and not cross_dataset:
        # 同数据集：使用训练时固定的 TEST_INDEX，保证可复现
        dataset_path = config.DATA.DATA_PATH
        # 相对路径转绝对路径（基于项目根目录）
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_path)
        batch_size = config.DATA.BATCH_SIZE
        test_index = config.SET.TEST_INDEX
        logger.info(f"{dataset_name} (in-dataset eval): {len(test_index)} samples")
        return build_dataloader(config, dataset_name, dataset_path, batch_size, test_index)

    # 跨数据集评估：使用全量数据
    DATASET_CONFIGS = {
        "live":      ("live/databaserelease2", 29,    12),
        "csiq":      ("CSIQ",                  30,    12),
        "tid2013":   ("tid2013",               25,    48),
        "livec":     ("ChallengeDB_release",   1162,  16),
        "koniq":     ("koniq-10k",             10073, 128),
        "kadid":     ("kadid",                 81,    128),
        "spaq":      ("SPAQ",                  11125, 128),
        "livefb":    ("liveFB",                39810, 128),
        "cviu17":    ("cviu17/SRimages",        1620,  16),
        "sisar":     ("sisar",                 8428,  16),
        "qads":      ("QADS/super-resolved_images", 980, 16),
        "waterloo15":("Waterloo15/WIND_all",   312,   16),
    }

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"未知数据集: {dataset_name}")

    rel_path, count, batch_size = DATASET_CONFIGS[dataset_name]
    dataset_path = os.path.join(base_path, rel_path)
    test_index = _full_index(count) if cross_dataset else _split_index(count)
    logger.info(f"{dataset_name} ({'cross' if cross_dataset else 'split'} eval): {len(test_index)} samples")
    return build_dataloader(config, dataset_name, dataset_path, batch_size, test_index)
        

def cross_eval(config, model, logger, use_tta=True, in_domain_only=False):
    """
    评估函数，支持同数据集验证 + 跨数据集泛化验证，支持 TTA。

    返回 [in_plcc, in_srcc, cross_plcc, cross_srcc]：
    - in_plcc/in_srcc：在训练数据集的测试集上的指标
    - cross_plcc/cross_srcc：在跨数据集（全量）上的平均指标，无跨数据集时返回 0.0

    跨数据集配置（在训练集 A 上训练，在数据集 B 全量上测试）：
    """
    # 同数据集测试集（8:2 划分中的 20%）
    in_domain_dataset = config.DATA.DATASET

    # 跨数据集配置：key=训练数据集, value=跨数据集评估目标列表
    CROSS_DATASET_MAP = {
        "cviu17":    ["qads", "waterloo15"],
        "qads":      ["cviu17", "waterloo15"],
        "waterloo15":["cviu17", "qads"],
        "live":      ["csiq"],
        "csiq":      ["live"],
        "tid2013":   ["kadid"],
        "kadid":     ["tid2013"],
        "livec":     ["koniq"],
        "koniq":     ["livec"],
        "spaq":      ["livefb"],
        "livefb":    ["livec"],
    }
    cross_datasets = CROSS_DATASET_MAP.get(in_domain_dataset, [])

    result = []
    model.eval()
    model.float()  # 验证时转为 float32

    if use_tta:
        logger.info("使用 Test Time Augmentation (TTA) - 水平翻转")

    def _eval_single(dataset_name, cross_dataset=False):
        """对单个数据集运行评估，返回 (plcc, srcc)"""
        val_loader, val_len = get_dataloader(config, dataset_name, logger, cross_dataset=cross_dataset)
        temp_pred_scores = []
        temp_gt_scores = []
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_loader):
                img = batch_data[0]
                labels = batch_data[1]
                img = img.cuda(non_blocking=True).float()
                labels = labels.cuda(non_blocking=True)

                out = model(img, eval=True)
                preds = out[0] if isinstance(out, (tuple, list)) else out

                if use_tta:
                    img_flipped = torch.flip(img, dims=[3])
                    out_flipped = model(img_flipped, eval=True)
                    preds_flipped = out_flipped[0] if isinstance(out_flipped, (tuple, list)) else out_flipped
                    preds = (preds + preds_flipped) / 2.0

                temp_pred_scores.append(preds.reshape(-1))
                temp_gt_scores.append(labels.reshape(-1))

        pred_scores = torch.cat(temp_pred_scores)
        gt_scores = torch.cat(temp_gt_scores)
        gather_preds = pred_scores[:val_len]
        gather_preds = (gather_preds.view(-1, config.DATA.PATCH_NUM)).mean(dim=-1).squeeze()
        gather_grotruth = gt_scores[:val_len]
        gather_grotruth = (gather_grotruth.view(-1, config.DATA.PATCH_NUM)).mean(dim=-1).squeeze()
        final_preds = gather_preds.cpu().numpy()
        final_grotruth = gather_grotruth.cpu().numpy()

        valid_mask = np.isfinite(final_preds) & np.isfinite(final_grotruth)
        if not np.all(valid_mask):
            skip_count = np.sum(~valid_mask)
            logger.info(f"[{dataset_name}] 跳过 {skip_count} 个异常样本，剩余 {np.sum(valid_mask)} 个有效样本")
            final_preds = final_preds[valid_mask]
            final_grotruth = final_grotruth[valid_mask]

        if len(final_preds) < 2:
            return 0.0, 0.0
        srcc, _ = stats.spearmanr(final_preds, final_grotruth)
        plcc, _ = stats.pearsonr(final_preds, final_grotruth)
        return float(plcc), float(srcc)

    # 1. 同数据集评估
    in_plcc, in_srcc = _eval_single(in_domain_dataset, cross_dataset=False)
    logger.info(f"[In-domain] {in_domain_dataset}: PLCC={in_plcc:.4f}, SRCC={in_srcc:.4f}")
    result.extend([in_plcc, in_srcc])

    # 2. 跨数据集评估（全量数据，取平均）
    if cross_datasets and not in_domain_only:
        cross_plcc_list, cross_srcc_list = [], []
        for ds in cross_datasets:
            try:
                plcc, srcc = _eval_single(ds, cross_dataset=True)
                logger.info(f"[Cross-dataset] {in_domain_dataset} -> {ds}: PLCC={plcc:.4f}, SRCC={srcc:.4f}")
                cross_plcc_list.append(plcc)
                cross_srcc_list.append(srcc)
            except Exception as e:
                logger.warning(f"[Cross-dataset] {in_domain_dataset} -> {ds} 评估失败: {e}")
        if cross_plcc_list:
            result.extend([float(np.mean(cross_plcc_list)), float(np.mean(cross_srcc_list))])
        else:
            result.extend([0.0, 0.0])
    else:
        result.extend([0.0, 0.0])

    model.train()
    return result