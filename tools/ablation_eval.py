"""
消融实验独立评估脚本
直接加载 checkpoint，调用 cross_eval，输出 PLCC/SRCC 到标准输出和日志文件。

用法:
    python ablation_eval.py --cfg <config.yaml> --resume <checkpoint.pth> --output <output_dir>
"""

import argparse
import logging
import os
import pickle
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import get_config
from eval import cross_eval
from logger import create_logger
from stmp_net import STMPNet


def parse_option():
    parser = argparse.ArgumentParser("消融实验评估脚本", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, help="配置文件路径")
    parser.add_argument("--resume", type=str, required=True, help="checkpoint 路径")
    parser.add_argument("--output", type=str, default="ablation_eval_out", help="输出目录")
    parser.add_argument("--opts", nargs="+", default=None)
    # 以下参数保持与 train.py 兼容，供 get_config 使用
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache-mode", type=str, default="part")
    parser.add_argument("--pretrained")
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--amp-opt-level", type=str)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--eval", action="store_true", default=True)
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--rnum", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--token", type=int)
    parser.add_argument("--prompt", type=int)
    parser.add_argument("--scene", action="store_true")
    parser.add_argument("--dist", action="store_true")
    parser.add_argument("--texture", action="store_true")
    parser.add_argument("--structure", action="store_true")
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--data_percent", type=float)
    parser.add_argument("--print", action="store_true")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args, _ = parser.parse_known_args()
    # 强制 eval 模式
    args.eval = True
    config = get_config(args, local_rank)
    return args, config


def load_model(config, checkpoint_path, logger):
    """加载模型和 checkpoint"""
    model = STMPNet(config)
    model.cuda()

    logger.info(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"]

    # branch_weights 在消融时分支数可能与 checkpoint 不一致，跳过以避免 size mismatch
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k not in model_state or v.shape == model_state[k].shape
    }
    skipped = [k for k in state_dict if k in model_state and state_dict[k].shape != model_state[k].shape]
    if skipped:
        logger.warning(f"跳过 size 不匹配的参数: {skipped}")

    msg = model.load_state_dict(filtered, strict=False)
    logger.info(f"load_state_dict: {msg}")
    del checkpoint
    torch.cuda.empty_cache()
    return model


def prepare_test_index(config, logger):
    """准备测试集索引（与 train.py 保持一致的划分逻辑）"""
    os.makedirs(config.SEL_PATH, exist_ok=True)
    filename = f"{config.SEED}_sel_num0.data"
    sel_path = os.path.join(config.SEL_PATH, filename)

    if os.path.exists(sel_path):
        logger.info(f"使用已有划分文件: {sel_path}")
        with open(sel_path, "rb") as f:
            sel_num = pickle.load(f)
    else:
        logger.info("生成新的数据划分")
        sel_num = list(range(0, config.SET.COUNT))
        random.shuffle(sel_num)
        with open(sel_path, "wb") as f:
            pickle.dump(sel_num, f)

    config.defrost()
    config.SET.TRAIN_INDEX = sel_num[0: int(round(config.data_percent * len(sel_num)))]
    config.SET.TEST_INDEX = sel_num[int(round(0.8 * len(sel_num))): len(sel_num)]
    config.freeze()
    logger.info(f"测试集大小: {len(config.SET.TEST_INDEX)}")


def main():
    args, config = parse_option()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = logging.getLogger(name=f"{config.MODEL.NAME}")
    create_logger(logger, output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    logger.info(f"配置: {config}")

    # 固定随机种子
    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # 准备测试集索引
    prepare_test_index(config, logger)

    # 加载模型
    model = load_model(config, config.MODEL.RESUME, logger)

    # 执行评估
    use_tta = getattr(config.TEST, 'USE_TTA', False)
    logger.info(f"开始评估 (TTA={'开启' if use_tta else '关闭'})")

    result = cross_eval(config, model, logger, use_tta=use_tta, in_domain_only=True)
    val_plcc = result[0] if len(result) > 0 else 0.0
    val_srcc = result[1] if len(result) > 1 else 0.0

    logger.info(f"ABLATION_RESULT PLCC={val_plcc:.6f} SRCC={val_srcc:.6f}")
    # 同时打印到 stdout，方便父进程捕获
    print(f"ABLATION_RESULT PLCC={val_plcc:.6f} SRCC={val_srcc:.6f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
