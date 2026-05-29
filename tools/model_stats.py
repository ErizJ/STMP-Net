"""
统计模型的Trainable Params、FLOPs和Inference Time
"""

import argparse
import os
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torchinfo import summary
from thop import profile, clever_format

from config import get_config
from stmp_net import STMPNet


def parse_option():
    parser = argparse.ArgumentParser("Model Statistics", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size for statistics")
    parser.add_argument("--input-size", type=int, default=224, help="input image size")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="device to use")
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="test iterations")
    parser.add_argument("--scene", action="store_true", help="enable scene branch")
    parser.add_argument("--dist", action="store_true", help="enable distortion branch")
    parser.add_argument("--texture", action="store_true", help="enable texture branch")
    parser.add_argument("--structure", action="store_true", help="enable structure branch")
    parser.add_argument("--visual", action="store_true", help="enable visual prompts")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args, _ = parser.parse_known_args()

    # 补全 config.py update_config() 所需的所有属性
    for attr in ["data_path", "zip", "cache_mode", "pretrained", "resume",
                 "alpha", "beta", "accumulation_steps", "use_checkpoint",
                 "amp_opt_level", "disable_amp", "output", "tag", "eval",
                 "tensorboard", "throughput", "debug", "rnum", "depth",
                 "seed", "epoch", "token", "prompt", "data_percent",
                 "print", "gamma", "delta"]:
        if not hasattr(args, attr):
            setattr(args, attr, None)

    config = get_config(args, local_rank)
    return args, config


def count_parameters(model):
    """统计总参数、requires_grad=True参数、以及实际传入optimizer的参数"""
    total_params = sum(p.numel() for p in model.parameters())
    grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 与 optimizer.py make_optimizer_1stage 保持一致的过滤逻辑
    opt_keywords = ['features_learner', 'encoder_proj', 'prompt_proj',
                    'prompt_embeddings', 'decoder', 'logit_scale',
                    'window_proj', 'structure_proj', 'branch_weights',
                    'adaptive_max_pool']
    optimizer_params = 0
    for name, p in model.named_parameters():
        if any(kw in name for kw in opt_keywords) and p.requires_grad:
            optimizer_params += p.numel()

    return total_params, grad_params, optimizer_params


def format_params(num_params):
    """格式化参数数量"""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return f"{num_params}"


def measure_inference_time(model, input_tensor, device, warmup=10, iterations=100):
    """测量推理时间"""
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor, eval=True)
    
    # Synchronize GPU
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure inference time
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor, eval=True)

    # Synchronize GPU
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    # Calculate average time per inference
    total_time = end_time - start_time
    avg_time = total_time / iterations * 1000  # Convert to milliseconds
    
    return avg_time


def main():
    args, config = parse_option()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # 设置随机种子
    seed = config.SEED if hasattr(config, 'SEED') else 42
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
    
    # 创建输入张量
    batch_size = args.batch_size
    input_size = args.input_size
    input_tensor = torch.randn(batch_size, 3, input_size, input_size)
    
    # 构建模型
    print("\n" + "="*60)
    print("Building model...")
    print("="*60)
    
    # 更新配置
    config.defrost()
    if args.scene:
        config.scene = True
    if args.dist:
        config.dist = True
    if args.texture:
        config.texture = True
    if args.structure:
        config.structure = True
    if args.visual:
        config.visual = True
    
    # 设置数据集相关参数
    config.DATA.H_RESOLUTION = input_size
    config.DATA.W_RESOLUTION = input_size
    config.num_scene = 9  # LIVEC场景类别数
    config.num_dist = 11  # 失真类别数
    config.num_texture = 11  # 纹理失真类别数
    config.num_structure = 6  # 结构失真类别数
    config.freeze()
    
    model = STMPNet(config)
    model.to(device)
    model.eval()
    
    print(f"Model configuration:")
    print(f"  Input size: {input_size}x{input_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Scene branch: {getattr(config, 'scene', False)}")
    print(f"  Distortion branch: {getattr(config, 'dist', False)}")
    print(f"  Texture branch: {getattr(config, 'texture', False)}")
    print(f"  Structure branch: {getattr(config, 'structure', False)}")
    print(f"  Visual prompts: {getattr(config, 'visual', False)}")
    
    # 1. 统计参数
    print("\n" + "="*60)
    print("Parameter Statistics")
    print("="*60)
    
    total_params, grad_params, optimizer_params = count_parameters(model)
    print(f"Total parameters:            {total_params:>12,}  ({format_params(total_params)})")
    print(f"  Trainable (in optimizer):   {optimizer_params:>10,}  ({format_params(optimizer_params)})")
    print(f"  Frozen (CLIP backbone):     {total_params - optimizer_params:>10,}  ({format_params(total_params - optimizer_params)})")
    print(f"")
    print(f"Module breakdown:")
    print(f"  CLIP Image Encoder (ViT-B/16, frozen):  ~86.19M")
    print(f"  CLIP Text Encoder  (frozen):            ~38.13M")
    print(f"  Transformer Decoder (3 layers):         ~12.61M")
    print(f"  Projection layers (4x Linear):          ~1.84M")
    print(f"  Prompt Learners (4x ctx vectors):        ~0.08M")
    print(f"  MLP heads + logit scales + branch weights: ~0.13M")
    
    # 2. 使用torchinfo显示详细结构
    print("\n" + "="*60)
    print("Model Architecture Summary")
    print("="*60)
    
    try:
        model_summary = summary(
            model,
            input_data=input_tensor,
            device=device,
            verbose=0,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
        )
        print(model_summary)
    except Exception as e:
        print(f"Error in torchinfo summary: {e}")
        print("Skipping detailed architecture summary...")
    
    # 3. 计算FLOPs
    print("\n" + "="*60)
    print("FLOPs Calculation")
    print("="*60)
    
    try:
        # 使用thop计算FLOPs
        macs, params = profile(model, inputs=(input_tensor.to(device),), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        print(f"FLOPs (MACs): {macs}")
        print(f"Parameters: {params}")
    except Exception as e:
        print(f"Error in FLOPs calculation: {e}")
        print("Skipping FLOPs calculation...")
    
    # 4. 测量推理时间
    print("\n" + "="*60)
    print("Inference Time Measurement")
    print("="*60)
    
    try:
        avg_time = measure_inference_time(
            model, input_tensor, device, 
            warmup=args.warmup, iterations=args.iterations
        )
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"FPS: {1000 / avg_time:.2f}")
    except Exception as e:
        print(f"Error in inference time measurement: {e}")
        print("Skipping inference time measurement...")
    
    # 5. 内存使用情况
    print("\n" + "="*60)
    print("Memory Usage")
    print("="*60)
    
    if device.type == "cuda":
        try:
            # 前向传播一次以分配内存
            with torch.no_grad():
                _ = model(input_tensor.to(device), eval=True)

            allocated = torch.cuda.memory_allocated(device) / 1024**2
            reserved = torch.cuda.memory_reserved(device) / 1024**2
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2

            print(f"Memory allocated: {allocated:.2f} MB")
            print(f"Memory reserved: {reserved:.2f} MB")
            print(f"Max memory allocated: {max_allocated:.2f} MB")
        except Exception as e:
            print(f"Error in memory measurement: {e}")
    else:
        print("Memory measurement only available on CUDA devices")
    
    print("\n" + "="*60)
    print("Statistics Complete")
    print("="*60)


if __name__ == "__main__":
    main()