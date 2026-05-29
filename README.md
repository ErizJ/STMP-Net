# STMP-Net

**Texture-Structure Guided Multimodal Prompt Learning for Super-Resolution Image Quality Assessment**

Accepted / In press — *ICETIS 2026*

Official implementation of STMP-Net, a multimodal prompt learning framework for super-resolution image quality assessment (SR-IQA). The method leverages four learnable prompt branches — scene, general distortion, texture distortion, and structure distortion — to guide a frozen CLIP ViT-B/16 backbone, producing quality-aware text-visual representations fused through a cross-attention transformer decoder.

## Architecture Overview

STMP-Net consists of a frozen CLIP image + text encoder backbone with four learnable prompt learners and a cross-attention fusion decoder:

- **Scene Branch (P^sc)**: Models global scene-level semantics from CLS and visual prompt embeddings.
- **General Distortion Branch (P^gd)**: Captures universal distortion types via 3×3 dual-scale window pooling over patch tokens.
- **Texture Branch (P^tx)**: Targets local high-frequency texture artifacts (noise amplification, ringing, moiré, etc.) with an independent Conv1×1 visual transform.
- **Structure Branch (P^st)**: Handles global geometric/structural degradations (edge blur, geometric distortion, aliasing) via 2×2 coarse window pooling with a dedicated visual transform.

Key components:

| Component | Description |
|---|---|
| **Prompt Learners** | Four learnable context vectors (P^sc, P^gd, P^tx, P^st) producing text features T^* through a frozen CLIP text encoder with gradient passthrough |
| **Dual-Scale Window Pooling** | 3×3 fine windows (distortion/texture) + 2×2 coarse windows (structure) with Max+Avg dual pooling |
| **Branch-Specific Vision Transforms** | Independent Conv1×1→ReLU→Conv1×1 transforms for texture and structure branches |
| **Text-Visual Alignment** | Per-branch contrastive alignment with learnable logit scales and spatial softmax reweighting |
| **Cross-Attention Decoder** | 3-layer Transformer Decoder with text features as Query and visual features (CLS+E+P) as Key/Value |
| **Learnable Branch Aggregation** | Softmax-weighted fusion of per-branch decoded scores into the final quality prediction |

## Requirements

```bash
pip install -r requirements.txt
```

- PyTorch >= 1.11.0
- torchvision >= 0.12.0
- timm >= 0.5.4
- numpy, scipy, pyyaml, yacs, einops, pillow

## Quick Start

```python
import torch
from stmp_net import STMPNet
from config import get_config
import argparse
import os

# Build config
args = argparse.Namespace(
    cfg='configs/Pure/vit_small_pre_coder_qads.yaml',
    opts=None, batch_size=None, data_path=None, zip=False,
    cache_mode='part', pretrained=None, resume=None,
    alpha=None, beta=None, accumulation_steps=None,
    use_checkpoint=False, amp_opt_level=None, disable_amp=False,
    output='output', tag='default', eval=False,
    tensorboard=False, throughput=False, debug=False,
    repeat=False, rnum=1, seed=42, depth=11,
    epoch=None, token=4, prompt=8,
    scene=True, dist=True, texture=True, structure=True,
    gamma=1.0, delta=1.0, visual=False,
    data_percent=0.8, print=False,
)
local_rank = int(os.environ.get('LOCAL_RANK', 0))
config = get_config(args, local_rank)

# NR mode (Super-Resolution image only)
model = STMPNet(config).cuda()
model.eval()
with torch.no_grad():
    score, features = model(torch.rand(1, 3, 224, 224), eval=True)
    print(f'Predicted quality score: {score.item():.4f}')
```

## Training

```bash
# Full model with all four branches
python train.py --cfg configs/Pure/vit_small_pre_coder_qads.yaml \
    --scene --dist --texture --structure \
    --output log --tag default \
    --seed 42

# Visual-only baseline (no text prompts)
python train.py --cfg configs/Pure/vit_small_pre_coder_qads_visual_only.yaml \
    --visual --output log --tag visual_only

# Key training options
python train.py --cfg <config.yaml> \
    --batch-size 16 \
    --epoch 80 \
    --alpha 1.0 --beta 0.0 \     # texture / structure loss weights
    --gamma 1.0 --delta 1.0 \     # smooth L1 / fidelity loss weights
    --seed 42 --rnum 1
```

## Evaluation

```bash
# Cross-dataset evaluation (e.g., CVIU17 → QADS)
python tools/cross_dataset_eval.py \
    --train_dataset cviu17 \
    --test_dataset qads \
    --checkpoint log/cviu17/default/ckpt_epoch_20.pth \
    --use_tta

# Batch cross-dataset evaluation
python tools/cross_dataset_eval.py \
    --all \
    --cviu17_checkpoint log/cviu17/default/1/ckpt_epoch_20.pth \
    --qads_checkpoint log/qads/default/1/ckpt_epoch_90.pth
```

## Ablation Study

```bash
# Single ablation experiment
python tools/run_single_ablation.py \
    --dataset qads \
    --experiment full_model

# Batch ablation (all variants on all datasets)
python tools/run_ablation_experiments.py \
    --datasets waterloo15 cviu17 qads \
    --mode standard

# Available ablation variants:
#   full_model, wo_scene_prompt, wo_distortion_prompt,
#   wo_texture_prompt, wo_structure_prompt,
#   single_scale_window, wo_fidelity_loss,
#   wo_texture_loss, wo_structure_loss, visual_only
```

## Datasets

| `--dataset` | Dataset | Description |
|---|---|---|
| `cviu17` | CVIU17 | SR image quality dataset |
| `qads` | QADS | Quality Assessment of Diverse SR images |
| `waterloo15` | Waterloo15 (WIND) | Waterloo SR IQA dataset |
| `sisar` | SISAR | SR IQA dataset |
| `livec` | LIVE Challenge | In-the-wild IQA |
| `live` | LIVE | Legacy IQA |
| `csiq` | CSIQ | Contrast-distorted IQA |
| `tid2013` | TID2013 | Multi-distortion IQA |
| `kadid` | KADID-10k | Artifact-distorted IQA |
| `koniq` | KonIQ-10k | In-the-wild IQA |
| `spaq` | SPAQ | Smartphone IQA |
| `livefb` | LIVE-FB | In-the-wild IQA |

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{stmpnet,
  title     = {Texture-Structure Guided Multimodal Prompt Learning for Super-Resolution Image Quality Assessment},
  booktitle = {International Conference on Electronic Technology and Information Science (ICETIS)},
  year      = {2026},
  note      = {Accepted / In press}
}
```

## License

This project is released under the MIT License.
