"""
STMP-Net Network Architecture
=============================
Multi-Prompt Super-Resolution Image Quality Assessment

Corresponds to Chapter 4 of the thesis.  Every component below is labelled
with the notation used in the paper so that the code, paper text, and
framework diagram stay in sync.

Architecture (paper notation → code):

  Input image x
    │
    ▼
  ┌─ Frozen CLIP ViT-B/16 ─────────────────────────────────────────┐
  │  outputs:  CLS  (global semantic)                               │
  │            E    (visual prompt tokens, learnable)                │
  │            P    (patch tokens, 14×14 grid)                       │
  └─────────────────────────────────────────────────────────────────┘
         │                │                │
         ▼                ▼                ▼
  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────────┐
  │ Scene Branch │ │ Dist  Branch │ │ Texture / Structure Branches │
  │              │ │              │ │                              │
  │ CLS + E_avg  │ │ P→3×3 pool   │ │ P→3×3 pool→TexVision(Conv)  │
  │   ↓          │ │   ↓          │ │ P→2×2 pool→StrVision(Conv)  │
  │ scene_proj   │ │ win_features │ │   ↓              ↓           │
  │   ↓          │ │   ↓          │ │ tex_vis_feat   str_vis_feat │
  │ ↔ T^sc       │ │ ↔ T^gd       │ │ ↔ T^tx         ↔ T^st       │
  │ L_scene      │ │ L_distortion │ │ L_texture      L_structure  │
  └──────────────┘ └──────────────┘ └──────────────────────────────┘
         │                │                │                │
         └────────────────┴────────────────┴────────────────┘
                                  │
                    Text features → Q_sc, Q_gd, Q_tx, Q_st
                    (linear projection to query space)
                                  │
                                  ▼
         ┌──────────────────────────────────────────────┐
         │  Transformer Decoder (3-layer, cross-attn)    │
         │  Query  ← concat[Q_sc, Q_gd, Q_tx, Q_st]     │
         │  Key/Val ← concat[CLS, E, P]  (visual mem)   │
         │  → decoded multimodal features                │
         └──────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Regression MLP heads    │
                    │  Branch-weight aggregation│
                    │  → predicted quality score│
                    └─────────────────────────┘

Key design principles (Section 4.3):
  - Frozen CLIP image & text encoder parameters (requires_grad=False)
  - Four learnable prompt learners P^sc, P^gd, P^tx, P^st
  - Text encoder forward ALLOWS gradient flow to prompt ctx
    (no torch.no_grad wrapper on text encoder)
  - Branch-specific visual transforms (texture ≠ distortion)
  - Cross-attention fusion: text semantics query visual evidence
"""

import math
import os
from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Dropout, Identity
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from timm.models.layers import trunc_normal_

from models.clip import clip
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# ============================================================================
# Category definitions  (Table 4.1 / Section 4.2)
# ============================================================================

scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape',
          'night', 'plant', 'still_life', 'others']                 # 9 classes

# 通用失真 (General Distortion) — 11 classes, 与论文表 4-1 一致
# 注意：此分支不包含 'none'/'uncertain'，因为通用失真描述的是 SR 图像中
# 必然存在的退化类型（JPEG/噪声/模糊等），与纹理/结构分支的 'none' 语义不同。
dists_map = ['jpeg2000 compression', 'jpeg compression', 'noise', 'blur',
             'color', 'contrast', 'overexposure', 'underexposure',
             'spatial', 'quantization', 'other']

texture_dists = ['none', 'uncertain', 'noise_amplification', 'ringing_halo',
                 'checkerboard', 'moire', 'false_texture_hallucination',
                 'texture_smoothing', 'over_sharpening',
                 'compression_blockiness', 'other_artifact']        # 11 classes

structure_dists = ['none', 'uncertain', 'edge_blur', 'detail_loss',
                   'geometric_distortion', 'aliasing_jaggies']      # 6 classes


# ============================================================================
# Utility
# ============================================================================

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def load_clip_to_cpu(config, h, w):
    """Download / load frozen CLIP ViT backbone."""
    url = clip._MODELS[config.MODEL.BACKBONE]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict(), h, w)
    return model


# ============================================================================
# Text Encoder  (frozen CLIP text transformer)
# ============================================================================

class TextEncoder(nn.Module):
    """
    Wraps CLIP's frozen text transformer.

    IMPORTANT (Section 4.3.1):  The encoder parameters are frozen via
    requires_grad_(False), but we do NOT use torch.no_grad() during forward.
    This allows gradients from the contrastive / alignment losses to flow
    back through the frozen encoder to the learnable prompt ctx parameters.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_featuress):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)                     # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)                     # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # eot token (highest index) → text projection space
        idx = torch.arange(x.shape[0], device=x.device)
        eot = tokenized_featuress.argmax(dim=-1).to(x.device)
        x = x[idx, eot]
        x = x @ self.text_projection
        return x


# ============================================================================
# Branch-specific visual transforms  (Section 4.3.3)
# ============================================================================

class _BranchVisionTransform(nn.Module):
    """
    Per-branch visual feature transform:  Conv1×1 → ReLU → Conv1×1.
    Projects pooled window features into branch-specific visual space.
    Used by texture and structure branches to ensure independent
    visual representations (Section 4.3.3, Fig. 4.3).
    """
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, x_2d):
        # x_2d: (B, dim, H, W)  e.g. (B, 512, 3, 3) or (B, 512, 2, 2)
        return self.net(x_2d)


# ============================================================================
# Prompt Learners  (Section 4.3.2, Fig. 4.2)
# ============================================================================

class _BasePromptLearner(nn.Module):
    """
    Shared base for all four prompt learners.
    Learns context vectors ctx ∈ R^{n_cls × n_ctx × 512} that, when
    combined with class names and frozen token embeddings, produce
    prompt embeddings P^* → text encoder → T^* (text features).
    """
    class_token_position = "end"

    def __init__(self, config, num_class, dtype, token_embedding,
                 classnames, ctx_init_text):
        super().__init__()
        self.n_cls = num_class

        # Prompt 长度由 config.TRAIN.COOP_N_CTX 控制（论文中上下文长度超参数 M）
        target_n_ctx = config.TRAIN.COOP_N_CTX
        self.n_ctx = target_n_ctx

        # 语义初始化：用初始化文本的 token embedding 填充 ctx_vectors
        ctx_init = ctx_init_text.replace("_", " ")
        init_words = ctx_init.split(" ")
        prompt_tok = clip.tokenize(ctx_init)
        with torch.no_grad():
            emb = token_embedding(prompt_tok.to('cuda')).type(dtype)
        # emb[0, 1:1+len(init_words), :] 是初始化文本的 token embedding
        init_len = min(len(init_words), target_n_ctx)
        ctx_vectors = torch.empty(target_n_ctx, 512, dtype=dtype)
        # 前 init_len 个 token 用语义初始化
        ctx_vectors[:init_len] = emb[0, 1:1 + init_len, :]
        # 剩余 token 随机初始化
        if target_n_ctx > init_len:
            nn.init.normal_(ctx_vectors[init_len:], std=0.02)
        ctx_vectors = ctx_vectors.unsqueeze(0).expand(num_class, -1, -1).clone()
        prompt_prefix = ctx_init

        print(f'  Prompt init: "{prompt_prefix}"  (n_ctx={target_n_ctx}, '
              f'semantic_tokens={init_len}, random_tokens={target_n_ctx - init_len}, '
              f'n_cls={num_class})')
        self.ctx = nn.Parameter(ctx_vectors)       # learnable !

        # 构造完整 prompt: prefix_ctx + class_name + suffix
        # 注意：prefix/suffix 的切分基于 target_n_ctx（而非 init_len）
        prefix_text = prompt_prefix
        if target_n_ctx > init_len:
            prefix_text = prompt_prefix + " " + "X " * (target_n_ctx - init_len)
            prefix_text = prefix_text.strip()
        prompts = [prefix_text + " " + name + "." for name in classnames]
        tokenized_featuress = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = token_embedding(tokenized_featuress.to('cuda')).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])               # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + target_n_ctx:, :]) # class name + EOS
        self.tokenized_featuress = tokenized_featuress
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]

    def forward(self, label=None):
        if label is None:
            ctx = self.ctx
            prefix = self.token_prefix
            suffix = self.token_suffix
            tokenized_featuress = self.tokenized_featuress
        else:
            ctx = self.ctx[label:label + 1] if isinstance(label, int) else self.ctx[label]
            prefix = self.token_prefix[label:label + 1] if isinstance(label, int) else self.token_prefix[label]
            suffix = self.token_suffix[label:label + 1] if isinstance(label, int) else self.token_suffix[label]
            tokenized_featuress = self.tokenized_featuress[label:label + 1] if isinstance(label, int) else self.tokenized_featuress[label]

        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts, tokenized_featuress


class GlobalPromptLearner(_BasePromptLearner):
    """P^sc — Scene prompt learner  (9 classes, Section 4.3.2)"""
    def __init__(self, config, num_class, dtype, token_embedding):
        super().__init__(config, num_class, dtype, token_embedding,
                         classnames=[scenes[i] for i in range(num_class)],
                         ctx_init_text="a super-resolution image with")


class LocalPromptLearner(_BasePromptLearner):
    """P^gd — General distortion prompt learner  (11 classes, Section 4.3.2)"""
    def __init__(self, config, num_class, dtype, token_embedding):
        super().__init__(config, num_class, dtype, token_embedding,
                         classnames=[dists_map[i] for i in range(num_class)],
                         ctx_init_text="a super-resolution image with")


class TexturePromptLearner(_BasePromptLearner):
    """P^tx — Texture distortion prompt learner  (11 classes, Section 4.3.2)"""
    def __init__(self, config, num_class, dtype, token_embedding):
        super().__init__(config, num_class, dtype, token_embedding,
                         classnames=[texture_dists[i] for i in range(num_class)],
                         ctx_init_text="a super-resolution image with texture")


class StructurePromptLearner(_BasePromptLearner):
    """P^st — Structure distortion prompt learner  (6 classes, Section 4.3.2)"""
    def __init__(self, config, num_class, dtype, token_embedding):
        super().__init__(config, num_class, dtype, token_embedding,
                         classnames=[structure_dists[i] for i in range(num_class)],
                         ctx_init_text="a super-resolution image with structure")


# ============================================================================
# Core Model: STMPNet  (STMP-Net, Section 4.3)
# ============================================================================

class STMPNet(nn.Module):
    """
    STMP-Net: Multi-Prompt Super-Resolution Image Quality Assessment.

    Implements the full architecture described in Chapter 4:
      - Frozen CLIP ViT-B/16 image encoder
      - Frozen CLIP text encoder (gradients pass through to prompt ctx)
      - 4 learnable prompt learners → T^sc, T^gd, T^tx, T^st
      - Branch-specific visual transforms
      - Cross-attention fusion (text → query, visual → key/value)
      - Quality score regression with branch-weight aggregation
    """

    def __init__(self, config):
        super().__init__()
        self.h = config.DATA.H_RESOLUTION // config.MODEL.VIT.PATCH_SIZE
        self.w = config.DATA.W_RESOLUTION // config.MODEL.VIT.PATCH_SIZE
        self.dim = 512

        # ── Load CLIP backbone ──
        clip_model = load_clip_to_cpu(config, self.h, self.w)
        clip_model.to("cuda")
        clip_model.float()

        # ── FREEZE CLIP image encoder (Section 4.3.1, Eq. 4.1) ──
        self.image_encoder = clip_model.visual
        for p in self.image_encoder.parameters():
            p.requires_grad_(False)

        # ── FREEZE CLIP text encoder (Section 4.3.1, Eq. 4.2) ──
        self.text_encoder = TextEncoder(clip_model)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        self.dtype = clip_model.dtype

        # ── 4 learnable Prompt Learners (Section 4.3.2) ──
        self.num_scene = config.num_scene
        self.num_dist = config.num_dist
        self.num_texture = config.num_texture
        self.num_structure = config.num_structure

        self.global_features_learner = GlobalPromptLearner(
            config, config.num_scene, clip_model.dtype, clip_model.token_embedding)
        self.local_features_learner = LocalPromptLearner(
            config, config.num_dist, clip_model.dtype, clip_model.token_embedding)

        self.texture = getattr(config, 'texture', False)
        self.structure = getattr(config, 'structure', False)
        if self.texture:
            self.texture_features_learner = TexturePromptLearner(
                config, config.num_texture, clip_model.dtype, clip_model.token_embedding)
        if self.structure:
            self.structure_features_learner = StructurePromptLearner(
                config, config.num_structure, clip_model.dtype, clip_model.token_embedding)

        # ── Learnable logit scales (Eq. 4.7) ──
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.spatial_logit_scale = nn.Parameter(torch.tensor(3.0, dtype=self.dtype))

        # ── Dual-scale window pooling (Section 4.3.3) ──
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((3, 3))        # fine  (3×3)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.adaptive_max_pool_structure = nn.AdaptiveMaxPool2d((2, 2))  # coarse (2×2)
        self.adaptive_avg_pool_structure = nn.AdaptiveAvgPool2d((2, 2))

        self.single_scale_window = getattr(config, 'single_scale_window', False)

        # ── Scene branch: CLS + E aggregation (Section 4.3.3, Fig. 4.3 left) ──
        self.scene_proj = nn.Linear(self.dim * 2, self.dim)

        # ── General distortion branch: window pool → projection (Section 4.3.3) ──
        self.window_proj = nn.Linear(self.dim * 2, self.dim)

        # ── Texture branch: independent visual transform (Section 4.3.3) ──
        self.texture_vision_transform = _BranchVisionTransform(dim=self.dim)
        self.texture_vision_proj = nn.Linear(self.dim * 2, self.dim)

        # ── Structure branch: independent visual transform + projection (Section 4.3.3) ──
        self.structure_vision_transform = _BranchVisionTransform(dim=self.dim)
        self.structure_vision_proj = nn.Linear(self.dim * 2, self.dim)

        # ── Text→Query projections for multi-modal fusion (Section 4.3.4) ──
        self.query_proj_scene = nn.Linear(self.dim, self.dim)
        self.query_proj_dist = nn.Linear(self.dim, self.dim)
        self.query_proj_texture = nn.Linear(self.dim, self.dim)
        self.query_proj_structure = nn.Linear(self.dim, self.dim)

        # ── Transformer Decoder (Section 4.3.4, Fig. 4.3 right) ──
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.dim, dropout=0.0, nhead=8,
            activation=F.gelu, batch_first=True,
            dim_feedforward=self.dim * 4, norm_first=True,
        )
        self.bunch_decoder = nn.TransformerDecoder(dec_layer, num_layers=3)

        # ── Visual prompt tokens E (Section 4.3.3) ──
        self.visual = config.visual
        self.num_tokens = config.MODEL.NUM_TOKENS
        self.prompt_dropout = Dropout(config.MODEL.DROPOUT)

        self.prompt_proj = nn.Linear(self.dim, 768)
        self.encoder_proj = nn.Linear(768, self.dim)
        nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal_(self.encoder_proj.weight, a=0, mode='fan_out')

        if self.visual:
            patch_size = _pair(config.MODEL.VIT.PATCH_SIZE)
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + self.dim))

            self.prompt_embeddings = nn.Parameter(
                torch.zeros(1, self.num_tokens, self.dim))
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            self.depth = config.DEPTH
            self.deep_features_embeddings = nn.Parameter(
                torch.zeros(self.depth, self.num_tokens, self.dim))
            nn.init.uniform_(self.deep_features_embeddings.data, -val, val)

        # ── Regression head (Section 4.3.5) ──
        self.decoder_mlp1 = nn.Sequential(
            nn.Linear(self.dim, 256), nn.ReLU(), nn.Linear(256, 1),
        )

        # ── Branch flags ──
        self.scene = config.scene
        self.dist = config.dist
        self.visual_only = getattr(config, "visual_only", False)
        self.visual_only_num_queries = int(getattr(config, "visual_only_num_queries", 20))
        if self.visual_only:
            self.visual_only_queries = nn.Parameter(
                torch.randn(1, self.visual_only_num_queries, self.dim))

        # ── Learnable branch-weight aggregation (Section 4.3.5, Eq. 4.12) ──
        self.branch_weights = nn.Parameter(torch.ones(4) / 4)
        self.branch_weight_activation = nn.Softmax(dim=0)

    # =====================================================================
    # Prompt → Text features  (Section 4.3.2)
    # =====================================================================

    def invalidate_text_cache(self):
        """Call after optimizer.step() to force text feature recomputation."""
        self._text_cache = {}

    def _get_text_features(self, key, learner, label=None):
        """
        Compute text features T^* for a branch.
        NO torch.no_grad() — gradients flow from T^* back to ctx
        through the frozen (requires_grad=False) text encoder.
        """
        if not self.training:
            # Eval mode: cache for speed (gradients not needed)
            if not hasattr(self, '_text_cache'):
                self._text_cache = {}
            if key not in self._text_cache:
                prompts, toks = learner() if label is None else learner(label)
                self._text_cache[key] = self.text_encoder(prompts, toks)
            return self._text_cache[key]
        # Training mode: recompute every time so prompt ctx gets gradients
        prompts, toks = learner() if label is None else learner(label)
        return self.text_encoder(prompts, toks)

    def get_scene_features(self, label=None):
        return self._get_text_features('scene', self.global_features_learner, label)

    def get_dist_features(self, label=None):
        return self._get_text_features('dist', self.local_features_learner, label)

    def get_texture_features(self, label=None):
        return self._get_text_features('texture', self.texture_features_learner, label)

    def get_structure_features(self, label=None):
        return self._get_text_features('structure', self.structure_features_learner, label)

    # =====================================================================
    # Text → Query projection  (Section 4.3.4)
    # =====================================================================

    def _build_query(self, B):
        """
        Build multi-modal decoder query from text features.
        Q^* = query_proj_*(T^*)  for each active branch.
        Returns:  (B, total_query_tokens, dim)
        """
        query_list = []
        if self.scene:
            T_sc = self.get_scene_features()           # (n_scene, dim)
            Q_sc = self.query_proj_scene(T_sc)          # (n_scene, dim)
            query_list.append(Q_sc)
        if self.dist:
            T_gd = self.get_dist_features()             # (n_dist, dim)
            Q_gd = self.query_proj_dist(T_gd)           # (n_dist, dim)
            query_list.append(Q_gd)
        if self.texture:
            T_tx = self.get_texture_features()          # (n_texture, dim)
            Q_tx = self.query_proj_texture(T_tx)        # (n_texture, dim)
            query_list.append(Q_tx)
        if self.structure:
            T_st = self.get_structure_features()        # (n_structure, dim)
            Q_st = self.query_proj_structure(T_st)      # (n_structure, dim)
            query_list.append(Q_st)

        if len(query_list) > 0:
            query = torch.cat(query_list, dim=0)        # (total, dim)
            return query.unsqueeze(0).expand(B, -1, -1) # (B, total, dim)
        if self.visual_only:
            return self.visual_only_queries.expand(B, -1, -1)
        raise ValueError(
            "At least one branch (scene/dist/texture/structure) must be "
            "enabled, or set visual_only=True")

    # =====================================================================
    # Vision encoder:  CLS, E, P  extraction  (Section 4.3.3, Eq. 4.3-4.6)
    # =====================================================================

    def forward_deep_features(self, x):
        """
        Pass image through frozen ViT with learnable visual prompts E.
        Returns raw embedding before feature splitting.
        """
        B = x.shape[0]
        x = self.image_encoder.get_embedding(x)
        if self.visual:
            E_patch = self.prompt_dropout(
                self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1))
            embedding_output = torch.cat((
                x[:, :1, :],        # CLS
                E_patch,             # E (visual prompt tokens)
                x[:, 1:, :],        # P (image patch tokens)
            ), dim=1)
        else:
            embedding_output = x

        hidden_states = self.image_encoder.ln_pre(embedding_output)

        if self.visual:
            for i in range(12):
                if i > 0:
                    deep_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_features_embeddings[i - 1]).expand(B, -1, -1))
                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_emb,
                        hidden_states[:, 1 + self.num_tokens:, :],
                    ), dim=1)
                hidden_states = hidden_states.permute(1, 0, 2)
                hidden_states = self.image_encoder.transformer.resblocks[i](hidden_states)
                hidden_states = hidden_states.permute(1, 0, 2)
        else:
            hidden_states = hidden_states.permute(1, 0, 2)
            hidden_states = self.image_encoder.transformer(hidden_states)
            hidden_states = hidden_states.permute(1, 0, 2)

        hidden_states = self.image_encoder.ln_post(hidden_states)
        encoded = self.encoder_proj(hidden_states)
        return encoded

    def get_image_features(self, x):
        """
        Extract structured visual features.

        Returns:
          encoded_features   (B, 1+N+P, dim)  — full sequence for decoder memory
          cls_features       (B, 1, dim)       — CLS  (global semantic)
          prompt_features    (B, N, dim) or None — E  (visual prompt tokens)
          patch_features     (B, 196, dim)      — P  (patch tokens, 14×14 grid)
          scene_vis          (B, dim)           — CLS + E_avg  → scene_proj
          fine_win           (B, 9, dim)        — 3×3 window  (distortion branch)
          texture_vis        (B, 9, dim)        — 3×3 → tex_vision_transform
          structure_vis      (B, 4, dim)        — 2×2 → str_vision_transform
        """
        B = x.shape[0]
        embedding = self.forward_deep_features(x)

        if self.visual:
            cls_features = embedding[:, :1, :]                         # CLS
            prompt_features = embedding[:, 1:1 + self.num_tokens, :]   # E
            patch_features = embedding[:, 1 + self.num_tokens:, :]     # P
        else:
            cls_features = embedding[:, :1, :]                         # CLS
            prompt_features = None                                      # no E
            patch_features = embedding[:, 1:, :]                        # P

        # ── Decoder memory: CLS + E + P  (Section 4.3.4) ──
        if prompt_features is not None:
            encoded_features = torch.cat(
                (cls_features, prompt_features, patch_features), dim=1)
        else:
            encoded_features = torch.cat((cls_features, patch_features), dim=1)

        # ── Scene branch visual: CLS + avg(E) → scene_proj (Section 4.3.3) ──
        if prompt_features is not None:
            E_avg = prompt_features.mean(dim=1, keepdim=True)           # (B, 1, dim)
            scene_vis = self.scene_proj(
                torch.cat([cls_features, E_avg], dim=-1))              # (B, 1, dim)
        else:
            # No visual prompt: scene visual = CLS (compatible fallback)
            scene_vis = self.scene_proj(
                torch.cat([cls_features, cls_features], dim=-1))

        # ── Window pooling on P (Section 4.3.3) ──
        p2d = patch_features.reshape(
            B, self.h, self.w, self.dim).permute(0, 3, 1, 2)  # (B, 512, h, w)

        # Fine windows (3×3) → general distortion
        w_max = self.adaptive_max_pool(p2d)                            # (B, 512, 3, 3)
        w_avg = self.adaptive_avg_pool(p2d)                            # (B, 512, 3, 3)
        w_max_flat = w_max.permute(0, 2, 3, 1).reshape(B, 9, self.dim)
        w_avg_flat = w_avg.permute(0, 2, 3, 1).reshape(B, 9, self.dim)
        fine_win = self.window_proj(
            torch.cat([w_max_flat, w_avg_flat], dim=-1))              # (B, 9, dim)

        # ── Texture branch visual: independent transform (Section 4.3.3) ──
        # Same 3×3 pooled features, but through independent Conv1×1 transform
        tex_2d = self.texture_vision_transform(w_max)                   # (B, 512, 3, 3)
        tex_2d_avg = self.texture_vision_transform(w_avg)               # (B, 512, 3, 3)
        tex_flat_max = tex_2d.permute(0, 2, 3, 1).reshape(B, 9, self.dim)
        tex_flat_avg = tex_2d_avg.permute(0, 2, 3, 1).reshape(B, 9, self.dim)
        texture_vis = self.texture_vision_proj(
            torch.cat([tex_flat_max, tex_flat_avg], dim=-1))          # (B, 9, dim)

        # ── Structure branch visual: independent transform (Section 4.3.3) ──
        if self.single_scale_window:
            # Ablation: same 3×3 scale as fine window
            str_max = self.adaptive_max_pool(p2d)
            str_avg = self.adaptive_avg_pool(p2d)
        else:
            str_max = self.adaptive_max_pool_structure(p2d)             # (B, 512, 2, 2)
            str_avg = self.adaptive_avg_pool_structure(p2d)             # (B, 512, 2, 2)
        str_2d = self.structure_vision_transform(str_max)               # (B, 512, 2, 2)
        str_2d_avg = self.structure_vision_transform(str_avg)           # (B, 512, 2, 2)
        H_s = str_2d.shape[2]  # 2 or 3
        str_flat_max = str_2d.permute(0, 2, 3, 1).reshape(B, H_s * H_s, self.dim)
        str_flat_avg = str_2d_avg.permute(0, 2, 3, 1).reshape(B, H_s * H_s, self.dim)
        structure_vis = self.structure_vision_proj(
            torch.cat([str_flat_max, str_flat_avg], dim=-1))          # (B, 4, dim) or (B, 9, dim)

        return (encoded_features, cls_features, prompt_features,
                patch_features, scene_vis, fine_win, texture_vis, structure_vis)

    # =====================================================================
    # Branch score aggregation  (Section 4.3.5, Eq. 4.12)
    # =====================================================================

    def _aggregate_branch_scores(self, decoded_features):
        """
        Split decoded features into per-branch segments, mean-pool each,
        then weighted sum with learnable branch_weights.
        Order MUST match _build_query: scene, dist, texture, structure.
        """
        branch_scores = []
        start_idx = 0
        for enabled, num_cls in [
            (self.scene, self.num_scene),
            (self.dist, self.num_dist),
            (self.texture, self.num_texture),
            (self.structure, self.num_structure),
        ]:
            if enabled:
                end_idx = start_idx + num_cls
                branch_scores.append(
                    decoded_features[:, start_idx:end_idx].mean(dim=1, keepdim=True))
                start_idx = end_idx

        if not branch_scores:
            return decoded_features.mean(dim=1)

        branch_scores = torch.cat(branch_scores, dim=1)       # (B, num_active)
        n_active = branch_scores.shape[1]
        weights = self.branch_weight_activation(
            self.branch_weights[:n_active])                    # softmax
        return torch.sum(branch_scores * weights.unsqueeze(0), dim=1)

    # =====================================================================
    # Forward  (Section 4.3, Algorithm 4.1)
    # =====================================================================

    def forward(self, x, eval=False):
        B = x.shape[0]

        # ── Step 1: Extract visual features ──
        (encoded_features, cls_features, prompt_features,
         patch_features, scene_vis, fine_win, texture_vis, structure_vis) = \
            self.get_image_features(x)

        if eval:
            # ── Eval mode: decoder + regression only ──
            # temp_feat: 多模态融合特征，可用于 t-SNE / 特征可视化
            query = self._build_query(B)
            temp_feat = self.bunch_decoder(query, encoded_features)
            decoded = self.decoder_mlp1(temp_feat).squeeze(dim=-1)  # (B, total_tokens)

            if self.visual_only and not (self.scene or self.dist or
                                          self.texture or self.structure):
                predict_score = decoded.mean(dim=1)
            else:
                predict_score = self._aggregate_branch_scores(decoded)

            return predict_score, torch.mean(temp_feat, dim=1)

        # ── Step 2: Normalize visual features ──
        scene_vis_norm = scene_vis.squeeze(1)
        scene_vis_norm = scene_vis_norm / scene_vis_norm.norm(dim=-1, keepdim=True)
        fine_win_norm = fine_win / fine_win.norm(dim=-1, keepdim=True)
        texture_vis_norm = texture_vis / texture_vis.norm(dim=-1, keepdim=True)
        structure_vis_norm = structure_vis / structure_vis.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        spatial_logit_scale = self.spatial_logit_scale.exp()

        # ── Step 3: Branch-specific text-visual alignment (Eq. 4.7-4.10) ──
        logits_global = None
        logits_local = None
        logits_texture = None
        logits_structure = None

        if self.scene:
            # L_scene: scene_vis ↔ T^sc  (Eq. 4.7)
            T_sc = self.get_scene_features()
            T_sc = T_sc / T_sc.norm(dim=-1, keepdim=True)
            logits_global = logit_scale * scene_vis_norm @ T_sc.t()

        if self.dist:
            # L_distortion: fine_win ↔ T^gd  (Eq. 4.8)
            T_gd = self.get_dist_features()
            T_gd = T_gd / T_gd.norm(dim=-1, keepdim=True)
            logits_ = logit_scale * fine_win_norm @ T_gd.t()
            prob = F.softmax(logits_ * spatial_logit_scale, dim=1)
            logits_local = torch.sum(logits_ * prob, dim=1)

        if self.texture:
            # L_texture: texture_vis ↔ T^tx  (Eq. 4.9)
            T_tx = self.get_texture_features()
            T_tx = T_tx / T_tx.norm(dim=-1, keepdim=True)
            logits_t = logit_scale * texture_vis_norm @ T_tx.t()
            prob_t = F.softmax(logits_t * spatial_logit_scale, dim=1)
            logits_texture = torch.sum(logits_t * prob_t, dim=1)

        if self.structure:
            # L_structure: structure_vis ↔ T^st  (Eq. 4.10)
            T_st = self.get_structure_features()
            T_st = T_st / T_st.norm(dim=-1, keepdim=True)
            logits_s = logit_scale * structure_vis_norm @ T_st.t()
            prob_s = F.softmax(logits_s * spatial_logit_scale, dim=1)
            logits_structure = torch.sum(logits_s * prob_s, dim=1)

        # ── Step 4: Multi-modal fusion (Section 4.3.4, Eq. 4.11) ──
        # Query: projected text features  Q^* = proj(T^*)
        # Memory: CLS + E + P
        query = self._build_query(B)
        decoded = self.bunch_decoder(query, encoded_features)
        decoded = self.decoder_mlp1(decoded).squeeze(dim=-1)

        # ── Step 5: Quality score prediction (Section 4.3.5, Eq. 4.12) ──
        if self.visual_only and not (self.scene or self.dist or
                                      self.texture or self.structure):
            predict_score = decoded.mean(dim=1)
        else:
            predict_score = self._aggregate_branch_scores(decoded)

        return (predict_score,
                logits_global.squeeze(dim=1) if logits_global is not None else None,
                logits_local if logits_local is not None else None,
                logits_texture if logits_texture is not None else None,
                logits_structure if logits_structure is not None else None)
