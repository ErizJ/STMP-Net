import os

import torch
import math
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .clip import clip
from torch.nn import functional as F
from torch.nn import Dropout
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

number_map = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
              6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven',
              12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen',
              16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen',
              20: 'twenty', 30: 'thirty', 40: 'forty', 50: 'fifty',
              60: 'sixty', 70: 'seventy', 80: 'eighty', 90: 'ninety'}

scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']

dists_map = ['jpeg2000 compression', 'jpeg compression', 'noise', 'blur', 'color', 'contrast', 'overexposure',
            'underexposure', 'spatial', 'quantization', 'other']

# 纹理失真类型 (11类) - 局部/高频问题
texture_dists = ['none', 'uncertain', 'noise_amplification', 'ringing_halo', 'checkerboard', 
                 'moire', 'false_texture_hallucination', 'texture_smoothing', 'over_sharpening', 
                 'compression_blockiness', 'other_artifact']

# 结构失真类型 (6类) - 全局/几何问题
structure_dists = ['none', 'uncertain', 'edge_blur', 'detail_loss', 'geometric_distortion', 'aliasing_jaggies']


def get_number(i):
    if i in number_map:
        return number_map[i]
    else:
        tens = (i // 10) * 10
        ones = i % 10
        return number_map[tens]+"-"+number_map[ones]


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


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_featuress):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_featuress.argmax(dim=-1)] @ self.text_projection
        return x

class STMPNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.h = config.DATA.H_RESOLUTION // config.MODEL.VIT.PATCH_SIZE
        self.w = config.DATA.W_RESOLUTION // config.MODEL.VIT.PATCH_SIZE
        clip_model = load_clip_to_cpu(config, self.h, self.w)
        clip_model.to("cuda")
        # 强制将 CLIP 模型转为 float32，避免 JIT 加载路径下 half-precision
        # 在 Windows/某些 cuDNN 版本上触发 CUDNN_STATUS_INTERNAL_ERROR
        clip_model.float()
        # self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.global_features_learner = GlobalPromptLearner(config, config.num_scene, clip_model.dtype, clip_model.token_embedding)
        self.local_features_learner = LocalPromptLearner(config, config.num_dist, clip_model.dtype, clip_model.token_embedding)
        
        # Store number of classes for each branch
        self.num_scene = config.num_scene
        self.num_dist = config.num_dist
        self.num_texture = config.num_texture
        self.num_structure = config.num_structure
        
        # 三分支: 纹理和结构 Prompt Learner
        self.texture = getattr(config, 'texture', False)
        self.structure = getattr(config, 'structure', False)
        if self.texture:
            self.texture_features_learner = TexturePromptLearner(config, config.num_texture, clip_model.dtype, clip_model.token_embedding)
        if self.structure:
            self.structure_features_learner = StructurePromptLearner(config, config.num_structure, clip_model.dtype, clip_model.token_embedding)
        
        # self.tokenized_featuress = self.prompt_learner.tokenized_featuress
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype

        # self.logit_scale = clip_model.logit_scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        spatial_T = torch.tensor(3.0, dtype=self.dtype)  # 20
        self.spatial_logit_scale = nn.Parameter(spatial_T)
        
        # 纹理窗口池化 (3x3=9): 同时使用 Max 和 Avg 捕获更丰富的局部信息
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((3, 3))
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((3, 3))
        
        # 结构窗口池化 (2x2=4): 同时使用 Max 和 Avg
        self.adaptive_max_pool_structure = nn.AdaptiveMaxPool2d((2, 2))
        self.adaptive_avg_pool_structure = nn.AdaptiveAvgPool2d((2, 2))
        
        # Single-scale window 消融实验标记
        self.single_scale_window = getattr(config, 'single_scale_window', False)
        
        # Define dim before using it
        self.dim = 512
        
        # 投影层: 将拼接后的特征 (2*dim) 映射回原始维度 (dim)
        self.window_proj = nn.Linear(self.dim * 2, self.dim)
        self.structure_proj = nn.Linear(self.dim * 2, self.dim)

        # self.juery = 8
        bunch_layer = nn.TransformerDecoderLayer(
            d_model=512,
            dropout=0.0,
            nhead=8,
            activation=F.gelu,
            batch_first=True,
            dim_feedforward=(512 * 4),
            norm_first=True,
        )
        self.bunch_decoder = nn.TransformerDecoder(bunch_layer, num_layers=3)

        # self.bunch_embedding = nn.Parameter(torch.randn(1, 8, 512))
        # # self.heads = nn.Linear(512, 512, bias=False)
        # trunc_normal_(self.bunch_embedding, std=0.02)

        self.num_tokens = config.MODEL.NUM_TOKENS
        self.prompt_dropout = Dropout(config.MODEL.DROPOUT)
        #
        # # if project the prompt embeddings
        # # if self.prompt_config.PROJECT > -1:
        # #     # only for prepend / add
        # self.dim already defined above
        self.prompt_proj = nn.Linear(self.dim, 768)
        self.encoder_proj = nn.Linear(768, self.dim)
        nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal_(self.encoder_proj.weight, a=0, mode='fan_out')
        # # else:
        # #     self.dim = config.hidden_size
        # #     self.prompt_proj = nn.Identity()
        #
        # # initiate prompt:
        self.visual = config.visual
        if config.visual:
            patch_size = _pair(config.MODEL.VIT.PATCH_SIZE)
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + self.dim))  # noqa
            #
            # self.decoder_features_embeddings = nn.Parameter(torch.zeros(
            #     1, self.num_tokens, self.dim))
            # # xavier_uniform initialization
            # nn.init.uniform_(self.decoder_features_embeddings.data, -val, val)
            #
            # self.depth = config.DEPTH
            # self.deep_features_embeddings = nn.Parameter(torch.zeros(
            #     self.depth, self.num_tokens, self.dim))
            # # xavier_uniform initialization
            # nn.init.uniform_(self.deep_features_embeddings.data, -val, val)

            patch_size = _pair(config.MODEL.VIT.PATCH_SIZE)
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + self.dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, self.num_tokens, self.dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            self.depth = config.DEPTH
            self.deep_features_embeddings = nn.Parameter(torch.zeros(
                self.depth, self.num_tokens, self.dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.deep_features_embeddings.data, -val, val)

        # self.decoder_mlp = nn.Sequential(
        #     nn.Flatten(start_dim=1, end_dim=2),  # 展平[32,20,512]为[32,5120]
        #     nn.Linear(4096, 1024),  # 全连接层1
        #     nn.ReLU(),  # 激活函数
        #     nn.Linear(1024, 512),  # 全连接层2，输出为[32,512]
        # )
        self.decoder_mlp1 = nn.Sequential(
            nn.Linear(512, 256),  
            nn.ReLU(), 
            nn.Linear(256, 1), 
        )
        self.decoder_mlp2 = nn.Linear(20, 1)

        self.scene = config.scene
        self.dist = config.dist
        self.visual_only = getattr(config, "visual_only", False)
        self.visual_only_num_queries = int(getattr(config, "visual_only_num_queries", 20))
        if self.visual_only:
            # Pure visual baseline: learned visual queries instead of text prompts.
            self.visual_only_queries = nn.Parameter(torch.randn(1, self.visual_only_num_queries, self.dim))
        
        # 可学习的分支权重聚合
        # 固定为 4（最大分支数: scene+dist+texture+structure），保证 checkpoint 兼容性
        # 消融时按实际激活分支数切片使用
        self.branch_weights = nn.Parameter(torch.ones(4) / 4)
        self.branch_weight_activation = nn.Softmax(dim=0)

        # temperature = torch.tensor(3.91, dtype=self.dtype)  # 50
        # self.temperature = nn.Parameter(temperature)

    def _build_query(self, B):
        query_list = []
        if self.scene:
            query_list.append(self.get_scene_features())
        if self.dist:
            query_list.append(self.get_dist_features())
        if self.texture:
            query_list.append(self.get_texture_features())
        if self.structure:
            query_list.append(self.get_structure_features())

        if len(query_list) > 0:
            return torch.cat(query_list, dim=0).squeeze(dim=0).expand(B, -1, -1)
        if self.visual_only:
            return self.visual_only_queries.expand(B, -1, -1)
        raise ValueError("At least one branch (scene/dist/texture/structure) must be enabled, or set visual_only=True")

    def forward_deep_features(self, x):
        B = x.shape[0]
        x = self.image_encoder.get_embedding(x)
        if self.visual:
            embedding_output = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                x[:, 1:, :]
            ), dim=1)
        else:
            embedding_output = x

        hidden_states = self.image_encoder.ln_pre(embedding_output)

        if self.visual:
            for i in range(12):
                if i > 0:
                    deep_features_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_features_embeddings[i-1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_features_emb,
                        hidden_states[:, 1+self.num_tokens:, :]
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

    def invalidate_text_cache(self):
        """optimizer.step() 后调用，清除缓存，下次 forward 时重算一次。"""
        self._text_cache = {}

    def _get_cached_text_features(self, key, learner, label=None):
        if not hasattr(self, '_text_cache'):
            self._text_cache = {}
        if key not in self._text_cache:
            if label is None:
                prompts, tokenized_featuress = learner()
            else:
                prompts, tokenized_featuress = learner(label)
            with torch.no_grad():
                feats = self.text_encoder(prompts, tokenized_featuress)
            self._text_cache[key] = feats
        return self._text_cache[key]

    def get_scene_features(self, label=None):
        return self._get_cached_text_features('scene', self.global_features_learner, label)

    def get_dist_features(self, label=None):
        return self._get_cached_text_features('dist', self.local_features_learner, label)

    def get_texture_features(self, label=None):
        return self._get_cached_text_features('texture', self.texture_features_learner, label)

    def get_structure_features(self, label=None):
        return self._get_cached_text_features('structure', self.structure_features_learner, label)

    def get_image_features(self, x):
        B = x.shape[0]
        embedding = self.forward_deep_features(x)
        if self.visual:
            cls_features, patch_features = embedding[:, :1, :], embedding[:, 1+self.num_tokens:, :]
        else:
            cls_features, patch_features = embedding[:, :1, :], embedding[:, 1:, :]
        encoded_features = torch.cat((cls_features, patch_features), dim=1)
        square_num = 14
        patch_features_2d = patch_features.reshape(B, square_num, square_num, self.dim).permute(0, 3, 1, 2)
        
        # 细窗口池化 (3x3=9) 用于纹理/原有distortion
        # 同时使用 MaxPool 和 AvgPool 捕获更丰富的局部信息
        window_features_max = self.adaptive_max_pool(patch_features_2d).permute(0, 2, 3, 1)
        window_features_avg = self.adaptive_avg_pool(patch_features_2d).permute(0, 2, 3, 1)
        window_features_max = window_features_max.reshape(B, 9, self.dim)
        window_features_avg = window_features_avg.reshape(B, 9, self.dim)
        # 拼接 Max 和 Avg 特征，然后通过投影层映射回原始维度
        window_features_concat = torch.cat([window_features_max, window_features_avg], dim=-1)  # (B, 9, 2*dim)
        window_features = self.window_proj(window_features_concat)  # (B, 9, dim)
        
        # 粗窗口池化 (2x2=4) 用于结构失真
        # Single-scale window 消融: 退化为与细窗口相同的 3x3 尺度
        if self.single_scale_window:
            structure_window_features_max = self.adaptive_max_pool(patch_features_2d).permute(0, 2, 3, 1)
            structure_window_features_avg = self.adaptive_avg_pool(patch_features_2d).permute(0, 2, 3, 1)
            structure_window_features_max = structure_window_features_max.reshape(B, 9, self.dim)
            structure_window_features_avg = structure_window_features_avg.reshape(B, 9, self.dim)
            structure_window_features_concat = torch.cat([structure_window_features_max, structure_window_features_avg], dim=-1)
            # 用 window_proj 投影（同尺度），再通过 structure_proj 适配接口
            structure_window_features = self.window_proj(structure_window_features_concat)  # (B, 9, dim)
            # 为保持接口一致，聚合到 4 个 token（avg pooling over 9->4）
            structure_window_features = structure_window_features.mean(dim=1, keepdim=True).expand(B, 4, self.dim)
        else:
            structure_window_features_max = self.adaptive_max_pool_structure(patch_features_2d).permute(0, 2, 3, 1)
            structure_window_features_avg = self.adaptive_avg_pool_structure(patch_features_2d).permute(0, 2, 3, 1)
            structure_window_features_max = structure_window_features_max.reshape(B, 4, self.dim)
            structure_window_features_avg = structure_window_features_avg.reshape(B, 4, self.dim)
            structure_window_features_concat = torch.cat([structure_window_features_max, structure_window_features_avg], dim=-1)  # (B, 4, 2*dim)
            structure_window_features = self.structure_proj(structure_window_features_concat)  # (B, 4, dim)
        
        return encoded_features, cls_features, window_features, structure_window_features

    def forward(self, x, eval=False):
        B = x.shape[0]
        if eval:
            encoded_features, _, _, _ = self.get_image_features(x)
            query = self._build_query(B)
            decoded_features = self.bunch_decoder(query, encoded_features)
            temp_feat = decoded_features
            decoded_features = self.decoder_mlp1(decoded_features).squeeze(dim=-1)  # (B, total_classes)
            
            if self.visual_only and not (self.scene or self.dist or self.texture or self.structure):
                # Pure visual baseline: average over learned visual query responses.
                predict_score = decoded_features.mean(dim=1)
            else:
                # Split decoded_features back into branches and aggregate within each branch
                branch_scores = []
                start_idx = 0
                if self.scene:
                    end_idx = start_idx + self.num_scene
                    branch_scores.append(decoded_features[:, start_idx:end_idx].mean(dim=1, keepdim=True))
                    start_idx = end_idx
                if self.dist:
                    end_idx = start_idx + self.num_dist
                    branch_scores.append(decoded_features[:, start_idx:end_idx].mean(dim=1, keepdim=True))
                    start_idx = end_idx
                if self.texture:
                    end_idx = start_idx + self.num_texture
                    branch_scores.append(decoded_features[:, start_idx:end_idx].mean(dim=1, keepdim=True))
                    start_idx = end_idx
                if self.structure:
                    end_idx = start_idx + self.num_structure
                    branch_scores.append(decoded_features[:, start_idx:end_idx].mean(dim=1, keepdim=True))
                    start_idx = end_idx
                
                branch_scores = torch.cat(branch_scores, dim=1)  # (B, num_active_branches)
                
                # 使用可学习的加权求和替代简单均值
                num_active_branches = len(branch_scores[0])
                weights = self.branch_weight_activation(self.branch_weights[:num_active_branches])
                predict_score = torch.sum(branch_scores * weights.unsqueeze(0), dim=1)
            
            return predict_score, torch.mean(temp_feat, dim=1)
        else:
            encoded_features, cls_features, window_features, structure_window_features = self.get_image_features(x)
            cls_features = cls_features / cls_features.norm(dim=-1, keepdim=True)
            window_features = window_features / window_features.norm(dim=-1, keepdim=True)
            structure_window_features = structure_window_features / structure_window_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            spatial_logit_scale = self.spatial_logit_scale.exp()

            logits_global, logits_local, logits_texture, logits_structure = None, None, None, None
            if self.scene:
                global_features = self.get_scene_features()
                global_features = global_features / global_features.norm(dim=-1, keepdim=True)
                logits_global = logit_scale * cls_features @ global_features.t()
            
            if self.dist:
                local_features = self.get_dist_features()
                local_features = local_features / local_features.norm(dim=-1, keepdim=True)
                logits_ = logit_scale * window_features @ local_features.t()
                prob = F.softmax(logits_ * spatial_logit_scale, dim=1)
                logits_local = torch.sum(logits_ * prob, dim=1)

            if self.texture:
                texture_features = self.get_texture_features()
                texture_features = texture_features / texture_features.norm(dim=-1, keepdim=True)
                # 纹理用细窗口 (3x3)
                logits_tex = logit_scale * window_features @ texture_features.t()
                prob_tex = F.softmax(logits_tex * spatial_logit_scale, dim=1)
                logits_texture = torch.sum(logits_tex * prob_tex, dim=1)

            if self.structure:
                structure_features = self.get_structure_features()
                structure_features = structure_features / structure_features.norm(dim=-1, keepdim=True)
                # 结构用粗窗口 (2x2)
                logits_str = logit_scale * structure_window_features @ structure_features.t()
                prob_str = F.softmax(logits_str * spatial_logit_scale, dim=1)
                logits_structure = torch.sum(logits_str * prob_str, dim=1)
            query = self._build_query(B)
            
            decoded_features = self.bunch_decoder(query, encoded_features)
            decoded_features = self.decoder_mlp1(decoded_features).squeeze(dim=-1)  # (B, total_classes)
            
            if self.visual_only and not (self.scene or self.dist or self.texture or self.structure):
                predict_score = decoded_features.mean(dim=1)
            else:
                # Split decoded_features back into branches and aggregate within each branch
                branch_scores = []
                start_idx = 0
                if self.scene:
                    end_idx = start_idx + self.num_scene
                    branch_scores.append(decoded_features[:, start_idx:end_idx].mean(dim=1, keepdim=True))
                    start_idx = end_idx
                if self.dist:
                    end_idx = start_idx + self.num_dist
                    branch_scores.append(decoded_features[:, start_idx:end_idx].mean(dim=1, keepdim=True))
                    start_idx = end_idx
                if self.texture:
                    end_idx = start_idx + self.num_texture
                    branch_scores.append(decoded_features[:, start_idx:end_idx].mean(dim=1, keepdim=True))
                    start_idx = end_idx
                if self.structure:
                    end_idx = start_idx + self.num_structure
                    branch_scores.append(decoded_features[:, start_idx:end_idx].mean(dim=1, keepdim=True))
                    start_idx = end_idx
                
                branch_scores = torch.cat(branch_scores, dim=1)  # (B, num_active_branches)
                
                # 使用可学习的加权求和替代简单均值
                num_active_branches = len(branch_scores[0])
                weights = self.branch_weight_activation(self.branch_weights[:num_active_branches])
                predict_score = torch.sum(branch_scores * weights.unsqueeze(0), dim=1)

            # 返回格式: (score, scene_logits, dist_logits, texture_logits, structure_logits)
            return predict_score, \
                   logits_global.squeeze(dim=1) if logits_global is not None else None, \
                   logits_local if logits_local is not None else None, \
                   logits_texture if logits_texture is not None else None, \
                   logits_structure if logits_structure is not None else None


def load_clip_to_cpu(config, h, w):
    url = clip._MODELS[config.MODEL.BACKBONE]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict(), h, w)
    return model


# class PromptLearner(nn.Module):
#     def __init__(self, num_class, dtype, token_embedding):
#         super().__init__()
#         # ctx_init = "A photo of X X X X point."
#         ctx_init = "A photo with a quality score of X X X X."
#         ctx_dim = 512
#         # use given words to initialize context vectors
#         ctx_init = ctx_init.replace("_", " ")
#         n_ctx = 7
#
#         tokenized_featuress = clip.tokenize(ctx_init).cuda()
#         with torch.no_grad():
#             embedding = token_embedding(tokenized_featuress).type(dtype)
#         self.tokenized_featuress = tokenized_featuress  # torch.Tensor
#
#         n_cls_ctx = 4
#         cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
#         nn.init.normal_(cls_vectors, std=0.02)
#         self.cls_ctx = nn.Parameter(cls_vectors)
#
#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
#         self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
#         self.num_class = num_class
#         self.n_cls_ctx = n_cls_ctx
#
#     def forward(self, label):
#         cls_ctx = self.cls_ctx[label]
#         b = label.shape[0]
#         prefix = self.token_prefix.expand(b, -1, -1)
#         suffix = self.token_suffix.expand(b, -1, -1)
#
#         prompts = torch.cat(
#             [
#                 prefix,  # (n_cls, 1, dim)
#                 cls_ctx,  # (n_cls, n_ctx, dim)
#                 suffix,  # (n_cls, *, dim)
#             ],
#             dim=1,
#         )
#
#         return prompts

# scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']

# dists_map = ['jpeg2000 compression', 'jpeg compression', 'noise', 'blur', 'color', 'contrast', 'overexposure',
#             'underexposure', 'spatial', 'quantization', 'other']


class GlobalPromptLearner(nn.Module):
    def __init__(self, config, num_class, dtype, token_embedding):
        super().__init__()
        ctx_init = "a super-resolution image with"  # Use semantic prior for SR quality context
        ctx_dim = 512
        n_ctx = config.TRAIN.COOP_N_CTX
        self.n_cls = num_class
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = token_embedding(prompt.to('cuda')).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]  # (n_ctx, dim)
            # Expand to (num_class, n_ctx, dim) to match expected shape
            ctx_vectors = ctx_vectors.unsqueeze(0).expand(num_class, -1, -1).clone()
            prompt_prefix = ctx_init

        else:
            if config.TRAIN.COOP_CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(num_class, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).unsqueeze(0).expand(num_class, -1, -1).clone()
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [scenes[i] for i in range(num_class)]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_featuress = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = token_embedding(tokenized_featuress.to('cuda')).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_ctx = n_ctx
        self.tokenized_featuress = tokenized_featuress  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = config.TRAIN.COOP_CLASS_TOKEN_POSITION

    def forward(self, label=None):
        if label is None:
            ctx = self.ctx
            prefix = self.token_prefix
            suffix = self.token_suffix
            tokenized_featuress = self.tokenized_featuress
        else:
            # Keep batch dimension when indexing to maintain 3D shape
            ctx = self.ctx[label:label+1] if isinstance(label, int) else self.ctx[label]
            prefix = self.token_prefix[label:label+1] if isinstance(label, int) else self.token_prefix[label]
            suffix = self.token_suffix[label:label+1] if isinstance(label, int) else self.token_suffix[label]
            tokenized_featuress = self.tokenized_featuress[label:label+1] if isinstance(label, int) else self.tokenized_featuress[label]

        # Use actual batch size from the sliced tensors
        n_cls = ctx.shape[0]
        
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(n_cls):
                name_len = self.name_lens[i] if label is None else self.name_lens[label if isinstance(label, int) else label[i]]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(n_cls):
                name_len = self.name_lens[i] if label is None else self.name_lens[label if isinstance(label, int) else label[i]]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        # print(prompts.shape)
        return prompts, tokenized_featuress


class LocalPromptLearner(nn.Module):
    def __init__(self, config, num_class, dtype, token_embedding):
        super().__init__()
        ctx_init = "a super-resolution image with"  # Use semantic prior for SR distortion context
        ctx_dim = 512
        n_ctx = config.TRAIN.COOP_N_CTX
        self.n_cls = num_class
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = token_embedding(prompt.to('cuda')).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]  # (n_ctx, dim)
            # Expand to (num_class, n_ctx, dim) to match expected shape
            ctx_vectors = ctx_vectors.unsqueeze(0).expand(num_class, -1, -1).clone()
            prompt_prefix = ctx_init

        else:
            if config.TRAIN.COOP_CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(num_class, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).unsqueeze(0).expand(num_class, -1, -1).clone()
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [dists_map[i] for i in range(num_class)]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_featuress = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = token_embedding(tokenized_featuress.to('cuda')).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_ctx = n_ctx
        self.tokenized_featuress = tokenized_featuress  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = config.TRAIN.COOP_CLASS_TOKEN_POSITION

    def forward(self, label=None):
        if label is None:
            ctx = self.ctx
            prefix = self.token_prefix
            suffix = self.token_suffix
            tokenized_featuress = self.tokenized_featuress
        else:
            # Keep batch dimension when indexing to maintain 3D shape
            ctx = self.ctx[label:label+1] if isinstance(label, int) else self.ctx[label]
            prefix = self.token_prefix[label:label+1] if isinstance(label, int) else self.token_prefix[label]
            suffix = self.token_suffix[label:label+1] if isinstance(label, int) else self.token_suffix[label]
            tokenized_featuress = self.tokenized_featuress[label:label+1] if isinstance(label, int) else self.tokenized_featuress[label]

        # Use actual batch size from the sliced tensors
        n_cls = ctx.shape[0]
        
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(n_cls):
                name_len = self.name_lens[i] if label is None else self.name_lens[label if isinstance(label, int) else label[i]]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(n_cls):
                name_len = self.name_lens[i] if label is None else self.name_lens[label if isinstance(label, int) else label[i]]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        # print(prompts.shape)
        return prompts, tokenized_featuress



class TexturePromptLearner(nn.Module):
    """纹理失真 Prompt 学习器 - 用于局部纹理/高频失真分类"""
    def __init__(self, config, num_class, dtype, token_embedding):
        super().__init__()
        ctx_init = "a super-resolution image with texture"  # Use semantic prior for SR texture distortion
        ctx_dim = 512
        n_ctx = config.TRAIN.COOP_N_CTX
        self.n_cls = num_class
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = token_embedding(prompt.to('cuda')).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]  # (n_ctx, dim)
            # Expand to (num_class, n_ctx, dim) to match expected shape
            ctx_vectors = ctx_vectors.unsqueeze(0).expand(num_class, -1, -1).clone()
            prompt_prefix = ctx_init
        else:
            if config.TRAIN.COOP_CSC:
                print("Initializing class-specific contexts for texture")
                ctx_vectors = torch.empty(num_class, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context for texture")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).unsqueeze(0).expand(num_class, -1, -1).clone()
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Texture Initial context: "{prompt_prefix}"')
        print(f"Texture Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [texture_dists[i] for i in range(num_class)]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_featuress = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = token_embedding(tokenized_featuress.to('cuda')).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_ctx = n_ctx
        self.tokenized_featuress = tokenized_featuress
        self.name_lens = name_lens
        self.class_token_position = config.TRAIN.COOP_CLASS_TOKEN_POSITION

    def forward(self, label=None):
        if label is None:
            ctx = self.ctx
            prefix = self.token_prefix
            suffix = self.token_suffix
            tokenized_featuress = self.tokenized_featuress
        else:
            # Keep batch dimension when indexing to maintain 3D shape
            ctx = self.ctx[label:label+1] if isinstance(label, int) else self.ctx[label]
            prefix = self.token_prefix[label:label+1] if isinstance(label, int) else self.token_prefix[label]
            suffix = self.token_suffix[label:label+1] if isinstance(label, int) else self.token_suffix[label]
            tokenized_featuress = self.tokenized_featuress[label:label+1] if isinstance(label, int) else self.tokenized_featuress[label]

        # Use actual batch size from the sliced tensors
        n_cls = ctx.shape[0]
        
        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(n_cls):
                name_len = self.name_lens[i] if label is None else self.name_lens[label if isinstance(label, int) else label[i]]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts, tokenized_featuress


class StructurePromptLearner(nn.Module):
    """结构失真 Prompt 学习器 - 用于全局结构/几何失真分类"""
    def __init__(self, config, num_class, dtype, token_embedding):
        super().__init__()
        ctx_init = "a super-resolution image with structure"  # Use semantic prior for SR structure distortion
        ctx_dim = 512
        n_ctx = config.TRAIN.COOP_N_CTX
        self.n_cls = num_class
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = token_embedding(prompt.to('cuda')).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]  # (n_ctx, dim)
            # Expand to (num_class, n_ctx, dim) to match expected shape
            ctx_vectors = ctx_vectors.unsqueeze(0).expand(num_class, -1, -1).clone()
            prompt_prefix = ctx_init
        else:
            if config.TRAIN.COOP_CSC:
                print("Initializing class-specific contexts for structure")
                ctx_vectors = torch.empty(num_class, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context for structure")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).unsqueeze(0).expand(num_class, -1, -1).clone()
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Structure Initial context: "{prompt_prefix}"')
        print(f"Structure Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [structure_dists[i] for i in range(num_class)]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_featuress = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = token_embedding(tokenized_featuress.to('cuda')).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_ctx = n_ctx
        self.tokenized_featuress = tokenized_featuress
        self.name_lens = name_lens
        self.class_token_position = config.TRAIN.COOP_CLASS_TOKEN_POSITION

    def forward(self, label=None):
        if label is None:
            ctx = self.ctx
            prefix = self.token_prefix
            suffix = self.token_suffix
            tokenized_featuress = self.tokenized_featuress
        else:
            # Keep batch dimension when indexing to maintain 3D shape
            ctx = self.ctx[label:label+1] if isinstance(label, int) else self.ctx[label]
            prefix = self.token_prefix[label:label+1] if isinstance(label, int) else self.token_prefix[label]
            suffix = self.token_suffix[label:label+1] if isinstance(label, int) else self.token_suffix[label]
            tokenized_featuress = self.tokenized_featuress[label:label+1] if isinstance(label, int) else self.tokenized_featuress[label]

        # Use actual batch size from the sliced tensors
        n_cls = ctx.shape[0]
        
        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(n_cls):
                name_len = self.name_lens[i] if label is None else self.name_lens[label if isinstance(label, int) else label[i]]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts, tokenized_featuress
