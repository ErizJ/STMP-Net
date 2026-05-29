"""
消融实验共享常量定义
"""

ABLATION_CONFIGS = {
    'full_model':            {},
    'visual_only':           {'scene': False, 'dist': False, 'texture': False, 'structure': False, 'visual_only': True},
    'wo_scene_prompt':       {'scene': False},
    'wo_distortion_prompt':  {'dist': False},
    'wo_texture_prompt':     {'texture': False},
    'wo_structure_prompt':   {'structure': False},
    'single_scale_window':   {'single_scale_window': True},
    'wo_fidelity_loss':      {'DELTA': 0.0},
    'wo_texture_loss':       {'ALPHA': 0.0},
    'wo_structure_loss':     {'BETA': 0.0},
}

DISPLAY_NAMES = {
    'full_model':            'Ours (MM-Prompt)',
    'visual_only':           'Visual-only',
    'wo_scene_prompt':       'w/o Scene Prompt',
    'wo_distortion_prompt':  'w/o Distortion Prompt',
    'wo_texture_prompt':     'w/o Texture Prompt',
    'wo_structure_prompt':   'w/o Structure Prompt',
    'single_scale_window':   'Single-scale Window',
    'wo_fidelity_loss':      'w/o Fidelity Loss',
    'wo_texture_loss':       'w/o Texture Loss',
    'wo_structure_loss':     'w/o Structure Loss',
}

DATASET_DEFAULTS = {
    'waterloo15': 'configs/Pure/vit_small_pre_coder_waterloo15.yaml',
    'cviu17':     'configs/Pure/vit_small_pre_coder_cviu17.yaml',
    'qads':       'configs/Pure/vit_small_pre_coder_qads.yaml',
}

ABLATION_ORDER = list(ABLATION_CONFIGS.keys())
