'''
ConvNeXt-T + BiFPN backbone (v2).

ConvNeXt-T 的各级输出 C2~C5（stride 4/8/16/32）先经过各自独立的 LayerNorm，
再由 BiFPNLayer 内部的 1x1 通道投影送入 BiFPN；通道投影与 BiFPN 内部均使用 GroupNorm。
骨干原有的 LayerNorm 全部保留，新增的输出 LayerNorm 只作用于送往 BiFPN 的分支。
BiFPN 拓扑与 SPNv2/EfficientDet 一致：输入 [P1, C2, C3, C4, C5]，由 C5 下采样生成 P6/P7，
输出 P2~P7。
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from .bifpn import BiFPNLayer

logger = logging.getLogger(__name__)


class LayerNorm2d(nn.Module):
    """ 对 [B, C, H, W] 特征图做逐通道 LayerNorm。 """
    def __init__(self, num_channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ConvNeXtBiFPN(nn.Module):
    """ ConvNeXt-T 各级输出先做独立 LayerNorm，再进入 GN 版 BiFPN。 """
    def __init__(self, cfg, scaled_parameters):
        super(ConvNeXtBiFPN, self).__init__()

        # ImageNet 预训练 ConvNeXt-T
        pretrained = cfg.MODEL.BACKBONE.PRETRAINED

        weights, local_path = None, None

        if pretrained:
            if isinstance(pretrained, str) and pretrained.lower() != 'imagenet':
                if not os.path.isfile(pretrained):
                    raise FileNotFoundError(
                        f'MODEL.BACKBONE.PRETRAINED file not found: {pretrained}'
                    )
                local_path = pretrained
            else:
                weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1

        model = convnext_tiny(weights=weights)

        if local_path is not None:
            state_dict = torch.load(local_path, map_location='cpu')
            state_dict = state_dict.get('state_dict', state_dict)
            model.load_state_dict(state_dict, strict=True)
            logger.info(f'   - ConvNeXt-T ImageNet weights loaded from {local_path}')

        # 只保留特征提取部分，丢弃 avgpool / classifier
        self.features = model.features

        # C2~C5 通道数（stride 4/8/16/32）
        self.feat_channels = [96, 192, 384, 768]

        # 各级输出独立的 LayerNorm，只作用于送往 BiFPN 的分支
        self.use_output_ln = cfg.MODEL.BACKBONE.USE_OUTPUT_LN
        if self.use_output_ln:
            self.input_norms = nn.ModuleList([
                LayerNorm2d(c) for c in self.feat_channels
            ])
        else:
            self.input_norms = nn.ModuleList()

        # BiFPN：拓扑不变，1x1 通道投影与内部 separable conv 的归一化由配置控制
        self.use_p2 = True
        bifpn_width = scaled_parameters['bifpn_width']
        self.bifpn = nn.Sequential(*[
            BiFPNLayer([None] + self.feat_channels,
                       bifpn_width,
                       first_time=True if _ == 0 else False,
                       attention=True,
                       use_p2=self.use_p2,
                       use_group_norm=cfg.MODEL.USE_GROUPNORM_BACKBONE,
                       group_norm_size=cfg.MODEL.GROUPNORM_SIZE)
            for _ in range(scaled_parameters['bifpn_depth'])
        ])

        self.out_channels = [bifpn_width] * 6
        self.out_strides = [4, 8, 16, 32, 64, 128]
        self.head_in_channels = bifpn_width

    def forward(self, x):
        feats = []
        for i, blk in enumerate(self.features):
            x = blk(x)
            if i in (1, 3, 5, 7):
                feats.append(x)

        # 输出 LN 只在送进 BiFPN 的分支上，骨干内部 LN 不变
        if self.use_output_ln:
            feats = [norm(f) for norm, f in zip(self.input_norms, feats)]

        return self.bifpn([None] + feats)
