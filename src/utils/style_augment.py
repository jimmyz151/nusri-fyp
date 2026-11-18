"""在线风格随机化（online style augmentation）。

口径对齐 SPEED+ baseline（Park et al., 2021）：
- alpha 控制随机风格与原风格的插值比例，官方取 0.5；
- 每次 forward 重新采样随机风格嵌入，因此同一图像在每个 epoch 看到的纹理都不同；
- alpha = 0 时的基准是 SPEED+ 训练集的平均风格嵌入，而不是单图自身的内容风格。

与 tools/preprocess.py 预生成 `styles_<W>x<H>_RGB` 的离线方案相比，这里风格数量随
迭代次数增长，不会被固定成每图一张。

默认接口兼容 exp4 的 ImageNet 归一化输入；normalized=False 接受并返回 [0, 1]
浮点图像，用于先风格化、后像素增强、最后归一化的训练流程。
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import styleaug
from styleaug.ghiasi import Ghiasi

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# SPEED+ 训练集上的平均风格嵌入，由 speedplusbaseline 的
# get_embedding_mean_and_covariance.py 统计得到（见仓库 _prereq）。
BASE_EMBEDDING = Path(__file__).resolve().parent / 'models' / 'embedding_mean_speedplus.npy'


class OnlineStyleAugmentor(nn.Module):
    """ 对已归一化的图像 batch 做在线风格随机化，返回同形状、同统计量的 tensor """

    def __init__(self, alpha=0.5, device='cuda'):
        super().__init__()
        if not BASE_EMBEDDING.is_file():
            raise FileNotFoundError(f'找不到基准风格嵌入: {BASE_EMBEDDING}')

        self.alpha = float(alpha)
        self.device = torch.device(device)

        self.ghiasi = Ghiasi().to(self.device).eval()
        for param in self.ghiasi.parameters():
            param.requires_grad_(False)

        checkpoint_dir = Path(styleaug.__file__).resolve().parent / 'checkpoints'
        transformer = torch.load(checkpoint_dir / 'checkpoint_transformer.pth',
                                map_location='cpu', weights_only=False)
        self.ghiasi.load_state_dict(transformer['state_dict_ghiasi'], strict=False)

        embeddings = torch.load(checkpoint_dir / 'checkpoint_embeddings.pth',
                                map_location='cpu', weights_only=False)
        self.register_buffer('style_mean',
                             embeddings['pbn_embedding_mean'].float().reshape(1, -1).to(self.device))
        # 随机风格嵌入的采样矩阵：x ~ N(0, 1) -> x @ proj.T + style_mean
        u, s, _ = np.linalg.svd(embeddings['pbn_embedding_covariance'].float().numpy())
        proj = (u * np.sqrt(s)).astype(np.float32)
        self.register_buffer('style_proj', torch.from_numpy(proj).to(self.device))
        self.register_buffer('base_embedding',
                             torch.from_numpy(np.load(BASE_EMBEDDING).astype(np.float32).reshape(1, -1))
                             .to(self.device))
        self.register_buffer('norm_mean',
                             torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
                             .to(self.device))
        self.register_buffer('norm_std',
                             torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
                             .to(self.device))

    @torch.no_grad()
    def forward(self, images, normalized=True):
        dtype = images.dtype
        x = images.to(self.device).float()
        if normalized:
            x = x * self.norm_std + self.norm_mean
        x = x.clamp(0.0, 1.0)

        # 每次调用都为每张图像独立采样风格。
        style = torch.randn(x.size(0), self.style_mean.size(1),
                            device=self.device, dtype=torch.float32)
        style = style @ self.style_proj.transpose(0, 1) + self.style_mean
        style = self.alpha * style + (1.0 - self.alpha) * self.base_embedding

        styled = self.ghiasi(x, style)
        if normalized:
            styled = (styled - self.norm_mean) / self.norm_std
        return styled.to(dtype)
