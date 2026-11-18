"""Apply online style before the existing CPU pixel augmentations.

风格化必须发生在像素增强之前，因此在主进程用 GPU 完成风格化；像素增强仍交给
独立 worker 进程并行执行，并预取一个 batch，使其与 GPU 前向/反向重叠。
"""

import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from dataset.transforms import build_transforms

_WORKER_TRANSFORMS = None


def _init_worker(cfg):
    global _WORKER_TRANSFORMS
    # 每个增强子进程只用一个线程：否则 OpenCV/torch 会在每个进程内开满线程，
    # 与 DataLoader worker、其他 rank 严重超订，反而拖慢整体吞吐。
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    _WORKER_TRANSFORMS = build_transforms(cfg, is_train=True, defer_online=False)


def _augment_one(image, box, scale):
    transformed = _WORKER_TRANSFORMS(
        image=image,
        bboxes=[box / scale],
        class_labels=['tango'],
    )
    return transformed['image'], np.asarray(transformed['bboxes'][0], dtype=np.float32) * scale


def _noop(_token):
    return None


class StyleBeforeAugment:
    """ 在线风格化 -> 像素增强 -> 归一化，像素增强在子进程中并行完成 """

    def __init__(self, cfg, style_augmentor, workers=4):
        self.style_augmentor = style_augmentor
        self.probability = cfg.AUGMENT.RANDOM_TEXTURE.PROB
        # 父进程已初始化 CUDA，只能使用 spawn 创建子进程
        self.pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context('spawn'),
            initializer=_init_worker,
            initargs=(cfg,),
        )
        # 预热：spawn 子进程要重新导入 torch（每个约 10s），若留到第一批提交时
        # 才创建，第一个 iteration 会被拖到几十秒。这里在训练开始前一次性付掉。
        list(self.pool.map(_noop, range(workers)))

    def _prepare(self, images, targets):
        if images.dtype != torch.uint8 or images.ndim != 4:
            raise ValueError('Style-first training requires uint8 BCHW images from the loader')

        # 保留 exp4 的整批开关，只改变增强顺序
        if random.random() < self.probability:
            images = self.style_augmentor(images.float() / 255.0, normalized=False)
            # 还原成官方像素增强所需的 uint8 输入（风格化在 GPU 上完成，需搬回 CPU）
            images = images.clamp(0, 1).mul(255).round().to(torch.uint8).cpu()

        array = images.permute(0, 2, 3, 1).contiguous().numpy()
        height, width = array.shape[1:3]
        scale = np.array([width, height, width, height], dtype=np.float32)
        boxes = targets['boundingbox'].detach().cpu().numpy()

        futures = [
            self.pool.submit(_augment_one, np.ascontiguousarray(image), box, scale)
            for image, box in zip(array, boxes)
        ]
        return futures, targets

    @staticmethod
    def _collect(futures, targets):
        results = [future.result() for future in futures]
        targets = dict(targets)
        targets['boundingbox'] = torch.from_numpy(np.stack([box for _, box in results]))
        return torch.stack([image for image, _ in results]), targets

    def wrap(self, loader):
        """ 预取一个 batch：当前 batch 的像素增强与上一个 batch 的 GPU 计算重叠 """
        return _PrefetchedLoader(self, loader)

    def _iterate(self, loader):
        pending = None
        for images, targets in loader:
            current = self._prepare(images, targets)
            if pending is not None:
                yield self._collect(*pending)
            pending = current
        if pending is not None:
            yield self._collect(*pending)

    def close(self):
        self.pool.shutdown(wait=False)


class _PrefetchedLoader:
    """ 保留 len()，训练循环仍按 batch 数计算进度 """

    def __init__(self, pipeline, loader):
        self.pipeline = pipeline
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return self.pipeline._iterate(self.loader)
