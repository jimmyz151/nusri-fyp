"""Apply online style before the existing CPU pixel augmentations.

风格化必须发生在像素增强之前，因此在主进程用 GPU 完成风格化；像素增强仍交给
独立 worker 进程并行执行，并预取一个 batch，使其与 GPU 前向/反向重叠。

调度上只做重叠，不改增强顺序与算子：
- 风格化在独立 CUDA stream 的后台线程中执行，与当前 batch 的训练计算重叠；
- 风格化输入从 dataset 的 pinned uint8 直接异步 H2D（等价换算），结果异步写回
  pinned 缓冲后再交给像素增强 worker，主线程不再等待 GPU 同步点；
- 像素增强 worker 直接接收 CHW，转置在 worker 内并行完成；
- 汇总结果写入 pinned 输出，模型原有的 `to(gpu, non_blocking=True)` 才能真异步。
"""

import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

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
    # image 为 CHW uint8；转置放在 worker 内，避免主线程为整批再拷一份 HWC。
    image = np.ascontiguousarray(image.transpose(1, 2, 0))
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
        self.device = style_augmentor.device
        # 风格化独占一条 stream，使它的 GPU 计算能与默认 stream 上的训练重叠。
        self.stream = torch.cuda.Stream(device=self.device)
        # 父进程已初始化 CUDA，只能使用 spawn 创建子进程
        self.pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context('spawn'),
            initializer=_init_worker,
            initargs=(cfg,),
        )
        # 单线程负责一次（至多一个）batch 的风格化 + 提交像素增强，形成预取流水线。
        self.prepare_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='style-prefetch')
        # 预热：spawn 子进程要重新导入 torch（每个约 10s），若留到第一批提交时
        # 才创建，第一个 iteration 会被拖到几十秒。这里在训练开始前一次性付掉。
        list(self.pool.map(_noop, range(workers)))

    @torch.no_grad()
    def _style_to_uint8(self, images):
        # dataset 送来的 batch 是 pinned uint8；H2D 不阻塞主线程，uint8 -> float
        # 与 /255 放到 GPU 上完成（数值与 CPU float32 换算一致）。
        with torch.cuda.stream(self.stream):
            x = images.to(self.device, non_blocking=True).float().div_(255.0)
            styled = self.style_augmentor(x, normalized=False)
            styled = styled.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
            # D2H 到 pinned 缓冲；只在后台线程里同步，不阻塞训练主线程。
            pinned = torch.empty(styled.shape, dtype=torch.uint8, pin_memory=True)
            pinned.copy_(styled, non_blocking=True)
            self.stream.synchronize()
        return pinned

    def _prepare(self, images, targets, use_style):
        if images.dtype != torch.uint8 or images.ndim != 4:
            raise ValueError('Style-first training requires uint8 BCHW images from the loader')

        # 保留 exp4 的整批开关，只改变增强顺序
        source = self._style_to_uint8(images) if use_style else images

        array = source.numpy()  # CHW uint8；worker 内再转 HWC
        height, width = array.shape[2:4]
        scale = np.array([width, height, width, height], dtype=np.float32)
        boxes = targets['boundingbox'].detach().cpu().numpy()

        futures = [
            self.pool.submit(_augment_one, image, box, scale)
            for image, box in zip(array, boxes)
        ]
        return futures, targets

    def _submit(self, images, targets):
        # 概率判定留在主线程，保证每个 batch 仍只消费一次 random()，顺序与旧实现一致。
        return self.prepare_pool.submit(
            self._prepare, images, targets, random.random() < self.probability)

    @staticmethod
    def _collect(futures, targets):
        results = [future.result() for future in futures]
        targets = dict(targets)
        targets['boundingbox'] = torch.from_numpy(np.stack([box for _, box in results]))

        # 汇总到 pinned 输出，供模型 forward 中现有的 non_blocking H2D 真正异步。
        images = torch.empty((len(results),) + results[0][0].shape,
                             dtype=results[0][0].dtype, pin_memory=True)
        for i, (image, _) in enumerate(results):
            images[i].copy_(image)
        return images, targets

    def wrap(self, loader):
        """ 预取一个 batch：当前 batch 的风格化/像素增强与上一个 batch 的 GPU 计算重叠 """
        return _PrefetchedLoader(self, loader)

    def _iterate(self, loader):
        iterator = iter(loader)
        try:
            images, targets = next(iterator)
        except StopIteration:
            return

        # 第一批同步准备；此后每批都在 yield 当前结果前提交，使下一批的风格化
        # 与 CPU 像素增强在训练当前批时并行进行。
        pending = self._submit(images, targets)
        for images, targets in iterator:
            futures, job_targets = pending.result()
            pending = self._submit(images, targets)
            yield self._collect(futures, job_targets)

        futures, job_targets = pending.result()
        yield self._collect(futures, job_targets)

    def close(self):
        self.prepare_pool.shutdown(wait=False)
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
