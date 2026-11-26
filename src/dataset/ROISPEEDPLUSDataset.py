'''
ROI-cropped SPEED+ dataset for the seg-driven two-stage pipeline.

Training samples use the S1 mask box cached by
`src/tools/build_roi_cache.py`, with a random expand factor in
[DATASET.ROI.K_MIN, DATASET.ROI.K_MAX].  Validation uses a deterministic
expand factor.  Crops are taken from the original 1920x1200 images, not from
the pre-resized 768x512 copies, so the object resolution matches the GTBBox
experiment.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import random
import logging

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms.functional import to_tensor

from utils.roi import (
    mask_to_bbox, square_expand_box, shift_box, crop_resize,
    mask_crop_resize, points_to_crop, bbox_to_crop, center_crop_box,
)
from .SPEEDPLUSDataset import SPEEDPLUSDataset

logger = logging.getLogger(__name__)


class ROISPEEDPLUSDataset(SPEEDPLUSDataset):
    """SPEED+ samples cropped by S1 segmentation-driven ROI boxes."""

    def __init__(self, cfg, split='train', transforms=None, target_generators=None):
        super(ROISPEEDPLUSDataset, self).__init__(
            cfg, split, transforms, target_generators)

        if not cfg.DATASET.ROI.ENABLE:
            raise ValueError('ROISPEEDPLUSDataset requires DATASET.ROI.ENABLE')

        self.roi = cfg.DATASET.ROI
        self.imagefolder = self.roi.IMAGE_DIR
        self.maskfolder = self.roi.MASK_DIR

        self.roi_cache = self._load_roi_cache(split)

    # ------------------------------------------------------------------
    # Cache / raw files
    # ------------------------------------------------------------------
    def _load_roi_cache(self, split):
        path = {
            'train': self.roi.TRAIN_CACHE,
            'val': self.roi.VAL_CACHE,
            'test': self.roi.TEST_CACHE,
        }.get(split)

        if not path:
            raise ValueError(
                f'DATASET.ROI.{split.upper()}_CACHE is empty; '
                'run src/tools/build_roi_cache.py first')

        df = pd.read_csv(path)
        required = {'image', 'valid', 'x1', 'y1', 'x2', 'y2'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'{path} is missing columns: {sorted(missing)}')

        cache = {}
        for row in df.itertuples(index=False):
            values = np.array([row.x1, row.y1, row.x2, row.y2], dtype=np.float64)
            if int(row.valid) == 0 or not np.all(np.isfinite(values)):
                cache[str(row.image)] = None
            else:
                cache[str(row.image)] = values

        logger.info('   - ROI cache: {} ({:,d} entries, {:,d} valid)'.format(
            path, len(cache), sum(v is not None for v in cache.values())))
        return cache

    def _load_image(self, index, folder=None):
        folder = self.imagefolder if folder is None else folder
        imgpath = join(self.root, self.domain, folder, self.csv.iloc[index, 0])
        image = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'Could not read image "{imgpath}"')
        return image

    def _load_mask(self, index):
        imgpath = join(self.root, self.domain, self.maskfolder,
                       self.csv.iloc[index, 0])
        data = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        if data is None:
            raise FileNotFoundError(f'Could not read mask "{imgpath}"')
        data[data > 128] = 255
        data[data <= 128] = 0
        return data

    def _roi_box(self, index):
        """Cached S1 box, with GT-mask / center-crop fallbacks for rare misses."""
        fname = str(self.csv.iloc[index, 0])
        box = self.roi_cache.get(fname)
        if box is not None:
            return box

        mask = self._load_mask(index)
        box, _ = mask_to_bbox(mask, threshold=0, min_area=1)
        if box is not None:
            return box

        return center_crop_box(self.image_size)

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        assert index < len(self), 'Index range error'

        image = self._load_image(index)
        raw_box = self._roi_box(index)
        width, height = self.image_size

        if self.is_train:
            k = random.uniform(float(self.roi.K_MIN), float(self.roi.K_MAX))
            box = square_expand_box(raw_box, k, self.roi.MIN_SIDE)

            jitter = float(self.roi.CENTER_JITTER) * (box[2] - box[0])
            if jitter > 0:
                box = shift_box(box,
                                random.uniform(-jitter, jitter),
                                random.uniform(-jitter, jitter))
        else:
            box = square_expand_box(raw_box, self.roi.VAL_K, self.roi.MIN_SIDE)

        image_crop, affine = crop_resize(image, box, self.input_size)

        if not self.load_labels:
            if self.transforms is not None:
                image_crop = self.transforms(image=image_crop)['image']
            return image_crop

        anno = self._load_annotations(index)
        mask = self._load_mask(index)
        mask = mask_crop_resize(mask, box, self.output_size)
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)[:, :, None]

        keypoints_px = anno['keypoints'] * np.array([width, height], dtype=np.float32)
        bbox_px = anno['boundingbox'] * np.array(
            [width, height, width, height], dtype=np.float32)

        anno['keypoints'] = points_to_crop(
            keypoints_px, affine, self.input_size).astype(np.float32)
        anno['boundingbox'] = np.clip(
            bbox_to_crop(bbox_px, affine, self.input_size),
            0.0, 1.0).astype(np.float32)
        anno['mask'] = to_tensor(mask)

        if self.transforms is not None:
            transformed = self.transforms(
                image=image_crop,
                bboxes=[anno['boundingbox']],
                class_labels=['tango'],
            )
            image_crop = transformed['image']
            # The bbox is unused by heatmap/segmentation losses; Albumentations
            # may drop it when the ROI excludes the annotated box entirely.
            if len(transformed['bboxes']) == 0:
                anno['boundingbox'] = np.array(
                    [0.0, 0.0, 1.0, 1.0], dtype=np.float32)
            else:
                anno['boundingbox'] = np.array(
                    transformed['bboxes'][0], dtype=np.float32)

        anno['boundingbox'] *= np.array(
            [self.input_size[0], self.input_size[1],
             self.input_size[0], self.input_size[1]], dtype=np.float32)

        targets = {
            'domain':         anno['domain'],
            'imgpath':        anno['imgpath'],
            'boundingbox':    torch.from_numpy(anno['boundingbox']),
            'quaternion':     torch.from_numpy(anno['quaternion']),
            'rotationmatrix': torch.from_numpy(anno['rotationmatrix']),
            'translation':    torch.from_numpy(anno['translation']),
            'keypoints':      torch.from_numpy(anno['keypoints']),
            'mask':           anno['mask'],
            'roi_affine':     torch.tensor(affine, dtype=torch.float32),
        }

        if self.is_train:
            for i, h in enumerate(self.head_names):
                if h == 'heatmap':
                    heatmap = self.target_generators[i](anno['keypoints'])
                    targets['heatmap'] = torch.from_numpy(
                        heatmap.astype(np.float32))

        return image_crop, targets
