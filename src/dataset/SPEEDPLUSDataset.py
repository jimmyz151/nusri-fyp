'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import random
import numpy as np
import pandas as pd
import cv2
import logging
from tqdm import tqdm

import torch
from torchvision.transforms.functional import to_tensor

from utils.postprocess import quat2dcm
from utils.crop        import compute_crop_window, crop_with_pad, map_points_to_crop

logger = logging.getLogger(__name__)

class SPEEDPLUSDataset(torch.utils.data.Dataset):
    """ PyTorch Dataset class for SPEED+

    Args:
        cfg (dict): a dictionary for experiment config.
        split (string, optional): 'train', 'val', or 'test'. Defaults to 'train'.
        transforms (callable, optional): a set of Albumentations transformation functions for images.
        target_generators (callable, optional): a function to generate target labels.
    """
    def __init__(self,
                 cfg,
                 split='train',
                 transforms=None,
                 target_generators=None
    ):
        self.root        = join(cfg.DATASET.ROOT, cfg.DATASET.DATANAME)
        self.split       = split
        self.is_train    = split == 'train'

        self.image_size  = cfg.DATASET.IMAGE_SIZE # Original image size
        self.input_size  = cfg.DATASET.INPUT_SIZE # CNN input size
        self.output_size = [int(s / cfg.DATASET.OUTPUT_SIZE[0]) for s in self.input_size]
        self.num_keypts  = cfg.DATASET.NUM_KEYPOINTS

        # List of heads
        self.head_names  = cfg.MODEL.HEAD.LOSS_HEADS if self.is_train else cfg.TEST.HEAD
        self.load_masks  = True if 'segmentation' in self.head_names else False

        # Folder names
        # TODO: Make the naming automatic based on input image size or specify it at CFG
        self.imagefolder = 'images_768x512_RGB'
        self.maskfolder  = 'masks_192x128'
        self.stylefolder = 'styles_768x512_RGB'

        # Original-resolution folders (crop branches must never crop the resized copy)
        self.imagefolder_orig = 'images'
        self.maskfolder_orig  = 'masks'

        # Detection-error-simulated crop (training) and mask-box refinement (test)
        self.crop_cfg     = cfg.DATASET.CROP
        self.crop_refine  = cfg.TEST.CROP_REFINE

        # Load CSV & determine image domain
        self.csv, self.domain = self._load_csv(cfg, split)
        logger.info(f'   - Input size: {self.input_size[0]}x{self.input_size[1]}')

        # Texture (style) randomization
        # With probability `style_prob` the pre-generated stylized image
        # (`styles_768x512_RGB`) is used instead of the plain one. Those images are
        # produced during pre-processing, see tools/preprocess.py.
        if self.is_train and cfg.AUGMENT.APPLY_TEXTURE_RANDOMIZATION:
            self.styleAug   = True
            self.style_prob = cfg.AUGMENT.RANDOM_TEXTURE.PROB
            logger.info('   - Style augmentation ON (prob {})'.format(self.style_prob))
        else:
            self.styleAug   = False
            self.style_prob = 0.0

        # Image transforms
        self.transforms = transforms

        # Heatmap generator
        self.target_generators = target_generators
        self.load_labels = target_generators is not None

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        assert index < len(self), 'Index range error'

        #------------ Read all annotations
        anno = self._load_annotations(index) if self.load_labels else None

        #------------ Read image (+mask) with optional detection-error crop
        use_crop = self.is_train and self.crop_cfg.ENABLE \
            and random.random() < self.crop_cfg.PROB

        if use_crop:
            # IMPORTANT: crop from the ORIGINAL image, not the 768x512 copy
            image = self._load_image(index, folder=self.imagefolder_orig)
            mask  = self._load_mask(index, folder=self.maskfolder_orig) if self.load_masks else None

            window = self._sample_crop_window(anno['boundingbox'])
            image, mask, anno = self._apply_crop(image, mask, anno, window)

            mask_target = None
            if mask is not None:
                mask_target = torch.from_numpy(mask).float().div_(255.0).unsqueeze(0)
        else:
            image = self._load_image(index)
            mask_target = None
            if self.load_labels and self.load_masks:
                mask_target = to_tensor(self._load_mask(index))

        #------------ Image data transform
        if self.load_labels:
            transform_kwargs = {'image': image,
                                'bboxes': [anno['boundingbox']],
                                'class_labels': ['tango']}
        else:
            transform_kwargs = {'image': image}

        if self.transforms is not None:
            transformed = self.transforms(**transform_kwargs)

            # Clean up
            image = transformed['image']
            if self.load_labels:
                anno['boundingbox'] = np.array(transformed['bboxes'][0], dtype=np.float32)

        # Return just transformed image if not returning labels
        if not self.load_labels:
            return image

        # Bounding box in [0, 1] -> convert to pixels
        anno['boundingbox'] *= np.array(
            [self.input_size[0], self.input_size[1], self.input_size[0], self.input_size[1]],
            dtype=np.float32
        )

        #------------ Generate targets
        targets = {'domain':         anno['domain'],
                   'boundingbox':    torch.from_numpy(anno['boundingbox']),
                   'quaternion':     torch.from_numpy(anno['quaternion']),
                   'rotationmatrix': torch.from_numpy(anno['rotationmatrix']),
                   'translation':    torch.from_numpy(anno['translation'])}

        # Extra target for the segmentation head (required for both training and validation)
        if self.load_masks:
            targets['mask'] = mask_target

        if self.is_train:
            for i, h in enumerate(self.head_names):
                if h == 'heatmap':
                    heatmap = self.target_generators[i](anno['keypoints']).astype(np.float32)
                    targets['heatmap'] = torch.from_numpy(heatmap)
                elif h == 'efficientpose' or h == 'segmentation':
                    pass
                else:
                    raise ValueError(f'{h} is not a valid head name')

        #---------- Original image for the two-stage refinement at test time
        if (not self.is_train) and self.split == 'test' and self.crop_refine:
            image_orig = self._load_image(index, folder=self.imagefolder_orig)
            image_orig = np.ascontiguousarray(image_orig.transpose(2, 0, 1))
            targets['image_orig'] = torch.from_numpy(image_orig)

        return image, targets

    def _sample_crop_window(self, bbox_norm):
        """ Sample a 3:2 crop window around the GT box with detection-like errors """
        W, H = self.image_size
        bbox = bbox_norm * np.array([W, H, W, H], dtype=np.float32)

        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        bw = max(float(bbox[2] - bbox[0]), 1.0)
        bh = max(float(bbox[3] - bbox[1]), 1.0)

        if random.random() < self.crop_cfg.SEVERE_PROB:
            # Long-tail detection failures: large offset, wrong scale, partially out of frame
            angle = random.uniform(0, 2 * np.pi)
            mag   = random.uniform(0.25, self.crop_cfg.SEVERE_SHIFT)
            shift = (mag * np.cos(angle), mag * np.sin(angle))
            scale = random.choice(self.crop_cfg.SEVERE_SCALE)
        else:
            shift = (random.uniform(-self.crop_cfg.JITTER_SHIFT, self.crop_cfg.JITTER_SHIFT),
                     random.uniform(-self.crop_cfg.JITTER_SHIFT, self.crop_cfg.JITTER_SHIFT))
            scale = random.uniform(1.0 - self.crop_cfg.JITTER_SIDE,
                                   1.0 + self.crop_cfg.JITTER_SIDE)

        return compute_crop_window(cx, cy, bw, bh, self.image_size,
                                   expand=self.crop_cfg.EXPAND,
                                   shift=shift, scale=scale,
                                   min_side=self.crop_cfg.MIN_SIDE)

    def _apply_crop(self, image, mask, anno, window):
        """ Crop original image/mask and remap keypoints & bbox into the crop frame """
        W, H   = self.image_size
        iw, ih = self.input_size
        ow, oh = self.output_size

        # Image: original -> crop -> network input
        image = crop_with_pad(image, window)
        image = cv2.resize(image, (iw, ih), interpolation=cv2.INTER_LINEAR)

        # Mask: same window -> head output resolution
        if mask is not None:
            mask = crop_with_pad(mask, window)
            mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
            mask = np.asarray(mask)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask[mask >  128] = 255
            mask[mask <= 128] = 0

        # Keypoints: normalized -> original px -> crop-normalized
        keypts = anno['keypoints'] * np.array([W, H], dtype=np.float32)
        keypts = map_points_to_crop(keypts, window)
        anno['keypoints'] = keypts.astype(np.float32)

        # Bounding box: normalized -> original px -> crop-normalized
        bbox = anno['boundingbox'] * np.array([W, H, W, H], dtype=np.float32)
        bbox = map_points_to_crop(bbox.reshape(2, 2), window).reshape(4)
        anno['boundingbox'] = np.clip(bbox, 0.0, 1.0).astype(np.float32)

        return image, mask, anno

    def _load_csv(self, cfg, split='train'):
        """ Load CSV content into pandas.DataFrame """

        if split == 'train':
            csvfile, mode = cfg.TRAIN.TRAIN_CSV, 'Training  '
        elif split == 'val':
            csvfile, mode = cfg.TRAIN.VAL_CSV,   'Validating'
        elif split == 'test':
            csvfile, mode = cfg.TEST.TEST_CSV,   'Testing   '
        else:
            raise AssertionError('split must be either train, val or test')

        logger.info(f'{mode} on {csvfile}')

        # Current domain
        if 'speedplus' in cfg.DATASET.DATANAME:
            domain = csvfile.split('/')[0]
        elif cfg.DATASET.DATANAME == 'prisma25':
            domain = ''
        elif 'shirt' in cfg.DATASET.DATANAME:
            splits = csvfile.split('/')
            domain = splits[0] + '/' + splits[1]
        else:
            raise AssertionError('Only speedplus and prisma25 datasets are supported')

        # Read CSV file to pandas
        csv = pd.read_csv(join(self.root, csvfile), header=None)

        return csv, domain

    def _load_image(self, index, folder=None):
        """ Read image of given index from a folder, if specified """

        # Use the stylized version with probability `style_prob`
        if folder is None:
            folder = self.stylefolder if self.styleAug and random.random() < self.style_prob \
                            else self.imagefolder

        imgpath = join(self.root, self.domain, folder, self.csv.iloc[index, 0])
        image   = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(
                f'Could not read image "{imgpath}". Check DATASET.ROOT/DATANAME and that '
                f'"{folder}" has been pre-processed.'
            )

        # NOTE: SPEED+ images are stored as grayscale, so no BGR->RGB swap is needed here.
        return image

    def _load_mask(self, index, folder=None):
        """ Read mask image """

        if folder is None:
            folder = self.maskfolder

        imgpath = join(self.root, self.domain, folder, self.csv.iloc[index, 0])
        data    = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        if data is None:
            raise FileNotFoundError(
                f'Could not read mask "{imgpath}". Check DATASET.ROOT/DATANAME and that '
                f'"{self.maskfolder}" has been pre-processed.'
            )

        # Clean up any intermediate values
        data[data >  128] = 255
        data[data <= 128] = 0

        return data[:,:,None]

    def _load_annotations(self, index):
        """ Load labels into a dictionary """

        # Bounding box (xmin, ymin, xmax, ymax) (normalized)
        bbox = np.array(self.csv.iloc[index, 1:5], dtype=np.float32)

        # Clip box within image
        bbox[[0,2]] = np.clip(bbox[[0,2]], 0, 1)
        bbox[[1,3]] = np.clip(bbox[[1,3]], 0, 1)

        # Target origin [2,]
        origin = np.array(self.csv.iloc[index, 12:14], dtype=np.float32)
        origin = np.reshape(origin, (2,))

        # Keypoints (normalized) [N, 2]
        keypts = np.array(self.csv.iloc[index, 14:], dtype=np.float32)
        keypts = np.reshape(keypts, (self.num_keypts, 2)) # (N, 2)

        # Ground-truth pose
        q_gt = np.array(self.csv.iloc[index, 5:9],  dtype=np.float32) # [qw, qx, qy, qz]
        t_gt = np.array(self.csv.iloc[index, 9:12], dtype=np.float32)

        annotations = {'imgpath': self.csv.iloc[index, 0], # Image file name, e.g., 'img000001.jpg'
                       'domain': self.domain,              # Image domain, e.g., 'lightbox' of SPEED+
                       'boundingbox': bbox,
                       'origin': origin,
                       'keypoints': keypts,
                       'quaternion': q_gt,
                       'rotationmatrix': quat2dcm(q_gt),   # 3x3 rotation matrix of ground-truth orientation
                       'translation': t_gt}

        return annotations
