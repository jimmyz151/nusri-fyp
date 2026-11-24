'''
Statistics of mask-derived detection boxes vs. ground-truth boxes.

Run this on a stage-1 checkpoint to measure the detection error distribution
that the training-time crop jitter should mimic.

Example:
    python src/tools/bbox_error_stats.py --cfg configs/SPNv2_BiFPNConvNeXt_v1_261119_s1.yaml \
        EXP_NAME SPNv2_BiFPNConvNeXt_v1_261119 TEST.TEST_CSV synthetic/labels/validation.csv
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import argparse
from os.path import join

import numpy as np
import pandas as pd
import torch

import _init_paths

from config import cfg, update_config
from nets   import build_spnv2
from dataset import get_dataloader
from utils.postprocess import mask_to_bbox
from utils.metrics     import bbox_iou
from utils.utils       import set_seeds_cudnn

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Mask-box error statistics')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    return parser.parse_args()


def _percentiles(values, qs=(50, 75, 90, 95, 99)):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {q: float('nan') for q in qs}
    return {q: float(np.percentile(values, q)) for q in qs}


def main(cfg):
    args = parse_args()
    update_config(cfg, args)

    # No two-stage refinement needed here
    cfg.defrost()
    cfg.TEST.CROP_REFINE = False
    cfg.freeze()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)-15s %(message)s')

    set_seeds_cudnn(cfg, seed=cfg.SEED)

    device = torch.device('cuda:0') if cfg.CUDA and torch.cuda.is_available() \
        else torch.device('cpu')

    # Model
    model = build_spnv2(cfg)

    model_file = join(cfg.OUTPUT_DIR, cfg.TEST.MODEL_FILE) if cfg.TEST.MODEL_FILE else ''
    if not os.path.exists(model_file) or os.path.isdir(model_file):
        model_file = join(cfg.OUTPUT_DIR, cfg.EXP_NAME, 'model_best.pth.tar')
    model.load_state_dict(torch.load(model_file, map_location='cpu'), strict=True)
    model = model.to(device)
    model.eval()
    logger.info(f'   - Model loaded from {model_file}')

    if 'segmentation' not in cfg.TEST.HEAD:
        raise ValueError('segmentation head must be in TEST.HEAD for mask-box statistics')

    loader = get_dataloader(cfg, split='test', load_labels=True)

    in_w, in_h = cfg.DATASET.INPUT_SIZE
    seg_idx = cfg.TEST.HEAD.index('segmentation')

    rows = []
    with torch.no_grad():
        for idx, (images, targets) in enumerate(loader):
            outputs = model(images, is_train=False, gpu=device)

            mask_pr = outputs[seg_idx].sigmoid().cpu().numpy()[0, 0]
            mask_bin = (mask_pr > cfg.TEST.MASK_THRESHOLD).astype(np.uint8)

            bbox_pr = mask_to_bbox(mask_bin)
            bbox_gt = targets['boundingbox'][0].numpy()

            name = targets['domain'][0]
            empty = bbox_pr is None

            if empty:
                rows.append({'domain': name, 'empty': 1, 'iou': 0.0,
                             'dx': np.nan, 'dy': np.nan,
                             'dw': np.nan, 'dh': np.nan,
                             'center': np.nan})
                continue

            mh, mw = mask_bin.shape
            bbox_pr = bbox_pr * np.array([in_w / mw, in_h / mh, in_w / mw, in_h / mh],
                                         dtype=np.float32)

            iou = float(np.squeeze(bbox_iou(bbox_pr, bbox_gt, x1y1x2y2=True)))

            # Errors normalized by the GT box side (mean of width & height)
            gt_w = max(float(bbox_gt[2] - bbox_gt[0]), 1e-6)
            gt_h = max(float(bbox_gt[3] - bbox_gt[1]), 1e-6)
            gt_side = 0.5 * (gt_w + gt_h)

            cpr = np.array([(bbox_pr[0] + bbox_pr[2]) / 2, (bbox_pr[1] + bbox_pr[3]) / 2])
            cgt = np.array([(bbox_gt[0] + bbox_gt[2]) / 2, (bbox_gt[1] + bbox_gt[3]) / 2])

            dx = float((cpr[0] - cgt[0]) / gt_side)
            dy = float((cpr[1] - cgt[1]) / gt_side)
            dw = float((bbox_pr[2] - bbox_pr[0]) / gt_w - 1.0)
            dh = float((bbox_pr[3] - bbox_pr[1]) / gt_h - 1.0)

            rows.append({'domain': name, 'empty': 0, 'iou': iou,
                         'dx': dx, 'dy': dy, 'dw': dw, 'dh': dh,
                         'center': float(np.hypot(dx, dy))})

            if (idx + 1) % 100 == 0:
                logger.info(f'   - Processed {idx + 1}/{len(loader)}')

    df = pd.DataFrame(rows)

    out_dir = join(cfg.ROOT, 'experiments', cfg.EXP_NAME)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = join(out_dir, 'bbox_error_stats.csv')
    df.to_csv(out_csv, index=False)

    valid = df[df['empty'] == 0]
    iou_p   = _percentiles(df['iou'])
    center  = _percentiles(valid['center'])
    absdx   = _percentiles(valid['dx'].abs())
    absdy   = _percentiles(valid['dy'].abs())
    absdw   = _percentiles(valid['dw'].abs())
    absdh   = _percentiles(valid['dh'].abs())

    logger.info('')
    logger.info(f'Samples: {len(df):,d}  |  empty masks: {int(df["empty"].sum()):,d} '
                f'({100.0 * df["empty"].mean():.2f}%)')
    logger.info('IoU        ' + '  '.join([f'p{q}: {iou_p[q]:.4f}' for q in iou_p]))
    logger.info('|center|   ' + '  '.join([f'p{q}: {center[q]:.4f}' for q in center]))
    logger.info('|dx|       ' + '  '.join([f'p{q}: {absdx[q]:.4f}' for q in absdx]))
    logger.info('|dy|       ' + '  '.join([f'p{q}: {absdy[q]:.4f}' for q in absdy]))
    logger.info('|dw|       ' + '  '.join([f'p{q}: {absdw[q]:.4f}' for q in absdw]))
    logger.info('|dh|       ' + '  '.join([f'p{q}: {absdh[q]:.4f}' for q in absdh]))

    # Conservative jitter envelope (see the experiment plan)
    shift = float(np.clip(np.ceil(center[95] / 0.05) * 0.05, 0.10, 0.30))
    side  = float(np.clip(np.ceil(max(absdw[95], absdh[95]) / 0.05) * 0.05, 0.15, 0.35))
    logger.info('')
    logger.info(f'Suggested CROP.JITTER_SHIFT: {shift:.2f}')
    logger.info(f'Suggested CROP.JITTER_SIDE : {side:.2f}')
    logger.info(f'Per-sample stats saved to {out_csv}')


if __name__ == '__main__':
    main(cfg)
