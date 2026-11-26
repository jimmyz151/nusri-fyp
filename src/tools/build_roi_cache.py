'''
Offline ROI-cache builder for the seg-driven two-stage pipeline.

Runs the S1 full-frame model on a CSV split and stores the segmentation-driven
ROI box (largest connected component) in original-image pixels, together with
box-quality diagnostics against the GT mask.  The S2 crop dataset consumes this
cache; no GT box is used anywhere in the pipeline.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import os.path as osp

import numpy as np
import torch

import _init_paths

from config import cfg, update_config
from nets import build_spnv2
from dataset import get_dataloader
from utils.metrics import segment_iou
from utils.roi import mask_to_bbox, box_quality
from utils.utils import set_seeds_cudnn


def parse_args():
    parser = argparse.ArgumentParser(description='Build seg-driven ROI cache')

    parser.add_argument('--cfg', required=True, type=str,
                        help='S1 full-frame config')
    parser.add_argument('--csv', required=True, type=str,
                        help='Split CSV, e.g. synthetic/labels/train.csv')
    parser.add_argument('--out', required=True, type=str,
                        help='Output cache CSV')
    parser.add_argument('--model', default='', type=str,
                        help='Override S1 checkpoint (default: checkpoints/<EXP>/model_best.pth.tar)')
    parser.add_argument('--batch', default=8, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--min-area', default=32, type=int)
    parser.add_argument('--limit', default=0, type=int,
                        help='Optional sample limit for smoke tests')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    return parser.parse_args()


def main():
    args = parse_args()
    update_config(cfg, args)

    cfg.defrost()
    cfg.TEST.TEST_CSV = args.csv
    cfg.TEST.IMAGES_PER_GPU = args.batch
    cfg.DATASET.ROI.ENABLE = False
    cfg.freeze()

    assert 'segmentation' in cfg.TEST.HEAD, \
        'S1 TEST.HEAD must include segmentation'

    set_seeds_cudnn(cfg, seed=cfg.SEED)

    device = torch.device('cuda:0') if cfg.CUDA and torch.cuda.is_available() \
        else torch.device('cpu')

    model_path = args.model or cfg.TEST.MODEL_FILE
    if not model_path or not osp.exists(model_path) or osp.isdir(model_path):
        model_path = osp.join(cfg.OUTPUT_DIR, cfg.EXP_NAME, 'model_best.pth.tar')

    model = build_spnv2(cfg)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    model = model.to(device)
    model.eval()

    print(f'Build ROI cache: {args.csv}')
    print(f'   model: {model_path}')
    print(f'   split: {cfg.TEST.TEST_CSV}  (limit={args.limit or "all"})')

    loader = get_dataloader(cfg, split='test', load_labels=True)
    seg_index = cfg.TEST.HEAD.index('segmentation')
    mask_w = mask_h = None

    fields = [
        'image', 'valid',
        'x1', 'y1', 'x2', 'y2', 'side', 'mask_area',
        'iou', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2',
        'center_err', 'side_ratio', 'n_out', 'allin',
    ]
    os.makedirs(osp.dirname(args.out) or '.', exist_ok=True)

    written = 0
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        with torch.no_grad():
            for images, targets in loader:
                outputs = model(images, is_train=False, gpu=device)
                seg = outputs[seg_index].sigmoid().cpu().numpy()[:, 0]
                gt_mask = targets['mask'].numpy()[:, 0]
                keypoints = targets['keypoints'].numpy()
                images_names = targets['imgpath']

                if mask_w is None:
                    mask_h, mask_w = seg.shape[-2:]
                    scale = np.array([
                        cfg.DATASET.IMAGE_SIZE[0] / mask_w,
                        cfg.DATASET.IMAGE_SIZE[1] / mask_h,
                    ], dtype=np.float64)

                for b in range(seg.shape[0]):
                    mask = seg[b]
                    box, area = mask_to_bbox(
                        mask, threshold=args.threshold, min_area=args.min_area)
                    gt_box, _ = mask_to_bbox(gt_mask[b] > 0.5, threshold=0.5)

                    row = {
                        'image': str(images_names[b]),
                        'valid': int(box is not None),
                        'x1': '', 'y1': '', 'x2': '', 'y2': '',
                        'side': '', 'mask_area': int(area),
                        'iou': '', 'gt_x1': '', 'gt_y1': '',
                        'gt_x2': '', 'gt_y2': '',
                        'center_err': '', 'side_ratio': '',
                        'n_out': '', 'allin': '',
                    }

                    row['iou'] = float(segment_iou(
                        (mask > args.threshold).astype(np.float64),
                        (gt_mask[b] > 0.5).astype(np.float64)))

                    gt_box_orig = None
                    if gt_box is not None:
                        gt_box_orig = gt_box * np.tile(scale, 2)
                        row.update({
                            'gt_x1': float(gt_box_orig[0]),
                            'gt_y1': float(gt_box_orig[1]),
                            'gt_x2': float(gt_box_orig[2]),
                            'gt_y2': float(gt_box_orig[3]),
                        })

                    if box is not None:
                        box_orig = box * np.tile(scale, 2)
                        row.update({
                            'x1': float(box_orig[0]),
                            'y1': float(box_orig[1]),
                            'x2': float(box_orig[2]),
                            'y2': float(box_orig[3]),
                            'side': float(max(box_orig[2] - box_orig[0],
                                              box_orig[3] - box_orig[1])),
                        })

                        if gt_box_orig is not None:
                            kp_px = keypoints[b] * np.array(
                                [cfg.DATASET.IMAGE_SIZE[0],
                                 cfg.DATASET.IMAGE_SIZE[1]], dtype=np.float64)
                            quality = box_quality(box_orig, gt_box_orig, kp_px)
                            row.update({
                                'center_err': quality['center_err'],
                                'side_ratio': quality['side_ratio'],
                                'n_out': quality['n_out'],
                                'allin': int(quality['allin']),
                            })

                    writer.writerow(row)
                    written += 1

                    if args.limit and written >= args.limit:
                        break

                if written % 200 == 0:
                    print(f'   {written:,d} samples written')

                if args.limit and written >= args.limit:
                    break

    print(f'Wrote {written:,d} rows to {args.out}')


if __name__ == '__main__':
    main()
