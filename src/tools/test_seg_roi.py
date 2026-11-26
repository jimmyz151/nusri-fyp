'''
Two-stage seg-driven ROI evaluation + mask-box sensitivity sweep.

Pipeline per image:
    S1 full frame (heatmap + segmentation)
        -> mask threshold -> largest connected component -> ROI box
        -> expand k -> square crop from the original 1920x1200 image
        -> S2 crop model (heatmap) -> inverse affine -> EPnP
    optional fallback to the S1 full-frame pose

Test domains provide GT bboxes, but they may not align with the mask.  GT
bboxes are never used as the ROI prior here: the ROI comes only from the
predicted mask.  GT mask boxes are used only for diagnostics.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import os.path as osp

import cv2
import numpy as np
import pandas as pd
import torch

import _init_paths

from config import cfg as default_cfg, update_config
from nets import build_spnv2
from dataset import get_dataloader
from utils.metrics import error_orientation, error_translation, speed_score, segment_iou
from utils.postprocess import solve_pose_from_heatmaps
from utils.roi import mask_to_bbox, square_expand_box, crop_resize, box_quality
from utils.utils import (
    load_camera_intrinsics, load_tango_3d_keypoints, set_seeds_cudnn,
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description='Seg-driven ROI two-stage test')

    parser.add_argument('--cfg', required=True, type=str,
                        help='S2 crop-model config')
    parser.add_argument('--s1-cfg', default='', type=str,
                        help='S1 full-frame config (default: TEST.ROI.S1_CFG)')
    parser.add_argument('--s1-model', default='', type=str,
                        help='S1 checkpoint override')
    parser.add_argument('--s2-model', default='', type=str,
                        help='S2 checkpoint override')
    parser.add_argument('--csv', default='', type=str,
                        help='Test CSV (default: TEST.TEST_CSV)')
    parser.add_argument('--out-dir', default='', type=str)
    parser.add_argument('--k-list', default='', type=str,
                        help='Comma-separated expand factors; overrides TEST.ROI.K_LIST')
    parser.add_argument('--limit', default=0, type=int)

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    return parser.parse_args()


def load_cfg(path, opts=None):
    cfg = default_cfg.clone()
    args = argparse.Namespace(cfg=path, opts=list(opts or []))
    update_config(cfg, args)
    return cfg


def build_model(cfg, exp_name, device, path_override=''):
    path = path_override or osp.join(
        cfg.OUTPUT_DIR, exp_name, 'model_best.pth.tar')
    if not osp.exists(path):
        raise FileNotFoundError(path)

    cfg = cfg.clone()
    cfg.defrost()
    cfg.MODEL.PRETRAIN_FILE = None
    cfg.freeze()

    model = build_spnv2(cfg)
    model.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
    model = model.to(device)
    model.eval()
    print(f'   - Model loaded: {path}')
    return model, path


def score_pose(q_pr, t_pr, reject, q_gt, t_gt, domain, cfg):
    out = {
        'valid': 0, 'reject': int(bool(reject)),
        'eR': np.nan, 'eT': np.nan, 'HIL_pose': np.nan,
    }
    if reject or q_pr is None or t_pr is None:
        return out

    out['valid'] = 1
    out['eR'] = float(error_orientation(q_pr, q_gt, 'quaternion'))
    out['eT'] = float(error_translation(t_pr, t_gt))
    _, _, pose = speed_score(
        t_pr, q_pr, t_gt, q_gt,
        representation='quaternion',
        applyThreshold=domain in ['lightbox', 'sunlamp'],
        theta_q=cfg.TEST.SPEED_THRESHOLD_Q,
        theta_t=cfg.TEST.SPEED_THRESHOLD_T,
    )
    out['HIL_pose'] = float(pose)
    return out


def edge_hit(keypts, box, affine, margin_frac):
    """Detected keypoints within `margin_frac` of the crop border.

    Undetected keypoints are left at (0, 0) by the decoder and become the crop
    origin after the affine, so they are excluded explicitly.
    """
    if keypts is None:
        return False
    sx, sy, ox, oy = affine
    kp = np.asarray(keypts, dtype=np.float64)
    detected = ~(np.isclose(kp[:, 0], ox) & np.isclose(kp[:, 1], oy))
    if not detected.any():
        return False

    side = box[2] - box[0]
    margin = float(margin_frac) * side
    x, y = kp[detected, 0], kp[detected, 1]
    return bool(np.any((x < box[0] + margin) | (x > box[2] - margin) |
                       (y < box[1] + margin) | (y > box[3] - margin)))


def summarize(df, k_values):
    rows = []

    s1 = df.drop_duplicates('image')
    if len(s1):
        rows.append(_summary_row(s1, '-', 's1_fullframe', s1['s1_valid']))

    for k in k_values:
        sub = df[df['k'] == k]
        if not len(sub):
            continue
        rows.append(_summary_row(sub, k, 's2_only', sub['s2_valid']))
        rows.append(_summary_row(sub, k, 'fallback', sub['fb_valid']))

    return pd.DataFrame(rows)


def _summary_row(sub, k, mode, valid):
    valid = valid.astype(bool)
    if mode == 's1_fullframe':
        pose = sub['s1_HIL_pose'][valid].to_numpy(dtype=np.float64)
        eR = sub['s1_eR'][valid].to_numpy(dtype=np.float64)
        eT = sub['s1_eT'][valid].to_numpy(dtype=np.float64)
        reject = sub['s1_reject'].sum()
        fallback = 0.0
    elif mode == 'fallback':
        pose = sub['fb_HIL_pose'][valid].to_numpy(dtype=np.float64)
        eR = sub['fb_eR'][valid].to_numpy(dtype=np.float64)
        eT = sub['fb_eT'][valid].to_numpy(dtype=np.float64)
        reject = sub['fb_valid'].eq(0).sum()
        fallback = sub['fb_used'].mean() if len(sub) else np.nan
    else:
        pose = sub['s2_HIL_pose'][valid].to_numpy(dtype=np.float64)
        eR = sub['s2_eR'][valid].to_numpy(dtype=np.float64)
        eT = sub['s2_eT'][valid].to_numpy(dtype=np.float64)
        reject = sub['s2_valid'].eq(0).sum()
        fallback = 0.0

    n = len(sub)
    row = {
        'config': k,
        'mode': mode,
        'n': n,
        'valid': int(valid.sum()),
        'reject': int(reject),
        'roi_valid_pct': float(sub['roi_valid'].mean() * 100) if 'roi_valid' in sub else np.nan,
        'fallback_pct': float(fallback * 100),
        'eR_mean': _nanmean(eR),
        'eR_median': _nanmedian(eR),
        'eT_mean': _nanmean(eT),
        'HIL_pose_mean': _nanmean(pose),
        'HIL_pose_median': _nanmedian(pose),
        'HIL_pose_p90': _nanpercentile(pose, 90),
        'HIL_pose_p99': _nanpercentile(pose, 99),
        'fail_eR10_pct': float(np.mean(eR > 10.0) * 100) if len(eR) else np.nan,
        'fail_pose03_pct': float(np.mean(pose > 0.3) * 100) if len(pose) else np.nan,
        'allin_pct': float(sub['allin'].mean() * 100) if 'allin' in sub else np.nan,
        'mean_n_out': _nanmean(sub['n_out'].to_numpy(dtype=np.float64)),
        'mean_center_err': _nanmean(sub['center_err'].to_numpy(dtype=np.float64)),
        'mean_side_ratio': _nanmean(sub['side_ratio'].to_numpy(dtype=np.float64)),
    }
    return row


def _nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def _nanmedian(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan


def _nanpercentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanpercentile(x, q)) if np.isfinite(x).any() else np.nan


def main():
    args = parse_args()
    cfg = load_cfg(args.cfg, args.opts)

    s1_cfg_path = args.s1_cfg or cfg.TEST.ROI.S1_CFG
    if not s1_cfg_path:
        raise ValueError('Provide --s1-cfg or set TEST.ROI.S1_CFG')
    s1_cfg = load_cfg(s1_cfg_path)

    csv_path = args.csv or cfg.TEST.TEST_CSV
    s1_cfg.defrost()
    s1_cfg.TEST.TEST_CSV = csv_path
    s1_cfg.TEST.IMAGES_PER_GPU = 1
    s1_cfg.freeze()

    k_values = [float(v) for v in
                (args.k_list.split(',') if args.k_list else cfg.TEST.ROI.K_LIST)]

    set_seeds_cudnn(cfg, seed=cfg.SEED)
    device = torch.device('cuda:0') if cfg.CUDA and torch.cuda.is_available() \
        else torch.device('cpu')

    s1_exp = cfg.TEST.ROI.S1_EXP_NAME or s1_cfg.EXP_NAME
    s2_exp = cfg.TEST.ROI.S2_EXP_NAME or cfg.EXP_NAME
    s1_model, _ = build_model(s1_cfg, s1_exp, device, args.s1_model)
    s2_model, _ = build_model(cfg, s2_exp, device, args.s2_model)

    s1_heat_idx = s1_cfg.TEST.HEAD.index('heatmap')
    s1_seg_idx = s1_cfg.TEST.HEAD.index('segmentation')
    s2_heat_idx = cfg.TEST.HEAD.index('heatmap')

    camera = load_camera_intrinsics(s1_cfg.DATASET.CAMERA)
    keypts_3d = load_tango_3d_keypoints(s1_cfg.DATASET.KEYPOINTS)
    loader = get_dataloader(s1_cfg, split='test', load_labels=True)

    rows = []
    processed = 0

    with torch.no_grad():
        for images, targets in loader:
            image_name = str(targets['imgpath'][0])
            sample_domain = str(targets['domain'][0])
            q_gt = targets['quaternion'][0].numpy()
            t_gt = targets['translation'][0].numpy()

            # ---------------- S1: full-frame mask + pose ----------------
            outputs = s1_model(images, is_train=False, gpu=device)
            heat_s1 = outputs[s1_heat_idx].squeeze(0).cpu()
            seg_s1 = outputs[s1_seg_idx].sigmoid().cpu().numpy()[0, 0]

            kp_s1, q_s1, t_s1, reject_s1 = solve_pose_from_heatmaps(
                heat_s1,
                s1_cfg.DATASET.IMAGE_SIZE,
                s1_cfg.TEST.HEATMAP_THRESHOLD,
                camera, keypts_3d,
                udp=s1_cfg.DATASET.UDP,
            )
            s1_score = score_pose(q_s1, t_s1, reject_s1, q_gt, t_gt,
                                  sample_domain, cfg)

            gt_mask = targets['mask'][0, 0].numpy()
            mask_iou = segment_iou(
                (seg_s1 > cfg.TEST.ROI.MASK_THRESHOLD).astype(np.float64),
                (gt_mask > 0.5).astype(np.float64),
            )
            gt_box, _ = mask_to_bbox(gt_mask > 0.5, threshold=0.5)
            keypoints_gt = targets['keypoints'][0].numpy() * np.array(
                [s1_cfg.DATASET.IMAGE_SIZE[0], s1_cfg.DATASET.IMAGE_SIZE[1]],
                dtype=np.float64)

            mask_h, mask_w = seg_s1.shape
            scale = np.array([
                s1_cfg.DATASET.IMAGE_SIZE[0] / mask_w,
                s1_cfg.DATASET.IMAGE_SIZE[1] / mask_h,
            ], dtype=np.float64)
            mask_box, _ = mask_to_bbox(
                seg_s1,
                threshold=cfg.TEST.ROI.MASK_THRESHOLD,
                min_area=cfg.TEST.ROI.MIN_MASK_PX,
            )
            if mask_box is not None:
                mask_box = mask_box * np.tile(scale, 2)
            if gt_box is not None:
                gt_box = gt_box * np.tile(scale, 2)

            original = cv2.imread(
                osp.join(s1_cfg.DATASET.ROOT, s1_cfg.DATASET.DATANAME,
                         sample_domain, 'images', image_name),
                cv2.IMREAD_COLOR)
            if original is None:
                raise FileNotFoundError(image_name)

            # ---------------- S2 at each expand factor ----------------
            crops, crop_boxes = [], []
            if mask_box is not None:
                for k in k_values:
                    box = square_expand_box(
                        mask_box, k, cfg.TEST.ROI.MIN_SIDE)
                    crop, affine = crop_resize(
                        original, box, cfg.DATASET.INPUT_SIZE)
                    crop = crop.astype(np.float32) / 255.0
                    crop = (crop - MEAN) / STD
                    crops.append(crop.transpose(2, 0, 1))
                    crop_boxes.append((box, affine))

            s2_scores = [None] * len(k_values)
            s2_edges = [False] * len(k_values)
            if crops:
                batch = torch.from_numpy(
                    np.stack(crops)).float().to(device)
                s2_outputs = s2_model(batch, is_train=False, gpu=device)
                heat_batch = s2_outputs[s2_heat_idx].cpu()

                for j, (box, affine) in enumerate(crop_boxes):
                    kp_s2, q_s2, t_s2, reject_s2 = solve_pose_from_heatmaps(
                        heat_batch[j],
                        cfg.DATASET.INPUT_SIZE,
                        cfg.TEST.HEATMAP_THRESHOLD,
                        camera, keypts_3d,
                        udp=cfg.DATASET.UDP,
                        roi_affine=affine,
                    )
                    s2_scores[j] = score_pose(
                        q_s2, t_s2, reject_s2, q_gt, t_gt, sample_domain, cfg)
                    s2_edges[j] = edge_hit(
                        kp_s2, box, affine, cfg.TEST.ROI.EDGE_MARGIN)

            # ---------------- Record ----------------
            for j, k in enumerate(k_values):
                s2 = s2_scores[j] or {
                    'valid': 0, 'reject': 1,
                    'eR': np.nan, 'eT': np.nan, 'HIL_pose': np.nan,
                }
                edge = s2_edges[j]

                fb_used = s1_score['valid'] and (
                    not s2['valid'] or edge)
                final = s1_score if fb_used else s2

                box, affine = crop_boxes[j] if j < len(crop_boxes) else (
                    np.full(4, np.nan), (np.nan,) * 4)
                quality = box_quality(box, gt_box, keypoints_gt) \
                    if (np.isfinite(box).all() and gt_box is not None) else {
                        'center_err': np.nan, 'side_ratio': np.nan,
                        'n_out': np.nan, 'allin': np.nan,
                }

                rows.append({
                    'image': image_name,
                    'domain': sample_domain,
                    'k': k,
                    'roi_valid': int(mask_box is not None),
                    'mask_iou': mask_iou,
                    'box_x1': box[0], 'box_y1': box[1],
                    'box_x2': box[2], 'box_y2': box[3],
                    'center_err': quality['center_err'],
                    'side_ratio': quality['side_ratio'],
                    'n_out': quality['n_out'],
                    'allin': quality['allin'],
                    's1_valid': s1_score['valid'],
                    's1_reject': s1_score['reject'],
                    's1_eR': s1_score['eR'],
                    's1_eT': s1_score['eT'],
                    's1_HIL_pose': s1_score['HIL_pose'],
                    's2_valid': s2['valid'],
                    's2_reject': s2['reject'],
                    's2_eR': s2['eR'],
                    's2_eT': s2['eT'],
                    's2_HIL_pose': s2['HIL_pose'],
                    's2_edge': int(edge),
                    'fb_used': int(bool(fb_used)),
                    'fb_valid': final['valid'],
                    'fb_eR': final['eR'],
                    'fb_eT': final['eT'],
                    'fb_HIL_pose': final['HIL_pose'],
                })

            processed += 1
            if processed % 100 == 0:
                print(f'   {processed:,d} samples')
            if args.limit and processed >= args.limit:
                break

    df = pd.DataFrame(rows)
    summary = summarize(df, k_values)

    out_dir = args.out_dir or osp.join(
        cfg.RESULT_DIR, 'tables', cfg.EXP_NAME, 'probes', 'roi_sensitivity')
    os.makedirs(out_dir, exist_ok=True)

    prefix = (str(df['domain'].iloc[0]) if len(df) else 'test').replace('/', '_')
    per_sample = osp.join(out_dir, f'{prefix}_per_sample.csv')
    summary_path = osp.join(out_dir, f'{prefix}_summary.csv')
    df.to_csv(per_sample, index=False)
    summary.to_csv(summary_path, index=False)

    print(f'Wrote {per_sample} ({len(df):,d} rows)')
    print(f'Wrote {summary_path}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
