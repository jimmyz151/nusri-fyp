'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from scipy.io import savemat

import torch

from utils.utils       import AverageMeter, ProgressMeter
from utils.postprocess import solve_pose_from_heatmaps, rot_6d_to_matrix, mask_to_bbox
from utils.crop        import compute_crop_window, crop_to_input
from utils.metrics import *
from utils.visualize import *

logger = logging.getLogger("Testing")

# Must match dataset/transforms/build.py
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _get_metric_unit(name):
    unit = ''
    if 'eR' in name:
        unit = 'deg'
    elif 'eT' in name:
        unit = 'm'

    return unit

def do_valid(epoch, cfg, model, data_loader, camera, keypts_true_3D,
                valid_fraction=None, log_dir=None, device=torch.device('cpu'),
                allow_refine=False):

    batch_time = AverageMeter('', 'ms', ':6.1f')
    data_time  = AverageMeter('', 'ms', ':6.1f')

    # Two-stage refinement needs the heatmap, the mask detector and the original image
    refine_enabled = allow_refine and cfg.TEST.CROP_REFINE \
        and 'heatmap' in cfg.TEST.HEAD and 'segmentation' in cfg.TEST.HEAD

    metric_names = []
    if 'heatmap' in cfg.TEST.HEAD:
        metric_names += ['heat_eR', 'heat_eT', 'heat_eR_HIL', 'heat_eT_norm_HIL', 'heat_pose']
        if refine_enabled:
            metric_names += ['ref_eR', 'ref_eT', 'ref_eR_HIL', 'ref_eT_norm_HIL', 'ref_pose']
    if 'efficientpose' in cfg.TEST.HEAD:
        metric_names += ['effi_iou', 'effi_eR', 'effi_eT', 'effi_eR_HIL', 'effi_eT_norm_HIL', 'effi_pose']
    if 'segmentation' in cfg.TEST.HEAD:
        metric_names += ['segm_iou', 'segm_det_iou']

    metric_names += ['final_pose']

    metrics = {}
    per_sample_metrics = {}
    for name in metric_names:
        metrics[name] = AverageMeter(name, _get_metric_unit(name), ':6.2f')
        per_sample_metrics[name] = []

    # Predictions
    # heatmaps = []  # 注释：全量测试时累积会占 ~7GB 内存导致 OOM，不再缓存
    h_reject = []
    q_hmap   = []
    t_hmap   = []
    R_reg    = []
    t_reg    = []
    bbox_reg = []
    mask_bbox = []
    q_ref    = []
    t_ref    = []
    ref_reject = []

    # Check validation set fraction
    num_batches = len(data_loader)
    if valid_fraction:
        assert valid_fraction <= 1 and valid_fraction > 0, \
            'valid_fraction must be within (0, 1]'
        num_batches = int(len(data_loader) * valid_fraction)

    progress = ProgressMeter(
        num_batches,
        batch_time,
        [v for k, v in metrics.items()
            if k in (['heat_eR', 'effi_eT', 'final_pose'] + (['ref_eR'] if refine_enabled else []))],
        prefix="Testing {:03d} ".format(epoch+1))

    # switch to eval mode
    model.eval()

    # Loop through dataloader
    end = time.time()
    for idx, (images, targets) in enumerate(data_loader):
        start = time.time()
        data_time.update((start - end)*1000)

        assert images.shape[0] == 1, 'Use batch size = 1 for testing'

        # Ground-truth
        domain    = targets['domain'][0]
        q_gt      = targets['quaternion'][0].numpy()
        R_gt      = targets['rotationmatrix'][0].numpy()
        t_gt      = targets['translation'][0].numpy()
        bbox_gt   = targets['boundingbox'][0].numpy()
        if 'segmentation' in cfg.TEST.HEAD:
            mask_gt = targets['mask'][0].numpy()

        with torch.no_grad():
            # Forward pass
            outputs = model(images,
                            is_train=False,
                            gpu=device)

            # Debug (uncomment)
            # imshow(images[0])
            # imshowheatmap(images[0], heatmap[0], 8)

            # --- Mask head doubles as the detector: foreground -> bbox
            mask_pr_bin, bbox_mask = None, None
            if 'segmentation' in cfg.TEST.HEAD:
                seg_idx = cfg.TEST.HEAD.index('segmentation')
                mask_pr = outputs[seg_idx].sigmoid().cpu().numpy()  # [1,1,H,W]
                mask_pr[mask_pr > cfg.TEST.MASK_THRESHOLD]  = 1
                mask_pr[mask_pr <= cfg.TEST.MASK_THRESHOLD] = 0
                mask_pr_bin = mask_pr[0].astype(np.uint8)          # [1,H,W]

                bbox_mask = mask_to_bbox(mask_pr_bin[0])
                if bbox_mask is not None:
                    ms_h, ms_w = mask_pr_bin.shape[1:]
                    sx = cfg.DATASET.INPUT_SIZE[0] / ms_w
                    sy = cfg.DATASET.INPUT_SIZE[1] / ms_h
                    bbox_mask = bbox_mask * np.array([sx, sy, sx, sy], dtype=np.float32)

            # Detection box in original-normalized coordinates (for saving)
            if bbox_mask is None:
                mask_bbox.append(np.full(4, np.nan, dtype=np.float64))
            else:
                mask_bbox.append(
                    bbox_mask / np.array([cfg.DATASET.INPUT_SIZE[0], cfg.DATASET.INPUT_SIZE[1],
                                          cfg.DATASET.INPUT_SIZE[0], cfg.DATASET.INPUT_SIZE[1]],
                                         dtype=np.float32)
                )

            heat_ok = False
            stage1_speed_q = stage1_speed_t = None

            # Post-process outputs for each task
            for i, name in enumerate(cfg.TEST.HEAD):
                iou, err_q, err_t       = None, None, None
                speed_q, speed_t, speed = None, None, None
                reject = False

                if name == 'heatmap':
                    keypts_pr, q_pr, t_pr, reject = solve_pose_from_heatmaps(
                        outputs[i].squeeze(0).cpu(),
                        cfg.DATASET.IMAGE_SIZE,
                        cfg.TEST.HEATMAP_THRESHOLD,
                        camera, keypts_true_3D,
                        udp=cfg.DATASET.UDP
                    )

                    if not reject:
                        err_q = error_orientation(q_pr, q_gt, 'quaternion') # [deg]
                        err_t = error_translation(t_pr, t_gt)
                        speed_t, speed_q, speed = speed_score(t_pr, q_pr, t_gt, q_gt,
                            representation='quaternion',
                            applyThreshold=domain in ['lightbox', 'sunlamp'],
                            theta_q=cfg.TEST.SPEED_THRESHOLD_Q,
                            theta_t=cfg.TEST.SPEED_THRESHOLD_T)
                        heat_ok = True
                        stage1_speed_q, stage1_speed_t = speed_q, speed_t
                    else:
                        q_pr = t_pr = None

                    # Outputs
                    h_reject.append(reject)
                    q_hmap.append(q_pr)
                    t_hmap.append(t_pr)

                    # Metrics
                    result = {'_eR': err_q, '_eT': err_t, '_eR_HIL': speed_q, '_eT_norm_HIL': speed_t, '_pose': speed}

                elif name == 'segmentation':
                    iou = segment_iou(mask_pr_bin, mask_gt)

                    # Detection IoU from the mask-derived box
                    if bbox_mask is None:
                        det_iou = 0.0
                    else:
                        det_iou = float(np.squeeze(bbox_iou(bbox_mask, bbox_gt, x1y1x2y2=True)))

                    # Metrics
                    result = {'_iou': iou, '_det_iou': det_iou}

                elif name == 'efficientpose':
                    classification, bbox_prediction, \
                        rotation_raw, translation = outputs[i]
                    _, cls_argmax = torch.max(classification, dim=1)

                    # Bbox
                    bbox_pr = bbox_prediction[0,cls_argmax].squeeze().cpu().numpy()
                    R_pr    = rot_6d_to_matrix(rotation_raw[0,cls_argmax,:]).squeeze().cpu().numpy()
                    t_pr    = translation[0,cls_argmax].squeeze().cpu().numpy()

                    # IoU metric for bounding boxes
                    iou = np.squeeze(bbox_iou(bbox_pr, bbox_gt, x1y1x2y2=True))

                    # Pose metric(s)
                    err_q = error_orientation(R_pr, R_gt, 'rotationmatrix') # [deg]
                    err_t = error_translation(t_pr, t_gt)
                    speed_t, speed_q, speed = speed_score(t_pr, R_pr, t_gt, R_gt,
                        representation='rotationmatrix',
                        applyThreshold=domain in ['lightbox', 'sunlamp'],
                        theta_q=cfg.TEST.SPEED_THRESHOLD_Q,
                        theta_t=cfg.TEST.SPEED_THRESHOLD_T)

                    # Save & record
                    R_reg.append(R_pr)
                    t_reg.append(t_pr)
                    bbox_reg.append(bbox_pr)

                    # Metrics
                    result = {'_iou': iou, '_eR': err_q, '_eT': err_t, '_eR_HIL': speed_q, '_eT_norm_HIL': speed_t, '_pose': speed}

                # Update metrics of this head
                if not reject:
                    for m, v in result.items():
                        metrics[name[:4]+m].update(v, 1)
                        per_sample_metrics[name[:4]+m].append(v)

            # --- Stage-2: crop the original image with the mask box and run again
            ref_ok = False
            ref_speed_q = ref_speed_t = None
            if refine_enabled:
                valid_orig = ('image_orig' in targets) and (bbox_mask is not None)

                if heat_ok and valid_orig:
                    # Mask box (input px) -> original px
                    bbox_orig = bbox_mask * np.array(
                        [cfg.DATASET.IMAGE_SIZE[0] / cfg.DATASET.INPUT_SIZE[0],
                         cfg.DATASET.IMAGE_SIZE[1] / cfg.DATASET.INPUT_SIZE[1],
                         cfg.DATASET.IMAGE_SIZE[0] / cfg.DATASET.INPUT_SIZE[0],
                         cfg.DATASET.IMAGE_SIZE[1] / cfg.DATASET.INPUT_SIZE[1]],
                        dtype=np.float32
                    )
                    bw = max(float(bbox_orig[2] - bbox_orig[0]), 1.0)
                    bh = max(float(bbox_orig[3] - bbox_orig[1]), 1.0)
                    cx = (bbox_orig[0] + bbox_orig[2]) / 2
                    cy = (bbox_orig[1] + bbox_orig[3]) / 2

                    window = compute_crop_window(
                        cx, cy, bw, bh, cfg.DATASET.IMAGE_SIZE,
                        expand=cfg.TEST.CROP_EXPAND, min_side=64
                    )

                    crop = crop_to_input(targets['image_orig'][0], window,
                                         cfg.DATASET.INPUT_SIZE,
                                         _IMAGENET_MEAN, _IMAGENET_STD).to(device)

                    out2 = model(crop, is_train=False, gpu=device)

                    heat_idx = cfg.TEST.HEAD.index('heatmap')
                    _, q_ref_i, t_ref_i, reject2 = solve_pose_from_heatmaps(
                        out2[heat_idx].squeeze(0).cpu(),
                        cfg.DATASET.IMAGE_SIZE,
                        cfg.TEST.HEATMAP_THRESHOLD,
                        camera, keypts_true_3D,
                        udp=cfg.DATASET.UDP,
                        window=window
                    )

                    if not reject2:
                        err_q2 = error_orientation(q_ref_i, q_gt, 'quaternion') # [deg]
                        err_t2 = error_translation(t_ref_i, t_gt)
                        ref_speed_t, ref_speed_q, ref_speed = speed_score(
                            t_ref_i, q_ref_i, t_gt, q_gt,
                            representation='quaternion',
                            applyThreshold=domain in ['lightbox', 'sunlamp'],
                            theta_q=cfg.TEST.SPEED_THRESHOLD_Q,
                            theta_t=cfg.TEST.SPEED_THRESHOLD_T)

                        for m, v in {'_eR': err_q2, '_eT': err_t2,
                                     '_eR_HIL': ref_speed_q,
                                     '_eT_norm_HIL': ref_speed_t,
                                     '_pose': ref_speed}.items():
                            metrics['ref'+m].update(v, 1)
                            per_sample_metrics['ref'+m].append(v)

                        ref_ok = True
                        q_ref.append(q_ref_i)
                        t_ref.append(t_ref_i)
                    else:
                        q_ref.append(None)
                        t_ref.append(None)
                else:
                    q_ref.append(None)
                    t_ref.append(None)

                ref_reject.append(not ref_ok)

            # Final pose score
            if heat_ok:
                if ref_ok:
                    metrics['final_pose'].update(ref_speed_q + ref_speed_t, 1)
                else:
                    metrics['final_pose'].update(stage1_speed_q + stage1_speed_t, 1)

        # Elapsed time
        batch_time.update((time.time() - start)*1000)
        end = time.time()

        if cfg.VERBOSE:
            progress.display(idx+1)
        if idx+1 == num_batches:
            progress.display_summary()

        # Break
        if valid_fraction and idx + 1 == num_batches:
            break

    # ----- Write results
    if log_dir is not None:
        if '/' in cfg.DATASET.DATANAME:
            partition = cfg.DATASET.DATANAME.split('/')[-1]
        else:
            partition = ''

        if cfg.DATASET.DATANAME == 'prisma25':
            domain = 'prisma25'

        log_dir = os.path.join(log_dir, partition, domain)
        os.makedirs(log_dir, exist_ok=True)

        # Aggregate different performances
        results_str = []
        results_mat = {}
        for name in cfg.TEST.HEAD:
            # Final results in string
            results_str += [f'Head: {name}\n']
            results_str += [
                f'{n[5:]} [{_get_metric_unit(n)}]: {m.avg:.5f}\n' for n, m in metrics.items() if n[:4] in name
            ]
            results_str += ['\n']

            # Final results in .mat
            for n, m in metrics.items():
                if n[:4] in name:
                    results_mat[n] = m.avg

            # Number of rejections for heatmap
            if name == 'heatmap':
                results_str += [f'# reject: {sum(h_reject):,d}\n\n']
                results_mat['h_reject'] = sum(h_reject)

        # Two-stage refinement block
        if refine_enabled:
            results_str += ['Head: refine (stage-2 crop)\n']
            results_str += [
                f'{n[4:]} [{_get_metric_unit(n)}]: {m.avg:.5f}\n'
                for n, m in metrics.items() if n[:4] == 'ref_'
            ]
            results_str += [f'# fallback: {sum(ref_reject):,d}\n\n']

            for n, m in metrics.items():
                if n[:4] == 'ref_':
                    results_mat[n] = m.avg
            results_mat['ref_fallback'] = sum(ref_reject)

        per_sample_metrics["h_reject"] = h_reject
        if refine_enabled:
            per_sample_metrics["ref_reject"] = ref_reject

        # Final pose
        results_str += ['Final Pose: {:.5f}\n'.format(metrics['final_pose'].avg)]
        results_mat['final_pose'] = metrics['final_pose'].avg

        # Write average performances
        resultfn = os.path.join(log_dir, 'mean_performance.txt')
        with open(resultfn, 'w') as f:
            [f.write(m) for m in results_str]
        logger.info(f'Mean performance (text) written to {resultfn}')

        # Save average performance
        resultfn = os.path.join(log_dir, 'mean_performance.mat')
        savemat(resultfn, results_mat, appendmat=False)
        logger.info(f'Mean performance (val) written to {resultfn}')

        # Save individual performances
        errorfn = os.path.join(log_dir, 'per_sample_performance.mat')
        savemat(errorfn, per_sample_metrics, appendmat=False)
        logger.info(f'Per-sample performances saved to {errorfn}')

        # Write predictions
        # predfn = os.path.join(log_dir, 'predictions_raw_shirt.mat')
        # savemat(predfn, {'hmap': heatmaps, 'hmap_q': q_hmap, 'hmap_t': t_hmap, 'eff_R': R_reg, 'eff_t': t_reg}, appendmat=False)

        predfn = os.path.join(log_dir, 'predictions_pose.mat')
        # 注释：heatmap 头被拒的样本 q_pr/t_pr 为 None，savemat 无法序列化，改用 NaN 占位后再保存
        heat_q = [np.full(4, np.nan, dtype=np.float64) if q is None else q for q in q_hmap]
        heat_t = [np.full(3, np.nan, dtype=np.float64) if t is None else t for t in t_hmap]

        predictions = {'heat_q': heat_q, 'heat_t': heat_t, 'effi_R': R_reg, 'effi_t': t_reg,
                       'bbox': bbox_reg, 'reject': h_reject, 'mask_bbox': mask_bbox}

        if refine_enabled:
            ref_q = [np.full(4, np.nan, dtype=np.float64) if q is None else q for q in q_ref]
            ref_t = [np.full(3, np.nan, dtype=np.float64) if t is None else t for t in t_ref]
            predictions.update({'ref_q': ref_q, 'ref_t': ref_t, 'ref_reject': ref_reject})

        savemat(predfn, predictions, appendmat=False)
        logger.info(f'Pose predictions saved to {predfn}')

    return metrics['final_pose'].avg



