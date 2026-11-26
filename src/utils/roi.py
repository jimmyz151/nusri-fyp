'''
Segmentation-driven ROI utilities.

All boxes are [x1, y1, x2, y2] in pixels unless stated otherwise.  Cropping is
done with a single affine warp so that sub-pixel boxes and reflected borders are
handled consistently at train and test time.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


def mask_to_bbox(mask, threshold=0.5, min_area=1):
    """Largest connected component of a mask -> pixel box.

    Args:
        mask: 2D array. Float masks are thresholded at `threshold`; integer /
            boolean masks are interpreted as binary.
        min_area: minimum component area in mask pixels.

    Returns:
        (box, area), where box is a float array [x1, y1, x2, y2] in mask pixels
        (x2 / y2 exclusive) or None when no component passes `min_area`.
    """
    if mask.ndim != 2:
        mask = np.squeeze(mask)

    if mask.dtype == np.bool_:
        binary = mask.astype(np.uint8)
    elif np.issubdtype(mask.dtype, np.floating):
        binary = (mask > threshold).astype(np.uint8)
    else:
        binary = (mask > 0).astype(np.uint8)

    num, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return None, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    area = int(stats[idx, cv2.CC_STAT_AREA])
    if area < min_area:
        return None, area

    x1 = float(stats[idx, cv2.CC_STAT_LEFT])
    y1 = float(stats[idx, cv2.CC_STAT_TOP])
    x2 = x1 + float(stats[idx, cv2.CC_STAT_WIDTH])
    y2 = y1 + float(stats[idx, cv2.CC_STAT_HEIGHT])

    return np.array([x1, y1, x2, y2], dtype=np.float64), area


def square_expand_box(box, expand=1.0, min_side=1.0):
    """Make a box square around its center and scale the side length.

    `expand` is applied to the longer side before squaring.  Out-of-image parts
    are intentionally kept: `crop_resize` pads with reflection.
    """
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1) * float(expand)
    side = max(side, float(min_side))
    half = 0.5 * side
    return np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float64)


def shift_box(box, dx, dy):
    box = np.asarray(box, dtype=np.float64).copy()
    box[[0, 2]] += dx
    box[[1, 3]] += dy
    return box


def crop_resize(image, box, out_size, interpolation=cv2.INTER_LINEAR):
    """Crop `box` from `image` and resize to `out_size` with one affine warp.

    Returns:
        out, (sx, sy, ox, oy)
        where original = crop_pixel / (sx, sy) + (ox, oy).
    """
    out_w, out_h = int(out_size[0]), int(out_size[1])
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    side_x = max(x2 - x1, 1e-6)
    side_y = max(y2 - y1, 1e-6)
    sx = out_w / side_x
    sy = out_h / side_y

    mat = np.array([[sx, 0.0, -x1 * sx],
                    [0.0, sy, -y1 * sy]], dtype=np.float64)
    out = cv2.warpAffine(
        image, mat, (out_w, out_h),
        flags=interpolation,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return out, (sx, sy, x1, y1)


def points_to_crop(points, affine, out_size):
    """Map [N, 2] pixel points (original frame) to normalized crop coordinates."""
    sx, sy, ox, oy = affine
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    px = (points[:, 0] - ox) * sx / float(out_size[0])
    py = (points[:, 1] - oy) * sy / float(out_size[1])
    return np.stack((px, py), axis=-1)


def bbox_to_crop(box_px, affine, out_size):
    """Map a pixel bbox in the original frame to normalized crop coordinates."""
    box = np.asarray(box_px, dtype=np.float64)
    if box.ndim == 1:
        box = box[None, :]
    out = np.empty_like(box)
    sx, sy, ox, oy = affine
    out[:, [0, 2]] = (box[:, [0, 2]] - ox) * sx / float(out_size[0])
    out[:, [1, 3]] = (box[:, [1, 3]] - oy) * sy / float(out_size[1])
    return out[0]


def mask_crop_resize(mask, box, out_size):
    """Nearest-neighbour crop + resize for a binary mask."""
    out, _ = crop_resize(
        mask, box, out_size,
        interpolation=cv2.INTER_NEAREST,
    )
    return out


def box_quality(pred_box, gt_box, keypoints_px, eps=1e-6):
    """ROI-box diagnostics used by the mask-driven sensitivity sweep.

    `keypoints_px` is [N, 2] in original pixels.  Returns a dict with the same
    quantities as the GTBBox box-sensitivity probe: center error normalized by
    the GT side, side ratio, number of keypoints outside, and all-in flag.
    """
    pred_box = np.asarray(pred_box, dtype=np.float64)
    gt_box = np.asarray(gt_box, dtype=np.float64)

    gt_side = max(gt_box[2] - gt_box[0], gt_box[3] - gt_box[1], eps)
    pred_side = max(pred_box[2] - pred_box[0], pred_box[3] - pred_box[1], eps)

    gt_cx, gt_cy = 0.5 * (gt_box[0] + gt_box[2]), 0.5 * (gt_box[1] + gt_box[3])
    pr_cx, pr_cy = 0.5 * (pred_box[0] + pred_box[2]), 0.5 * (pred_box[1] + pred_box[3])
    center_err = float(np.hypot(pr_cx - gt_cx, pr_cy - gt_cy) / gt_side)
    side_ratio = float(pred_side / gt_side)

    if keypoints_px is None or len(keypoints_px) == 0:
        n_out = 0
        allin = True
    else:
        kp = np.asarray(keypoints_px, dtype=np.float64)
        inside = ((kp[:, 0] >= pred_box[0]) & (kp[:, 0] <= pred_box[2]) &
                  (kp[:, 1] >= pred_box[1]) & (kp[:, 1] <= pred_box[3]))
        n_out = int((~inside).sum())
        allin = bool(n_out == 0)

    return {
        'center_err': center_err,
        'side_ratio': side_ratio,
        'n_out': n_out,
        'allin': allin,
    }


def center_crop_box(image_size, side=None):
    """Square center crop fallback when the mask head produces nothing."""
    w, h = float(image_size[0]), float(image_size[1])
    side = min(w, h) if side is None else float(side)
    cx, cy = 0.5 * w, 0.5 * h
    return np.array([cx - side / 2, cy - side / 2,
                     cx + side / 2, cy + side / 2], dtype=np.float64)
