'''
Crop geometry shared by the training dataset (error-simulated crops) and the
two-stage test-time refinement (mask-box driven crops).

All windows are expressed in ORIGINAL image pixels as (x0, y0, w, h).
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import torch


def compute_crop_window(cx, cy, w, h, image_size, expand=1.2,
                        shift=(0.0, 0.0), scale=1.0, min_side=64):
    """ 3:2 crop window around (cx, cy) that fully contains a w x h box.

    Args:
        cx, cy (float): window center in original pixels
        w, h (float): box size to cover [px]
        image_size (tuple): original image size (W, H)
        expand (float): expansion factor around the covered box
        shift (tuple): center offset in units of the window height
        scale (float): extra scale on the window height
        min_side (int): minimum window height [px]

    Returns:
        (x0, y0, win_w, win_h) integers, with win_w = round(1.5 * win_h)
    """
    W, H = image_size

    # Window height that contains the box in a 3:2 window
    need_h = max(h, w * H / W)
    win_h  = max(need_h * expand * scale, min_side)

    win_h = int(round(win_h))
    win_w = int(round(win_h * W / H))

    x0 = int(round(cx + shift[0] * win_h - win_w / 2))
    y0 = int(round(cy + shift[1] * win_h - win_h / 2))

    return x0, y0, win_w, win_h


def crop_with_pad(array, window):
    """ Crop an HWC or HW array, padding out-of-frame regions symmetrically.

    np.pad(mode='symmetric') supports pads wider than the image, so severe
    crops that are partially (or mostly) out of frame are handled.
    """
    x0, y0, w, h = window
    H, W = array.shape[:2]

    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x0 + w - W), max(0, y0 + h - H)

    if pl or pt or pr or pb:
        pad = ((pt, pb), (pl, pr)) + ((0, 0),) * (array.ndim - 2)
        array = np.pad(array, pad, mode='symmetric')

    xs, ys = x0 + pl, y0 + pt

    return array[ys:ys + h, xs:xs + w]


def map_points_to_crop(pts_px, window):
    """ Map (N, 2) points from original pixels to crop pixels """
    x0, y0, w, h = window
    pts = np.asarray(pts_px, dtype=np.float32).reshape(-1, 2)

    return (pts - np.array([x0, y0], dtype=np.float32)) / \
        np.array([w, h], dtype=np.float32)


def map_points_from_crop(pts_px, window):
    """ Map (N, 2) points from crop pixels back to original pixels """
    x0, y0, w, h = window
    pts = np.asarray(pts_px, dtype=np.float32).reshape(-1, 2)

    return pts * np.array([w, h], dtype=np.float32) + \
        np.array([x0, y0], dtype=np.float32)


def crop_to_input(image_chw, window, input_size, mean, std):
    """ Crop a uint8 CHW tensor, resize to input_size and ImageNet-normalize.

    Returns:
        torch.FloatTensor [1, 3, H, W]
    """
    image = image_chw.cpu().numpy()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    crop = crop_with_pad(image, window)
    crop = cv2.resize(crop, tuple(input_size), interpolation=cv2.INTER_LINEAR)

    x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float()
    x = x.div_(255.0)

    mean = torch.as_tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std  = torch.as_tensor(std,  dtype=torch.float32).view(3, 1, 1)
    x = (x - mean) / std

    return x.unsqueeze(0)
