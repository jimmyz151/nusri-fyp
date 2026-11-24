'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import random
import logging
import time
import json
from plyfile  import PlyData
from pathlib  import Path
from scipy.io import loadmat
from enum     import Enum

import torch
from torchinfo import summary

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# For reporting & logging
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """ Computes and stores the average and current value

        Modified from
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self, name, unit='-', fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.unit = unit
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def avg_str(self):
        """ Compact `name avg unit` representation used by the progress line """
        unit = '' if self.unit in ('', '-') else f' {self.unit}'
        spec = self.fmt[1:] if self.fmt.startswith(':') else self.fmt
        return f'{self.name} {self.avg:{spec}}{unit}'.strip()

    def __str__(self):
        spec = self.fmt[1:] if self.fmt.startswith(':') else self.fmt
        return f'{self.name or ""} {self.val:{spec}} {self.avg:{spec}}'.strip()

def _format_duration(seconds):
    """ Format a duration in seconds as h:mm:ss (or mm:ss below one hour) """
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f'{hours:d}:{minutes:02d}:{secs:02d}'
    return f'{minutes:02d}:{secs:02d}'

class ProgressMeter(object):
    """ Compact training/testing progress reporter.

        - On a terminal the line is refreshed in place every batch.
        - When stdout is redirected to a file, one line is emitted every
          `log_freq` batches (about 60 lines per epoch) so logs stay readable.
    """
    def __init__(self, num_batches, timer, meters, prefix="", log_freq=None):
        self.num_batches = max(int(num_batches), 1)
        self.batch_fmtstr = self._get_batch_fmtstr(self.num_batches)
        self.timer  = timer
        self.meters = meters
        self.prefix = prefix
        self.log_freq = log_freq or max(1, self.num_batches // 60)

    def display(self, batch):
        finished = batch >= self.num_batches
        line_mode = not sys.stdout.isatty()

        # Redirected output: emit standalone lines at a reduced rate
        if line_mode and not finished and batch > 1 and batch % self.log_freq != 0:
            return

        msg = self._format(batch)
        if line_mode:
            sys.stdout.write(msg + '\n')
        else:
            # Refresh the same terminal line in place
            sys.stdout.write('\r' + msg + ('\n' if finished else ''))
        sys.stdout.flush()

    def display_summary(self):
        entries = [
            f'batches {self.num_batches}',
            f'{self.timer.avg:.1f} ms/batch',
        ]
        entries += [meter.avg_str() for meter in self.meters if meter.name]
        logger.info(f'{self.prefix} summary | ' + ' | '.join(entries))

    def _format(self, batch):
        progress = 100.0 * batch / self.num_batches
        elapsed = self.timer.avg * batch / 1000.0
        # A single sample is too noisy to give a meaningful ETA
        eta = '--:--' if batch <= 1 else \
            _format_duration(self.timer.avg * max(self.num_batches - batch, 0) / 1000.0)

        msg = (f'{self.prefix} {self.batch_fmtstr.format(batch)} '
               f'{progress:5.1f}% | {_format_duration(elapsed)}<{eta} | '
               f'{self.timer.avg:6.1f} ms/batch')
        if self.meters:
            msg += ' | ' + ' | '.join(meter.avg_str() for meter in self.meters)
        return msg

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class NoOp:
    # https://discuss.pytorch.org/t/ddp-training-log-issue/125808
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            """Accept every signature by doing non-operation."""
            pass

        return no_op

def setup_logger(log_dir, rank, phase, to_console=True):
    if rank == 0:
        # File to save logger outputs
        log_file = os.path.join(log_dir, f'{phase}_rank{rank}.log')

        # Configure logger formats
        format  = '%(asctime)-15s %(message)s'
        datefmt = '%Y/%m/%d %H:%M:%S'
        logging.basicConfig(filename=str(log_file),
                            datefmt=datefmt,
                            format=format,
                            level=logging.INFO)

        # Root logger to the file
        logger = logging.getLogger()

        # Stream handler to the console
        if to_console:
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter(fmt=format, datefmt=datefmt))
            logger.addHandler(console)
    else:
        logger = NoOp()

    return logger

def create_logger_directories(cfg, phase='train', write_cfg_to_file=True):
    # Where to save outputs (e.g., checkpoints)
    output_dir = Path(cfg.OUTPUT_DIR) / cfg.EXP_NAME

    # Where to save logs
    time_str = time.strftime('%Y%m%d_%H_%M_%S')
    log_dir  = Path(cfg.LOG_DIR) / cfg.EXP_NAME / f'{phase}_{time_str}'

    # Master rank make output directories
    if cfg.DIST.RANK == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        while not output_dir.exists():
            time.sleep(5)

    # Create logger
    log_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_logger(log_dir, cfg.DIST.RANK, phase, to_console=True)

    if write_cfg_to_file:
        with open(log_dir / 'config.txt', 'w') as f:
            f.write(str(cfg))

    logger.info(f'Outputs (e.g., checkpoints) are saved at:   {output_dir}')
    logger.info(f'Messages and tensorboard logs are saved at: {log_dir}')

    return logger, str(output_dir), str(log_dir)

# -----------------------------------------------------------------------
# Functions regarding 3D model loading
def load_tango_3d_keypoints(mat_dir):
    vertices  = loadmat(mat_dir)['tango3Dpoints'] # [3 x 11]
    corners3D = np.transpose(np.array(vertices, dtype=np.float32)) # [11 x 3]

    return corners3D

def load_cad_model(ply_dir, num_points=500):
    model  = PlyData.read(ply_dir)
    vertex = model['vertex']
    points_3d = np.stack([vertex[:]['x'],
                          vertex[:]['y'],
                          vertex[:]['z']], axis = -1) # [N x 3]
    points_3d /= 1000 # mm -> m

    num_model_points = points_3d.shape[0]
    if num_points:
        if num_points > num_model_points:
            raise AssertionError(f'num_points must be less than or equal to current model size ({num_model_points})')
        elif num_points == num_model_points:
            return points_3d
        else:
            # Sample num_points vertices
            step_size = (num_model_points // num_points) - 1
            if step_size < 1:
                step_size = 1
            points_3d = points_3d[::step_size, :]
            return points_3d[:num_points, :]

def load_camera_intrinsics(camera_json):
    with open(camera_json) as f:
        cam = json.load(f)

    # Into np.array
    cam = {
        k: np.array(v, dtype=np.float32) for k, v in cam.items()
    }

    # Compute horizontal FOV
    cam["horizontalFOV"] = 2 * np.arctan2(0.5 * cam["ppx"] * cam["Nu"], cam["fx"])

    return cam

# -----------------------------------------------------------------------
# Miscellaneous functions.
def set_seeds_cudnn(cfg, seed=None):
    if seed is None:
        seed = int(time.time())

    logger.info(f'Random seed: {seed}')

    # Set seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cfg.CUDA and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark     = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled       = cfg.CUDNN.ENABLED

        # empty any cached GPU memory
        torch.cuda.empty_cache()

def _write_model_info(module, log_dir, filename, depth=3):
    model_summary = summary(module,
                            depth=depth,
                            col_names=["kernel_size", "num_params"],
                            row_settings=["var_names"],
                            verbose=0)
    with open(os.path.join(log_dir, filename), 'w') as f:
        f.write(str(model_summary))

def write_model_info(model, log_dir):
    _write_model_info(model.backbone, log_dir, 'backbone_simple.txt', depth=3)
    _write_model_info(model.backbone, log_dir, 'backbone_all.txt', depth=10)
    _write_model_info(model.heads, log_dir, 'heads_simple.txt', depth=3)
    _write_model_info(model.heads, log_dir, 'heads_all.txt', depth=10)

def num_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

def num_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum(p.numel() for p in model_parameters)
