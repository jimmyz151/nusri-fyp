'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import argparse
import csv
import time
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # tensorboard 未安装时也不影响训练
    SummaryWriter = None

from config import cfg, update_config
from nets import build_spnv2
from dataset import get_dataloader
from solver import get_optimizer, adjust_learning_rate, get_scaler
from engine.trainer import do_train
from engine.inference import do_valid
from utils.checkpoints import load_checkpoint, save_checkpoint
from utils.utils import (
    set_seeds_cudnn, setup_logger, create_logger_directories,
    write_model_info, load_camera_intrinsics, load_tango_3d_keypoints,
)

# 已知无害的告警：DDP 对 1x1 卷积梯度的 stride 提示，以及 DataLoader 退出时的信号量提示
warnings.filterwarnings('ignore', message=r'.*Grad strides do not match bucket view strides.*')
warnings.filterwarnings('ignore', message=r'.*leaked semaphore objects.*')

PROBE_FIELDS = [
    'epoch', 'heat_eR', 'heat_eT', 'heat_eR_HIL', 'heat_eT_norm_HIL', 'heat_pose',
    'effi_eR', 'effi_eT', 'effi_pose', 'segm_iou', 'reject', 'heat_fail30',
    'heat_median_eR', 'final_pose',
]


def _tb_add(writer, tag, value, step):
    """TensorBoard 写入必须永远不抛出异常，避免监控影响训练。"""
    if writer is None or value is None:
        return
    try:
        writer.add_scalar(tag, float(value), int(step))
    except Exception:
        pass


def _tb_config_text(cfg):
    heads = cfg.MODEL.HEAD.NAMES
    return '\n'.join([
        f'backbone: {cfg.MODEL.BACKBONE.NAME}',
        f'heads: {heads}',
        f'neck_norm: {"GN" if cfg.MODEL.USE_GROUPNORM_BACKBONE else "BN"}',
        f'output_ln: {cfg.MODEL.BACKBONE.USE_OUTPUT_LN}',
        f'lr: {cfg.TRAIN.LR}',
        f'lr_step: {cfg.TRAIN.LR_STEP}',
        f'epochs: {cfg.TRAIN.END_EPOCH}',
        f'batch_per_gpu: {cfg.TRAIN.IMAGES_PER_GPU}',
        f'syncbn: {getattr(cfg.DIST, "SYNCBN", True)}',
    ])


def _append_probe_csv(path, epoch, metrics):
    """只写标量指标，不保存权重/预测/逐样本文件；失败静默。"""
    try:
        os.makedirs(osp.dirname(path), exist_ok=True)
        done = set()
        if osp.exists(path):
            with open(path, newline='') as f:
                for row in csv.DictReader(f):
                    try:
                        done.add(int(row.get('epoch', -1)))
                    except (TypeError, ValueError):
                        pass
        if epoch in done:
            return
        exists = osp.exists(path)
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=PROBE_FIELDS)
            if not exists:
                writer.writeheader()
            row = {k: metrics.get(k, '') for k in PROBE_FIELDS}
            row['epoch'] = epoch
            writer.writerow(row)
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description='Train SPNv2')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training (default: 1)')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training (default: 0)')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')

    args = parser.parse_args()

    return args

def main(cfg):
    args = parse_args()
    update_config(cfg, args)

    cfg.defrost()
    cfg.DIST.RANK = args.rank
    cfg.freeze()

    _, output_dir, log_dir = \
        create_logger_directories(cfg, phase='train', write_cfg_to_file=True)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.multiprocessing_distributed = cfg.DIST.MULTIPROCESSING_DISTRIBUTED
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Adjust world_size
        args.world_size = ngpus_per_node * args.world_size

        # 子进程退出时 multiprocessing 会打印一条无害的泄漏信号量提示，这里通过环境变量静默它
        os.environ.setdefault(
            'PYTHONWARNINGS', 'ignore::UserWarning:multiprocessing.resource_tracker')

        # Spawn distributed processes
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, output_dir, log_dir)
        )
    else:
        main_worker(
            args.gpu,
            ngpus_per_node,
            args,
            output_dir,
            log_dir
        )

def main_worker(gpu, ngpus_per_node, args, output_dir, log_dir):

    # Set all seeds & cudNN
    set_seeds_cudnn(cfg, seed=cfg.SEED)

    # GPU?
    args.gpu = gpu
    if args.gpu is not None:
        print(f'Use GPU: {args.gpu} for training')

    # Distributed?
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        print('Init process group: dist_url: {}, world_size: {}, rank: {}'.
              format(args.dist_url, args.world_size, args.rank))

        dist.init_process_group(
            backend=cfg.DIST.BACKEND,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )

    update_config(cfg, args)

    # setup logger
    logger = setup_logger(log_dir, args.rank, 'train', to_console=args.distributed)

    # build network
    model = build_spnv2(cfg)

    # 微调探针：冻结 ConvNeXt 特征提取器，只训练 neck / heads
    if cfg.TRAIN.FREEZE_BACKBONE:
        if not hasattr(model.backbone, 'features'):
            raise NotImplementedError('FREEZE_BACKBONE 目前只支持 ConvNeXt 骨干')
        for p in model.backbone.features.parameters():
            p.requires_grad = False
        logger.info('   - Backbone features frozen for fine-tuning')

    # GPU device
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda', args.gpu)
    else:
        device = torch.device('cuda')

    if args.distributed:
        # SyncBN
        if not cfg.MODEL.USE_GROUPNORM_BACKBONE or \
              not cfg.MODEL.USE_GROUPNORM_HEADS:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logger.info('   - SyncBN activated for distributed training')

        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            model = torch.nn.parallel.DistributedDataParallel(
                model.to(device), device_ids=[args.gpu],
                find_unused_parameters=cfg.MODEL.FIND_UNUSED_PARAM
            )
            logger.info('   - Model wrapped to nn.parallel.DistributedDataParallel')
        else:
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model.to(device))
    else:
        model = model.to(device)

    # write model summary to file
    if args.rank == 0:
        write_model_info(model.module if args.distributed else model, log_dir)

    # TensorBoard：仅 rank 0；任何失败都只告警，不影响训练
    tb_writer = None
    if args.rank == 0 and cfg.TRAIN.TENSORBOARD:
        if SummaryWriter is None:
            logger.warning('TensorBoard disabled: tensorboard is not installed')
        else:
            try:
                tb_dir = osp.join(cfg.ROOT, 'results', 'tensorboard', cfg.EXP_NAME)
                os.makedirs(tb_dir, exist_ok=True)
                open(osp.join(tb_dir, 'LIVE'), 'w').close()
                tb_writer = SummaryWriter(tb_dir)
                tb_writer.add_text('config/key_params', _tb_config_text(cfg))
                logger.info(f'   - TensorBoard logging to {tb_dir}')
            except Exception as e:
                logger.warning(f'TensorBoard disabled: {e}')
                tb_writer = None

    # Dataloaders
    train_loader = get_dataloader(cfg,
                                  split='train',
                                  distributed=args.distributed,
                                  load_labels=True)
    val_loader   = get_dataloader(cfg,
                                  split='val',
                                  distributed=args.distributed,
                                  load_labels=True)

    # 训练内 probe（目标域小子集，仅监控；失败不影响训练）
    probe_loader = None
    probe_cfg = None
    if cfg.TEST.PROBE_ENABLE and args.rank == 0:
        try:
            probe_cfg = cfg.clone()
            probe_cfg.defrost()
            probe_cfg.TEST.TEST_CSV = cfg.TEST.PROBE_CSV
            probe_cfg.TEST.IMAGES_PER_GPU = 1
            probe_cfg.freeze()
            probe_loader = get_dataloader(probe_cfg, split='test',
                                          distributed=False, load_labels=True)
            logger.info('   - Probe enabled: {} (every {} epoch(s))'
                        .format(cfg.TEST.PROBE_CSV, cfg.TEST.PROBE_FREQ))
        except Exception as e:
            logger.warning(f'Probe disabled: {e}')
            probe_loader = None

    # Optimizer & scaler for mixed-precision training
    optimizer = get_optimizer(cfg, model)
    scaler    = get_scaler(cfg) # None if cfg.FP16 = False, cfg.CUDA = False

    # Online style randomization (SPEED+ baseline setting: new style every batch)
    style_augmentor = None
    style_pipeline  = None
    if cfg.AUGMENT.ONLINE_TEXTURE:
        from utils.style_augment import OnlineStyleAugmentor
        style_augmentor = OnlineStyleAugmentor(cfg.AUGMENT.RANDOM_TEXTURE.ALPHA, device)
        logger.info('   - Online texture randomization: alpha {:.2f}, batch prob {:.2f}'
                    .format(cfg.AUGMENT.RANDOM_TEXTURE.ALPHA, cfg.AUGMENT.RANDOM_TEXTURE.PROB))
        if cfg.AUGMENT.ONLINE_TEXTURE_BEFORE_AUGMENT:
            from utils.style_pipeline import StyleBeforeAugment
            style_pipeline = StyleBeforeAugment(cfg, style_augmentor,
                                                workers=cfg.TRAIN.AUGMENT_WORKERS)
            # 风格化已包含在 loader 中，训练循环不再重复施加
            train_loader = style_pipeline.wrap(train_loader)
            style_augmentor = None
            logger.info('   - Augmentation order: online style -> pixel augmentations -> normalization')
            logger.info('   - Pixel augmentations run in {} worker processes'.format(
                cfg.TRAIN.AUGMENT_WORKERS))
        else:
            logger.info('   - Augmentation order: pixel augmentations -> online style (exp4)')

    # Load checkpoints
    checkpoint_file = osp.join(output_dir, f'checkpoint.pth.tar')
    if cfg.AUTO_RESUME and osp.exists(checkpoint_file):
        last_epoch, best_score = load_checkpoint(
                        checkpoint_file,
                        model,
                        optimizer,
                        scaler,
                        device)
        begin_epoch = last_epoch
    else:
        begin_epoch = cfg.TRAIN.BEGIN_EPOCH
        last_epoch  = -1
        best_score  = 1e20

    # For validation
    camera = load_camera_intrinsics(cfg.DATASET.CAMERA)
    keypts_true_3D = load_tango_3d_keypoints(cfg.DATASET.KEYPOINTS)

    # ---------------------------------------
    # Main loop
    # ---------------------------------------
    score   = best_score
    is_best = False
    probe_csv = osp.join(cfg.ROOT, 'experiments', 'probes', cfg.EXP_NAME, 'online_probe.csv')
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        # Learning rate adjustment
        adjust_learning_rate(optimizer, epoch, cfg)
        _tb_add(tb_writer, 'opt/lr', optimizer.param_groups[0]['lr'], epoch+1)

        # Single epoch training
        t_epoch = time.time()
        train_losses, ms_per_batch = do_train(epoch,
                 cfg,
                 model,
                 train_loader,
                 optimizer,
                 log_dir=log_dir,
                 device=device,
                 scaler=scaler,
                 rank=args.rank,
                 style_augmentor=style_augmentor)
        _tb_add(tb_writer, 'perf/epoch_seconds', time.time() - t_epoch, epoch+1)
        _tb_add(tb_writer, 'perf/ms_per_batch', ms_per_batch, epoch+1)
        for name, value in train_losses.items():
            _tb_add(tb_writer, f'train/{name}', value, epoch+1)

        if not args.distributed or (args.distributed and args.rank == 0):
            # Validate on the fraction of dataset
            if (epoch+1) % cfg.TRAIN.VALID_FREQ == 0:
                score, val_metrics = do_valid(epoch,
                                 cfg,
                                 model,
                                 val_loader,
                                 camera,
                                 keypts_true_3D,
                                 valid_fraction=cfg.TRAIN.VALID_FRACTION,
                                 log_dir=None,
                                 device=device,
                                 save_artifacts=False,
                                 return_metrics=True)
                for name, value in val_metrics.items():
                    _tb_add(tb_writer, f'val/{name}', value, epoch+1)

                if score < best_score:
                    best_score = score
                    is_best = True
                else:
                    is_best = False

            # 训练内 probe：只记录指标，不保存权重/预测；失败不影响训练，也不自动中止
            if probe_loader is not None and (epoch+1) % max(1, cfg.TEST.PROBE_FREQ) == 0:
                try:
                    free_mb = (torch.cuda.mem_get_info(device)[0] / (1024 ** 2)
                               if device.type == 'cuda' else float('inf'))
                    if free_mb < cfg.TEST.PROBE_MIN_FREE_MB:
                        logger.warning('Probe skipped at epoch %d: only %.0f MB free',
                                       epoch+1, free_mb)
                    else:
                        _, probe_metrics = do_valid(epoch,
                                                    probe_cfg,
                                                    model,
                                                    probe_loader,
                                                    camera,
                                                    keypts_true_3D,
                                                    valid_fraction=None,
                                                    log_dir=None,
                                                    device=device,
                                                    phase='Probe',
                                                    save_artifacts=False,
                                                    return_metrics=True,
                                                    show_progress=False)
                        _append_probe_csv(probe_csv, epoch+1, probe_metrics)
                        for name, value in probe_metrics.items():
                            _tb_add(tb_writer, f'probe/{name}', value, epoch+1)
                        logger.info('Probe %02d: final_pose %.4f | eR %.2f deg | eT %.3f m',
                                    epoch+1,
                                    probe_metrics.get('final_pose', float('nan')),
                                    probe_metrics.get('heat_eR', float('nan')),
                                    probe_metrics.get('heat_eT', float('nan')))
                except Exception as e:
                    logger.warning('Probe failed at epoch %d: %s', epoch+1, e)
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                finally:
                    model.train()

            # Save
            save_checkpoint({
                'epoch': epoch+1,
                'backbone': cfg.MODEL.BACKBONE.NAME,
                'heads': cfg.MODEL.HEAD.NAMES,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'best_score': best_score,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None else None
            }, is_best, epoch+1==cfg.TRAIN.END_EPOCH, output_dir)

    if tb_writer is not None:
        try:
            tb_writer.flush()
            tb_writer.close()
        except Exception:
            pass

    # 训练结束后显式关闭增强进程池：其子进程是非 daemon，解释器退出时会一直 join，
    # 导致训练跑完后进程不退出并继续占用显存
    if style_pipeline is not None:
        style_pipeline.close()

if __name__=='__main__':
    main(cfg)
