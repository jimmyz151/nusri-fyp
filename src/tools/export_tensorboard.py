'''Export existing training logs to TensorBoard event files.

Parses logs/<EXP>/train_*/train_rank0.log (the run with the most validation
lines) and writes scalars + key hyper-parameters to results/tensorboard/<EXP>.

This is read-only w.r.t. training artifacts. It does not modify train.py.
'''
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
from typing import Any, Dict, List, Optional

import yaml
from torch.utils.tensorboard import SummaryWriter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.path.join(ROOT, 'results', 'tensorboard')

NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
SUMMARY_RE = re.compile(
    rf'Epoch\s+(\d+)/\d+\s+train summary.*?hmap\s+({NUM})(?:.*?seg\s+({NUM}))?')
VAL_OLD_RE = re.compile(
    rf'\* Time:\s*{NUM}\s*ms heat_eR\s+({NUM})\s*deg'
    rf'(?:.*?effi_eT\s+({NUM})\s*m)?.*?final_pose\s+({NUM})')
VAL_NEW_RE = re.compile(
    rf'Validation\s+(\d+)/\d+\s+summary.*?heat_eR\s+({NUM})\s*deg.*?final_pose\s+({NUM})')
LR_RE = re.compile(rf'Current epoch learning rate:\s*({NUM})')


def _f(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def pick_train_log(exp_dir: str) -> Optional[str]:
    best, best_n = None, -1
    for fp in glob.glob(os.path.join(exp_dir, 'train_*', 'train_rank0.log')):
        try:
            txt = open(fp, errors='ignore').read()
        except OSError:
            continue
        n = len(re.findall(r'heat_eR', txt))
        if n > best_n:
            best, best_n = fp, n
    return best


def parse_log(fp: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pending: Dict[str, Any] = {}
    current_lr: Optional[float] = None
    for raw in open(fp, errors='ignore').read().replace('\r', '\n').split('\n'):
        m = VAL_NEW_RE.search(raw)
        if m:
            row = {'epoch': int(m.group(1)), 'val_heat_eR': _f(m.group(2)),
                   'val_final': _f(m.group(3)), 'lr': current_lr}
            row.update(pending)
            rows.append(row)
            pending = {}
            continue

        m = VAL_OLD_RE.search(raw)
        if m and 'heat_eR' in raw and 'Testing' not in raw:
            row = {'epoch': len(rows) + 1, 'val_heat_eR': _f(m.group(1)),
                   'val_effi_eT': _f(m.group(2)), 'val_final': _f(m.group(3)),
                   'lr': current_lr}
            row.update(pending)
            rows.append(row)
            pending = {}
            continue

        m = SUMMARY_RE.search(raw)
        if m:
            pending.update({'epoch': int(m.group(1)), 'train_hmap': _f(m.group(2))})
            if m.group(3):
                pending['train_seg'] = _f(m.group(3))
            mm = re.search(rf'({NUM})\s*ms/batch', raw)
            if mm:
                pending['train_ms'] = _f(mm.group(1))
            continue

        if '* Time:' in raw and 'hmap' in raw and 'heat_eR' not in raw:
            mm = re.search(rf'\* Time:\s*({NUM})\s*ms', raw)
            if mm:
                pending['train_ms'] = _f(mm.group(1))
                for key, tag in (('hmap', 'train_hmap'), ('cls', 'train_cls'),
                                 ('box', 'train_box'), ('pose', 'train_pose'),
                                 ('seg', 'train_seg')):
                    mk = re.search(rf'(?<![A-Za-z]){key}\s+({NUM})', raw)
                    if mk:
                        pending[tag] = _f(mk.group(1))
                continue

        m = LR_RE.search(raw)
        if m:
            current_lr = _f(m.group(1))
    return rows


def exp_record_dir(exp: str) -> str:
    """Probe experiments live under experiments/probes/<exp>; model runs under experiments/<exp>."""
    probe = os.path.join(ROOT, 'experiments', 'probes', exp)
    return probe if os.path.isdir(probe) else os.path.join(ROOT, 'experiments', exp)


def load_cfg(exp: str, log_fp: str) -> Dict[str, Any]:
    for p in (os.path.join(exp_record_dir(exp), 'config.yaml'),
              os.path.join(os.path.dirname(log_fp), 'config.txt')):
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}


def _get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for k in path.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_hparams(cfg: Dict[str, Any]) -> Dict[str, Any]:
    heads = _get(cfg, 'MODEL.HEAD.NAMES', []) or []
    hp = {
        'backbone': str(_get(cfg, 'MODEL.BACKBONE.NAME', '?')),
        'heads': '+'.join(heads) if isinstance(heads, list) else str(heads),
        'neck_norm': 'GN' if _get(cfg, 'MODEL.USE_GROUPNORM_BACKBONE', False) else 'BN',
        'output_ln': bool(_get(cfg, 'MODEL.BACKBONE.USE_OUTPUT_LN', False)),
        'lr': _get(cfg, 'TRAIN.LR', 0.0),
        'scheduler': str(_get(cfg, 'TRAIN.SCHEDULER', '?')),
        'lr_step': str(_get(cfg, 'TRAIN.LR_STEP', [])),
        'epochs': int(_get(cfg, 'TRAIN.END_EPOCH', 0) or 0),
        'batch_per_gpu': int(_get(cfg, 'TRAIN.IMAGES_PER_GPU', 0) or 0),
        'pretrain': str(_get(cfg, 'MODEL.BACKBONE.PRETRAINED', 'none')),
        'offline_texture': bool(_get(cfg, 'AUGMENT.APPLY_TEXTURE_RANDOMIZATION', False)),
        'online_texture': bool(_get(cfg, 'AUGMENT.ONLINE_TEXTURE', False)),
        'syncbn': bool(_get(cfg, 'DIST.SYNCBN', True)),
        'input': str(_get(cfg, 'DATASET.INPUT_SIZE', [])),
    }
    return hp


def read_metrics_csv(exp: str) -> Dict[str, float]:
    fp = os.path.join(exp_record_dir(exp), 'metrics.csv')
    out: Dict[str, float] = {}
    if not os.path.exists(fp):
        return out
    for line in open(fp):
        parts = line.strip().split(',')
        if len(parts) != 4 or parts[0] == 'domain':
            continue
        dom, head, metric, val = parts
        if metric == 'final_pose':
            out[f'{dom}_final'] = float(val)
    return out


def read_ladder(exp: str) -> List[Dict[str, float]]:
    candidates = [
        os.path.join(ROOT, 'experiments', 'probes', exp, 'epoch_ladder.csv'),
        os.path.join(ROOT, 'experiments', 'probes', exp, 'online_probe.csv'),
        os.path.join(ROOT, 'experiments', exp, 'epoch_ladder.csv'),
    ]
    fp = next((p for p in candidates if os.path.exists(p)), None)
    rows: List[Dict[str, float]] = []
    if fp is None:
        return rows
    keys = None
    for line in open(fp):
        parts = line.strip().split(',')
        if keys is None:
            keys = parts
            continue
        try:
            rows.append({k: float(v) for k, v in zip(keys, parts) if v != ''})
        except ValueError:
            pass
    return rows


def export(exp: str) -> bool:
    log_fp = pick_train_log(os.path.join(ROOT, 'logs', exp))
    if not log_fp:
        return False
    rows = parse_log(log_fp)
    if not rows:
        return False

    cfg = load_cfg(exp, log_fp)
    out_dir = os.path.join(OUT_ROOT, exp)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    writer = SummaryWriter(out_dir)
    for r in rows:
        step = r['epoch']
        for k in ('train_hmap', 'train_cls', 'train_box', 'train_pose', 'train_seg',
                  'val_heat_eR', 'val_effi_eT', 'val_final', 'lr', 'train_ms'):
            v = r.get(k)
            if isinstance(v, (int, float)):
                tag = {'val_final': 'val/final_pose', 'val_heat_eR': 'val/heat_eR',
                       'val_effi_eT': 'val/effi_eT', 'train_ms': 'perf/ms_per_batch',
                       'lr': 'opt/lr'}.get(k, 'train/' + k.replace('train_', ''))
                writer.add_scalar(tag, v, step)

    ladder = read_ladder(exp)
    source_map = {
        'probe/eR_HIL': ('eR_HIL', 'heat_eR_HIL'),
        'probe/eT_norm_HIL': ('eT_norm_HIL', 'heat_eT_norm_HIL'),
        'probe/heat_pose': ('heat_pose',),
        'probe/seg_iou': ('seg_iou', 'segm_iou'),
        'probe/final_pose': ('final_pose',),
        'probe/heat_fail30': ('heat_fail30',),
        'probe/heat_median_eR': ('heat_median_eR',),
        'probe/reject': ('reject',),
    }
    for r in ladder:
        step = int(r.get('epoch', 1))
        for tag, keys in source_map.items():
            for k in keys:
                if k in r:
                    writer.add_scalar(tag, r[k], step)
                    break

    hp = build_hparams(cfg)
    writer.add_text('config/key_params', '\n'.join(f'{k}: {v}' for k, v in hp.items()))
    writer.add_text('config/yaml', '```yaml\n' + yaml.safe_dump(cfg, allow_unicode=True) + '\n```')
    metrics = read_metrics_csv(exp)
    if metrics:
        writer.add_text('final/full_test',
                        '\n'.join(f'{k}: {v}' for k, v in sorted(metrics.items())))
    writer.close()
    print(f'{exp}: {len(rows)} epochs -> {os.path.join(OUT_ROOT, exp)}')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default=None, help='only export this experiment')
    ap.add_argument('--force', action='store_true',
                    help='overwrite a directory that already has live TensorBoard data')
    args = ap.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)
    exps = [args.exp] if args.exp else sorted(os.listdir(os.path.join(ROOT, 'logs')))
    n = 0
    for exp in exps:
        if not os.path.isdir(os.path.join(ROOT, 'logs', exp)):
            continue
        if os.path.exists(os.path.join(OUT_ROOT, exp, 'LIVE')) and not args.force:
            print(f'{exp}: live TensorBoard exists, skip (use --force to overwrite)')
            continue
        if export(exp):
            n += 1
    print(f'exported {n} runs to {OUT_ROOT}')


if __name__ == '__main__':
    main()
