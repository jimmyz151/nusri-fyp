"""为每个实验结果绘制跨域长尾失败图：两个域各 4 张，共 8 张放在一张图上。

- 数据来自已保存的评估结果（per_sample_performance.mat / predictions_pose.mat），
  不需要重新推理。
- 每个域按旋转误差分 4 个严重度区间（30–60°、60–90°、90–150°、≥150°），
  每个区间取误差最大的样本；同域内按帧号保持间隔，避免选中同一段视频的近似帧。
- 输出 results/visualizations/<EXP>/longtail_failures.{png,pdf}
"""

import argparse
import re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.io import loadmat

import _init_paths  # noqa: F401
from utils.postprocess import project_keypoints, quat2dcm
from utils.utils import load_camera_intrinsics

ROOT = Path(__file__).resolve().parents[2]
AXIS_COLORS = ("#ff5555", "#51db83", "#559eff")
BANDS = ((30.0, 60.0), (60.0, 90.0), (90.0, 150.0), (150.0, 181.0))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="+", default=None,
                        help="实验名或 results/tables 下的路径；默认处理全部")
    parser.add_argument("--domains", nargs="+", default=["lightbox", "sunlamp"])
    parser.add_argument("--head", default="auto",
                        choices=["auto", "heatmap", "final", "refine"],
                        help="auto：有 refine 结果时用最终两阶段输出，否则用 heatmap")
    parser.add_argument("--per-domain", type=int, default=4)
    parser.add_argument("--min-gap", type=int, default=100,
                        help="同域选样时的最小帧号间隔")
    parser.add_argument("--axis-length", type=float, default=0.2, help="坐标轴长度，单位米")
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/opt/data/private/SPEED+/speedplusv2"))
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results" / "visualizations")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    if args.per_domain != len(BANDS):
        parser.error(f"--per-domain 目前必须为 {len(BANDS)}（对应 4 个误差区间）")
    runs = args.run
    if runs is None:
        table_root = ROOT / "results" / "tables"
        runs = [p.name for p in sorted(table_root.iterdir())
                if p.is_dir() and any((p / d).is_dir() for d in args.domains)]
    normalized = []
    for run in runs:
        path = Path(run)
        normalized.append(path if path.is_absolute() or path.exists() else ROOT / "results" / "tables" / run)
    args.run = normalized
    return args


def project_path(path):
    return path if path.is_absolute() else ROOT / path


def expand_per_sample(values, mask):
    """把只记录在 mask=True 样本上的指标展开到全部样本；已对齐的则直接返回。"""
    values = np.asarray(values, dtype=np.float64).ravel()
    out = np.full(mask.size, np.nan)
    if values.size == mask.sum():
        out[mask] = values
    elif values.size == mask.size:
        out[:] = values
    else:
        raise ValueError(f"per-sample 长度不匹配：{values.size} vs mask {mask.sum()}/{mask.size}")
    return out


def load_domain(run_dir, domain, dataset_root, head):
    perf = loadmat(run_dir / domain / "per_sample_performance.mat")
    pred = loadmat(run_dir / domain / "predictions_pose.mat")
    labels = pd.read_csv(dataset_root / domain / "labels" / f"{domain}.csv", header=None)
    n = len(labels)

    h_reject = perf["h_reject"].ravel()[:n].astype(bool)
    heat_er = expand_per_sample(perf["heat_eR"], ~h_reject)
    heat_et = expand_per_sample(perf["heat_eT"], ~h_reject)
    q_heat = np.asarray(pred["heat_q"], dtype=np.float64)[:n]
    t_heat = np.asarray(pred["heat_t"], dtype=np.float64)[:n]
    src = np.full(n, "heat", dtype=object)

    use_refine = head in ("auto", "final", "refine") and "ref_eR" in perf
    if head == "refine" and not use_refine:
        raise ValueError(f"{run_dir.name}/{domain} 没有 refine 结果")
    if use_refine:
        ref_reject = perf["ref_reject"].ravel()[:n].astype(bool)
        ref_er = expand_per_sample(perf["ref_eR"], ~ref_reject)
        ref_et = expand_per_sample(perf["ref_eT"], ~ref_reject)
        q_ref = np.asarray(pred["ref_q"], dtype=np.float64)[:n]
        t_ref = np.asarray(pred["ref_t"], dtype=np.float64)[:n]
        valid_ref = (~ref_reject) & np.isfinite(ref_er) & np.isfinite(q_ref).all(axis=1)
        if head == "refine":
            er, et = ref_er, ref_et
            q, t = q_ref, t_ref
            src[:] = "refine"
            valid = valid_ref
        else:
            use = valid_ref
            er = np.where(use, ref_er, heat_er)
            et = np.where(use, ref_et, heat_et)
            q = np.where(use[:, None], q_ref, q_heat)
            t = np.where(use[:, None], t_ref, t_heat)
            src[use] = "refine"
            valid = np.isfinite(er) & np.isfinite(q).all(axis=1) & np.isfinite(t).all(axis=1)
    else:
        er, et, q, t = heat_er, heat_et, q_heat, t_heat
        valid = (~h_reject) & np.isfinite(er) & np.isfinite(q).all(axis=1) \
            & np.isfinite(t).all(axis=1)

    iou = expand_per_sample(perf["segm_iou"], np.ones(n, dtype=bool)) \
        if "segm_iou" in perf else np.full(n, np.nan)
    translation = labels.iloc[:, 9:12].to_numpy(dtype=np.float64)
    distance = np.linalg.norm(translation, axis=1)
    gt_box_norm = labels.iloc[:, 1:5].to_numpy(dtype=np.float64)

    bbox = np.full((n, 4), np.nan)
    for key in ("bbox", "mask_bbox"):
        if key in pred and np.asarray(pred[key]).size:
            candidate = np.asarray(pred[key], dtype=np.float64)[:n]
            if candidate.shape == (n, 4):
                bbox = candidate
                break

    mean_pose = np.nan
    mean_path = run_dir / domain / "mean_performance.mat"
    if mean_path.exists():
        mean_mat = loadmat(mean_path)
        if "final_pose" in mean_mat:
            mean_pose = float(np.asarray(mean_mat["final_pose"]).squeeze())

    return dict(
        domain=domain, labels=labels, er=er, et=et, q=q, t=t, src=src, valid=valid,
        iou=iou, distance=distance, gt_box_norm=gt_box_norm, bbox=bbox,
        n=n, n_reject=int(h_reject.sum()), mean_pose=mean_pose,
        head=("final" if use_refine and head in ("auto", "final") else
              "refine" if head == "refine" else "heatmap"),
    )


def frame_number(name):
    digits = re.findall(r"\d+", str(name))
    return int(digits[-1]) if digits else None


def pick_samples(er, valid, names, min_gap):
    order = [int(i) for i in np.argsort(-np.nan_to_num(er, nan=-np.inf))
             if valid[i] and np.isfinite(er[i])]
    frames = [frame_number(name) if frame_number(name) is not None else index
              for index, name in enumerate(names)]

    def far_enough(index, picked):
        return all(abs(frames[index] - frames[other]) >= min_gap for other in picked)

    picked = []
    for lo, hi in BANDS:
        candidates = [i for i in order if lo <= er[i] < hi]
        choice = next((i for i in candidates if far_enough(i, picked)), None)
        if choice is None and candidates:
            choice = candidates[0]
        if choice is not None:
            picked.append(choice)
    # 某个区间样本不足时，用剩余最高误差样本补齐
    for index in order:
        if len(picked) >= len(BANDS):
            break
        if index not in picked and far_enough(index, picked):
            picked.append(index)
    return picked[:len(BANDS)]


def axis_pixels(q, t, camera, image_size, length):
    points = np.vstack((np.zeros(3), np.eye(3) * length))
    camera_points = quat2dcm(q).T @ points.T + np.asarray(t).reshape(3, 1)
    if not np.isfinite(camera_points).all() or np.any(camera_points[2] <= 0):
        return None
    pixels = project_keypoints(
        q, t, camera["cameraMatrix"], camera["distCoeffs"], points
    ).T
    pixels *= np.asarray(image_size) / np.array([camera["Nu"], camera["Nv"]])
    return pixels if np.isfinite(pixels).all() else None


def draw_axes(ax, pixels, predicted):
    if pixels is None:
        return
    origin = pixels[0]
    for endpoint, color in zip(pixels[1:], AXIS_COLORS):
        if predicted:
            ax.annotate("", xy=endpoint, xytext=origin,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                                        mutation_scale=10, shrinkA=0, shrinkB=0,
                                        path_effects=[pe.Stroke(linewidth=2.6,
                                                                foreground="#181818"),
                                                      pe.Normal()]),
                        zorder=6)
        else:
            ax.plot([origin[0], endpoint[0]], [origin[1], endpoint[1]],
                    color=color, lw=1.0, linestyle=(0, (3, 2)), zorder=7,
                    path_effects=[pe.Stroke(linewidth=2.4, foreground="#181818"),
                                  pe.Normal()])
    ax.plot(*origin, marker="o", ms=3, color="white", mec="#333333", mew=0.5, zorder=8)


def render(run_dir, domains, args, camera, output_dir):
    panels = []
    for domain in domains:
        if not (run_dir / domain).is_dir():
            continue
        panels.append(load_domain(run_dir, domain, args.dataset_root, args.head))
    if not panels:
        return None

    rows = len(panels)
    cols = args.per_domain
    fig_h = 3.25 * rows + 1.35
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
    fig, axes = plt.subplots(rows, cols, figsize=(13.2, fig_h),
                             facecolor="white", squeeze=False)
    # 标题、副标题、图例、行标题按英寸留白，单行/双行布局都不会重叠。
    fig.subplots_adjust(left=0.018, right=0.985, bottom=0.055,
                        top=1 - 1.15 / fig_h, wspace=0.06, hspace=0.30)

    has_pred_box = any(np.isfinite(p["bbox"]).any() for p in panels)
    legend = [
        Line2D([], [], color="#333333", lw=1.0, ls="--", label="GT pose"),
        Line2D([], [], color="#333333", lw=1.8, label="Pred. pose"),
        *[Line2D([], [], color=color, lw=2, label=f"${name}$")
          for color, name in zip(AXIS_COLORS, "XYZ")],
        Line2D([], [], color="#ffffff", lw=1.0, ls="--", label="GT box"),
    ]
    if has_pred_box:
        legend.append(Line2D([], [], color="#f2cb65", lw=1.2, label="Pred. box"))
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.48 / fig_h),
               frameon=False, ncol=len(legend), fontsize=9, handlelength=1.7,
               columnspacing=1.3)
    fig.suptitle(f"{run_dir.name} — cross-domain long-tail failures",
                 fontsize=13.5, fontweight="bold", y=0.995)
    fig.text(0.5, 1 - 0.26 / fig_h,
             "one sample per rotation-error band (30–60°, 60–90°, 90–150°, ≥150°), "
             "ordered by severity",
             ha="center", va="top", fontsize=9, color="#555555")

    for row, panel in enumerate(panels):
        picked = pick_samples(panel["er"], panel["valid"],
                              panel["labels"].iloc[:, 0].to_numpy(), args.min_gap)
        row_ax = axes[row, 0]
        head_name = panel["head"]
        row_ax.text(0.0, 1.025,
                    f"{panel['domain'].capitalize()}   "
                    f"n={panel['n']:,}   mean final pose={panel['mean_pose']:.3f}   "
                    f"rejects={panel['n_reject']}   [head: {head_name}]",
                    transform=row_ax.transAxes, va="bottom", ha="left",
                    fontsize=10, fontweight="bold", color="#222222")

        for col, index in enumerate(picked):
            ax = axes[row, col]
            tag = str(panel["labels"].iloc[index, 0])
            image = cv2.imread(str(args.dataset_root / panel["domain"]
                                   / "images_768x512_RGB" / tag))
            if image is None:
                raise FileNotFoundError(tag)
            height, width = image.shape[:2]
            gt_norm = panel["gt_box_norm"][index]
            gt_box = gt_norm * np.array([width, height, width, height])
            if not (np.isfinite(gt_box).all() and gt_box[2] > gt_box[0]
                    and gt_box[3] > gt_box[1]):
                gt_box = np.array([0, 0, width, height], dtype=np.float64)

            q_gt = panel["labels"].iloc[index, 5:9].to_numpy(dtype=np.float64)
            t_gt = panel["labels"].iloc[index, 9:12].to_numpy(dtype=np.float64)
            gt_axes = axis_pixels(q_gt, t_gt, camera, (width, height), args.axis_length)
            pred_axes = axis_pixels(panel["q"][index], panel["t"][index],
                                    camera, (width, height), args.axis_length)

            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation="nearest")
            gt_patch = Rectangle(gt_box[:2], gt_box[2] - gt_box[0], gt_box[3] - gt_box[1],
                                 fill=False, edgecolor="#ffffff", lw=1.0, linestyle="--",
                                 zorder=4)
            gt_patch.set_path_effects([pe.Stroke(linewidth=2.0, foreground="#181818"),
                                       pe.Normal()])
            ax.add_patch(gt_patch)

            pred_box = panel["bbox"][index]
            if np.isfinite(pred_box).all():
                if np.nanmax(pred_box) <= 1.5:  # mask_bbox 为归一化坐标
                    pred_box = pred_box * np.array([width, height, width, height])
                if pred_box[2] > pred_box[0] and pred_box[3] > pred_box[1]:
                    pred_patch = Rectangle(pred_box[:2], pred_box[2] - pred_box[0],
                                           pred_box[3] - pred_box[1], fill=False,
                                           edgecolor="#f2cb65", lw=1.2, zorder=5)
                    pred_patch.set_path_effects([pe.Stroke(linewidth=2.0,
                                                          foreground="#181818"),
                                                 pe.Normal()])
                    ax.add_patch(pred_patch)

            draw_axes(ax, pred_axes, predicted=True)
            draw_axes(ax, gt_axes, predicted=False)

            center = (gt_box[:2] + gt_box[2:]) / 2
            side = max(gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]) * 3.0
            side = min(max(side, 48), min(width, height))
            half = side / 2
            # 窗口保持正方形；靠近图像边缘时平移窗口而不是缩小，保留场景上下文。
            cx = np.clip(center[0], half, width - half)
            cy = np.clip(center[1], half, height - half)
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy + half, cy - half)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            letter = f"({chr(97 + row * cols + col)})"
            ax.text(0.025, 0.975, letter, transform=ax.transAxes, va="top", ha="left",
                    fontsize=11, fontweight="bold", color="white", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.22", fc=(0, 0, 0, 0.45), ec="none"))

            note = []
            if pred_axes is None:
                note.append("proj. failed")
            if panel["src"][index] == "refine":
                note.append("refine")
            line2 = f"{tag}  ·  {panel['distance'][index]:.1f} m"
            box_px = np.sqrt(max(gt_box[2] - gt_box[0], 0) *
                             max(gt_box[3] - gt_box[1], 0))
            line2 += f"  ·  {box_px:.0f} px"
            if note:
                line2 += "  ·  " + "  ·  ".join(note)
            ax.text(0.0, -0.035,
                    rf"$e_R$ {panel['er'][index]:.1f}$^\circ$   "
                    rf"$e_t$ {panel['et'][index]:.2f} m   "
                    rf"mIoU {panel['iou'][index]:.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9)
            ax.text(0.0, -0.115, line2, transform=ax.transAxes, va="top", ha="left",
                    fontsize=7.5, color="#555555")

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = output_dir / f"longtail_failures.{extension}"
        fig.savefig(path, dpi=args.dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output_dir / "longtail_failures.png"


def main():
    args = parse_args()
    camera = load_camera_intrinsics(args.dataset_root / "camera.json")
    for run in args.run:
        run_dir = project_path(run)
        if not run_dir.is_dir():
            print(f"跳过不存在的实验目录：{run_dir}")
            continue
        output = render(run_dir, args.domains, args, camera, args.output_root / run_dir.name)
        if output is None:
            print(f"跳过没有评估结果的实验：{run_dir.name}")
        else:
            print(f"图已保存：{output}")


if __name__ == "__main__":
    main()
