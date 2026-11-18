"""可视化 HIL 域的失败样本：GT 姿态（虚线轴）与预测姿态（实线箭头）叠加。

样本取自已保存的评估结果（per_sample_performance.mat / predictions_pose.mat），
无需重新推理。默认挑出"官方旋转误差 < 10 度、自己 > 30 度"的样本，
按误差区间分层抽样，避免只看到最严重的 180 度样本。
"""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=Path("results/tables/SPNv2_baseline_v1_261113"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("results/tables/official_offline"))
    parser.add_argument("--domain", default="sunlamp")
    parser.add_argument("--num", type=int, default=12, help="子图数量，需能被 3 整除")
    parser.add_argument("--axis-length", type=float, default=0.2, help="坐标轴长度，单位米")
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/opt/data/private/SPEED+/speedplusv2"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.num % 3:
        parser.error("--num 需能被 3 整除")
    for name in ("run", "baseline"):
        path = getattr(args, name)
        setattr(args, name, path if path.is_absolute() else ROOT / path)
    if args.output is None:
        args.output = (ROOT / "results" / "visualizations" / args.run.name
                       / f"{args.domain}_failures.png")
    elif not args.output.is_absolute():
        args.output = ROOT / args.output
    return args


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
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                        mutation_scale=9, shrinkA=0, shrinkB=0),
                        zorder=5)
        else:
            ax.plot([origin[0], endpoint[0]], [origin[1], endpoint[1]],
                    color=color, lw=0.9, linestyle=(0, (3, 2)), zorder=6,
                    path_effects=[pe.Stroke(linewidth=2.0, foreground="#181818"),
                                  pe.Normal()])


def load_run(directory, domain):
    perf = loadmat(directory / domain / "per_sample_performance.mat")
    pred = loadmat(directory / domain / "predictions_pose.mat")
    reject = perf["h_reject"].ravel().astype(bool)
    rot = np.full(len(reject), np.nan)
    rot[~reject] = perf["heat_eR"].ravel()
    return dict(
        rot=rot, iou=perf["effi_iou"].ravel(), trans=perf["effi_eT"].ravel(),
        heat_q=np.asarray(pred["heat_q"], dtype=np.float64),
        effi_t=np.asarray(pred["effi_t"], dtype=np.float64),
        bbox=np.asarray(pred["bbox"], dtype=np.float64),
    )


def pick(mine, base, count):
    """官方好、自己坏；按旋转误差区间分层抽样。"""
    candidates = np.isfinite(mine["rot"]) & np.isfinite(base["rot"])
    candidates &= (base["rot"] < 10) & (mine["rot"] > 30)
    bins = [(30, 60), (60, 90), (90, 120), (120, 150), (150, 181)]
    rng = np.random.default_rng(0)
    pools = [candidates & (mine["rot"] >= lo) & (mine["rot"] < hi) for lo, hi in bins]
    picked = []
    while len(picked) < count:
        added = False
        for pool in pools:
            if len(picked) >= count:
                break
            rest = np.setdiff1d(np.where(pool)[0], picked)
            if rest.size:
                picked.append(int(rng.choice(rest)))
                added = True
        if not added:  # 所有区间都取完时用剩余样本补齐
            rest = np.setdiff1d(np.where(candidates)[0], picked)
            picked.extend(rest[:count - len(picked)].tolist())
            break
    return picked


def main():
    args = parse_args()
    camera = load_camera_intrinsics(args.dataset_root.parent / "speedplusv2" / "camera.json")
    csv = args.dataset_root / args.domain / "labels" / f"{args.domain}.csv"
    labels = pd.read_csv(csv, header=None)
    mine = load_run(args.run, args.domain)
    base = load_run(args.baseline, args.domain)

    picked = pick(mine, base, args.num)
    rows = args.num // 3
    fig, axes = plt.subplots(rows, 3, figsize=(11.0, 3.35 * rows), facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.06, top=0.90,
                        wspace=0.03, hspace=0.16)
    legend = [
        Line2D([], [], color="#333333", lw=0.9, ls="--", label="GT axes"),
        Line2D([], [], color="#333333", lw=1.6, label="Predicted axes"),
        *[Line2D([], [], color=c, lw=2, label=f"${n}$")
          for c, n in zip(AXIS_COLORS, "XYZ")],
        Line2D([], [], color="#f2cb65", lw=1, label="Pred. box"),
        Line2D([], [], color="#dddddd", lw=1, ls="--", label="GT box"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               frameon=False, ncol=7, fontsize=10, handlelength=1.8)

    print(f"{'filename':<18}{'yours eR':>10}{'offi eR':>9}{'IoU':>7}{'eT':>7}{'range':>8}")
    for ax, index in zip(axes.flat, picked):
        tag = str(labels.iloc[index, 0])
        image = cv2.imread(str(args.dataset_root / args.domain / "images_768x512_RGB" / tag))
        height, width = image.shape[:2]
        q_gt = labels.iloc[index, 5:9].to_numpy(float)
        t_gt = labels.iloc[index, 9:12].to_numpy(float)
        gt_box = labels.iloc[index, 1:5].to_numpy(float) * [width, height, width, height]
        pred_box = mine["bbox"][index].reshape(4) * [width, height, width, height]
        q_pr, t_pr = mine["heat_q"][index], mine["effi_t"][index]
        gt_axes = axis_pixels(q_gt, t_gt, camera, (width, height), args.axis_length)
        pr_axes = axis_pixels(q_pr, t_pr, camera, (width, height), args.axis_length)

        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation="nearest")
        for box, color, style in ((gt_box, "#dddddd", "--"), (pred_box, "#f2cb65", "-")):
            ax.add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False,
                                   edgecolor=color, lw=0.9, linestyle=style))
        draw_axes(ax, pr_axes, predicted=True)
        draw_axes(ax, gt_axes, predicted=False)
        # 裁剪以 GT 框为锚点：预测姿态错到 180 度时坐标轴会飞出画面，
        # 若把坐标轴纳入包围盒会把图像缩成一个点。
        gt_center = (gt_box[:2] + gt_box[2:]) / 2
        side = max(gt_box[2:] - gt_box[:2]) * 2.6
        side = min(side, min(width, height))
        lo = np.clip(gt_center - side / 2, 0, [width, height])
        hi = np.clip(gt_center + side / 2, 0, [width, height])
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(hi[1], lo[1])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.02, 0.98, f"{tag}", transform=ax.transAxes, va="top", ha="left",
                fontsize=8, color="white", zorder=8,
                bbox=dict(boxstyle="round,pad=0.22", fc=(0, 0, 0, 0.5), ec="none"))
        ax.text(0.0, -0.03,
                f"yours $e_R$={mine['rot'][index]:.1f}$^\\circ$  "
                f"$e_t$={mine['trans'][index]:.2f}m  IoU={mine['iou'][index]:.2f}\n"
                f"official $e_R$={base['rot'][index]:.1f}$^\\circ$  "
                f"range={np.linalg.norm(t_gt):.1f}m",
                transform=ax.transAxes, va="top", ha="left", fontsize=8)
        print(f"{tag:<18}{mine['rot'][index]:>10.1f}{base['rot'][index]:>9.1f}"
              f"{mine['iou'][index]:>7.2f}{mine['trans'][index]:>7.2f}"
              f"{np.linalg.norm(t_gt):>8.1f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"\n图已保存：{args.output}")


if __name__ == "__main__":
    main()
