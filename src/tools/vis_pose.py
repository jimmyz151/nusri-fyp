"""可视化验证集前四张图像，导出论文排版用的 2x2 PNG / PDF。"""

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
import torch

import _init_paths  # noqa: F401
from config import cfg, update_config
from dataset.build import build_dataset
from nets import build_spnv2
from utils.metrics import error_orientation, error_translation, segment_iou
from utils.postprocess import project_keypoints, quat2dcm, solve_pose_from_heatmaps
from utils.utils import load_camera_intrinsics, load_tango_3d_keypoints


ROOT = Path(__file__).resolve().parents[2]
AXIS_COLORS = ("#ff5555", "#51db83", "#559eff")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default="configs/baseline.yaml",
                        help="配置路径，相对路径以项目根目录为基准")
    parser.add_argument("--checkpoint", type=Path, help="默认读取实验的 model_best.pth.tar")
    parser.add_argument("--output-dir", type=Path, help="默认输出到实验目录下 vis_pose")
    parser.add_argument("--axis-length", type=float, default=0.2, help="坐标轴长度，单位米")
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda:0")
    parser.add_argument("--with-mask", action="store_true",
                        help="把分割头加入 TEST.HEAD，图上叠加预测掩膜与 GT 掩膜，并给出 mask IoU")
    args = parser.parse_args()
    if not np.isfinite(args.axis_length) or args.axis_length <= 0:
        parser.error("--axis-length 必须为正数")
    args.cfg = str(project_path(Path(args.cfg)))
    args.opts = []
    return args


def project_path(path):
    return path if path.is_absolute() else ROOT / path


def axis_pixels(q, t, camera, image_size, length):
    points = np.vstack((np.zeros(3), np.eye(3) * length))
    # 与官方 project_keypoints 一致：物体系到相机系使用 quat2dcm(q).T。
    camera_points = quat2dcm(q).T @ points.T + np.asarray(t).reshape(3, 1)
    if not np.isfinite(camera_points).all() or np.any(camera_points[2] <= 0):
        return None
    pixels = project_keypoints(
        q, t, camera["cameraMatrix"], camera["distCoeffs"], points
    ).T
    # PnP 和投影均使用原始相机分辨率，最后才缩放到展示图片。
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
                                        mutation_scale=10, shrinkA=0, shrinkB=0),
                        zorder=5)
        else:
            # 虚线与实线编码姿态来源，颜色只编码轴方向。
            ax.plot([origin[0], endpoint[0]], [origin[1], endpoint[1]],
                    color=color, lw=1.0, linestyle=(0, (3, 2)), zorder=6,
                    path_effects=[pe.Stroke(linewidth=2.3, foreground="#181818"),
                                  pe.Normal()])
    ax.plot(*origin, marker="o", ms=3, color="white", mec="#333333", mew=0.5, zorder=7)


def crop_bounds(sample):
    # 裁剪仅影响展示；推理与误差计算仍使用完整图像和原始内参。
    boxes = np.stack((sample["gt_bbox"], sample["bbox"]))
    points = [boxes[:, :2], boxes[:, 2:]]
    points += [sample[key] for key in ("gt_axes", "pred_axes") if sample[key] is not None]
    points = np.concatenate(points)
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2
    side = max(hi - lo) * 1.28
    return center - side / 2, center + side / 2


MASK_FILL_RGB = np.array([80, 220, 120], dtype=np.float32)


def overlay_mask(image, pred_mask, gt_mask):
    """把 192x128 的二值掩膜放大到原图：预测半透明填充 + 绿描边，GT 白描边。"""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    size = (rgb.shape[1], rgb.shape[0])
    if pred_mask is not None:
        fill = cv2.resize(pred_mask.astype(np.uint8), size,
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        rgb[fill] = (0.55 * rgb[fill] + 0.45 * MASK_FILL_RGB).astype(np.uint8)
    for mask, color, thickness in ((gt_mask, (255, 255, 255), 1),
                                   (pred_mask, (40, 190, 90), 2)):
        if mask is None:
            continue
        binary = cv2.resize(mask.astype(np.uint8), size,
                            interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, color, thickness)
    return rgb


def write_outputs(samples, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
    has_masks = samples[0].get("mask") is not None
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 8.1 if has_masks else 7.8),
                             facecolor="white")
    # 每个子图下方留出统计文字安全带；hspace 保证上一行的文字不与下一行图像相撞。
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.085,
                        top=0.885 if has_masks else 0.90,
                        wspace=0.05, hspace=0.26)
    legend = [
        Line2D([], [], color="#333333", lw=1, ls="--", label="Ground truth"),
        Line2D([], [], color="#333333", lw=1.8, label="Prediction"),
        *[Line2D([], [], color=color, lw=2, label=f"${name}$")
          for color, name in zip(AXIS_COLORS, "XYZ")],
        Line2D([], [], color="#f2cb65", lw=1, label="Pred. box"),
        Line2D([], [], color="#888888", lw=1, ls="--", label="GT box"),
    ]
    if has_masks:
        legend += [Line2D([], [], color="#28be5a", lw=2, label="Pred. mask"),
                   Line2D([], [], color="#aaaaaa", lw=1, label="GT mask")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               frameon=False, ncol=4, fontsize=9, handlelength=1.8,
               columnspacing=1.4, labelspacing=0.5)
    for index, (ax, sample) in enumerate(zip(axes.flat, samples)):
        ax.set_facecolor("black")
        display = (overlay_mask(sample["image"], sample["mask"], sample["gt_mask"])
                   if has_masks
                   else cv2.cvtColor(sample["image"], cv2.COLOR_BGR2RGB))
        ax.imshow(display, interpolation="nearest")
        for bbox, color, style in ((sample["gt_bbox"], "#dddddd", "--"),
                                   (sample["bbox"], "#f2cb65", "-")):
            x1, y1, x2, y2 = bbox
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=color, lw=0.8, linestyle=style))
        draw_axes(ax, sample["pred_axes"], predicted=True)
        draw_axes(ax, sample["gt_axes"], predicted=False)
        lo, hi = crop_bounds(sample)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(hi[1], lo[1])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # 面板编号放进图内（带半透明底衬），不与下方统计文字争位置。
        ax.text(0.025, 0.975, f"({chr(97 + index)})", transform=ax.transAxes,
                va="top", ha="left", fontsize=11, fontweight="bold",
                color="white", zorder=8,
                bbox=dict(boxstyle="round,pad=0.25", fc=(0, 0, 0, 0.45), ec="none"))
        # 统计文字左右分列对齐，各自留在本列宽度内，避免跨列重叠。
        pose_text = (rf"$e_R={sample['er']:.2f}^\circ$   "
                     rf"$e_t={sample['et']:.3f}\,\mathrm{{m}}$"
                     if sample["pred_axes"] is not None else "Pose unavailable")
        ax.text(0.0, -0.05, pose_text, transform=ax.transAxes, va="top",
                ha="left", fontsize=9)
        if has_masks:
            ax.text(1.0, -0.05, rf"mask IoU$={sample['iou']:.3f}$",
                    transform=ax.transAxes, va="top", ha="right", fontsize=9)
    stem = "val_pose_mask_2x2" if has_masks else "val_pose_2x2"
    for extension in ("png", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=450, facecolor="white")
        print(f"演示图：{path}", flush=True)
    plt.close(fig)


def main():
    args = parse_args()
    update_config(cfg, args)
    cfg.defrost()
    cfg.TEST.HEAD = ["heatmap", "efficientpose"] + (["segmentation"] if args.with_mask else [])
    cfg.MODEL.PRETRAIN_FILE = None
    cfg.freeze()
    experiment_dir = project_path(Path(cfg.OUTPUT_DIR)) / cfg.EXP_NAME
    checkpoint = (project_path(args.checkpoint) if args.checkpoint
                  else experiment_dir / "model_best.pth.tar")
    output_dir = (project_path(args.output_dir) if args.output_dir
                  else project_path(Path(cfg.RESULT_DIR)) / "visualizations" / cfg.EXP_NAME)
    device = torch.device(
        ("cuda:0" if cfg.CUDA and torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    torch.set_num_threads(4)
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    cv2.setRNGSeed(cfg.SEED)
    print(f"权重：{checkpoint}\n推理设备：{device}", flush=True)
    model = build_spnv2(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()

    dataset = build_dataset(cfg, split="val", load_labels=True)
    camera = load_camera_intrinsics(cfg.DATASET.CAMERA)
    keypoints = load_tango_3d_keypoints(cfg.DATASET.KEYPOINTS)
    count = 4
    if len(dataset) < count:
        raise ValueError("2x2 演示图需要至少四张验证图像")
    samples = []
    with torch.inference_mode():
        for index in range(count):
            image_tensor, target = dataset[index]
            filename = str(dataset.csv.iloc[index, 0])
            image_path = Path(dataset.root) / dataset.domain / dataset.imagefolder / filename
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            image_size = (image.shape[1], image.shape[0])
            if image_size != tuple(cfg.DATASET.INPUT_SIZE):
                raise ValueError(f"{filename} 尺寸 {image_size} 与配置 INPUT_SIZE 不一致")
            head_out = dict(zip(cfg.TEST.HEAD,
                                model(image_tensor.unsqueeze(0), is_train=False, gpu=device)))
            heatmap, efficientpose = head_out["heatmap"], head_out["efficientpose"]
            classification, boxes, _, translations = efficientpose
            best = int(classification[0].reshape(-1).argmax().item())
            bbox = boxes[0, best].cpu().numpy()
            translation = translations[0, best].cpu().numpy()
            # 与 do_valid 相同：热力图坐标先还原到 IMAGE_SIZE 再解 PnP。
            _, quaternion, _, rejected = solve_pose_from_heatmaps(
                heatmap[0].cpu(), cfg.DATASET.IMAGE_SIZE,
                cfg.TEST.HEATMAP_THRESHOLD, camera, keypoints
            )
            q_gt = target["quaternion"].numpy()
            t_gt = target["translation"].numpy()
            pixels = None
            er = et = float("nan")
            if rejected:
                status = "PnP rejected: prediction axes omitted"
            else:
                pixels = axis_pixels(quaternion, translation, camera, image_size, args.axis_length)
                er = error_orientation(quaternion, q_gt, "quaternion")
                et = error_translation(translation, t_gt)
                status = f"eR={er:.3f} deg   eT={et:.3f} m"
                if pixels is None:
                    status += " | invalid projection"
            pred_mask = gt_mask = None
            iou = float("nan")
            if args.with_mask:
                # 与 do_valid 一致：分割头输出 [1, 1, 192, 128] logits，阈值 0.5 后与 GT 掩膜算 IoU。
                # GT 掩膜直接向数据集取，因为 SPEEDPLUSDataset 只在 is_train 时把 mask 放进 targets。
                pred_mask = head_out["segmentation"][0, 0].sigmoid().cpu().numpy() > 0.5
                gt_mask = dataset._load_mask(index)[:, :, 0] > 0
                iou = segment_iou(pred_mask, gt_mask)
                status += f"   maskIoU={iou:.3f}"
            samples.append(dict(
                image=image, bbox=bbox, gt_bbox=target["boundingbox"].numpy(),
                gt_axes=axis_pixels(q_gt, t_gt, camera, image_size, args.axis_length),
                pred_axes=pixels, er=er, et=et,
                mask=pred_mask, gt_mask=gt_mask, iou=iou,
            ))
            print(f"[{index + 1}/{count}] {filename}: {status}", flush=True)
    write_outputs(samples, output_dir)


if __name__ == "__main__":
    main()
#  vis_pose.py