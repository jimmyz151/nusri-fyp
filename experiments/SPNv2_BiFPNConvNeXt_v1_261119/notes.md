# SPNv2_BiFPNConvNeXt_v1_261119

- 模型：ConvNeXt-T + **原版 BiFPN neck**（C2~C5 输入，输出 P2~P7）+ `heatmap` / `segmentation` 两头；**不加 EfficientPose 检测头**
- 实验类型：骨干替换（EfficientNet-D3 → ConvNeXt-T），保留多尺度 neck；mask 头兼任检测并驱动两阶段裁剪
- 日期：2026-11-19
- 配置快照：`config.yaml`（s2/最终）、`config_s1.yaml`（s1）；源码在 `configs/`
- 权重：`checkpoints/SPNv2_BiFPNConvNeXt_v1_261119/`
- 日志：`logs/SPNv2_BiFPNConvNeXt_v1_261119/`
- 结果：`results/tables/SPNv2_BiFPNConvNeXt_v1_261119/{lightbox,sunlamp}/`
- 指标汇总：`metrics.csv`

## 为什么保留 BiFPN

无 neck 对照试验（已删除，仅保留结论）的合成域同期验证结果：

| epoch | baseline heat_eR | 无 neck heat_eR |
| --- | --- | --- |
| 1 | 3.57° | 103.08° |
| 4 | 1.69° | 56.59° |
| 8 | 1.38° | 28.68° |

热图分支基本没学到关键点（`hmap` loss 卡在 9.5，接近全背景下限），而分割正常收敛，说明问题在 P2 缺少 P3~P7 的多尺度上下文，而不是数据或检测头。因此本实验恢复 BiFPN。

## 结构

- 骨干：`convnext_bifpn_tiny`
  - ConvNeXt-T stem/stage1~4 → C2/C3/C4/C5（96/192/384/768，stride 4/8/16/32）
  - 作为 `[None, C2, C3, C4, C5]` 输入现有 `BiFPNLayer`；P6/P7 由 C5 下采样生成
  - 输出 P2~P7，宽度 160（phi=3），深度 4
- 头部：`heatmap`（192×128，UDP）+ `segmentation`（192×128）读 P2；无 `efficientpose`
- 检测：mask 最大连通域外接框（`segm_det_iou`），并驱动 stage-2 裁剪
- 两阶段推理：全帧 → mask 框 → 3:2 裁剪原图 → 同一权重第二次前向 → `ref_*`；失败回退 stage-1

## 与 baseline 的差异

- 骨干：EfficientNet-D3 → ConvNeXt-T（BiFPN 结构保持不变）
- 头：heatmap + efficientpose + segmentation → heatmap + segmentation
- 检测：EfficientPose 框回归 → mask 外接框
- 推理：单次全帧 → mask 框驱动的两次前向
- 热图：`DATASET.UDP: True`

## 训练流程

| 阶段 | 配置 | 初始化 | 说明 |
| --- | --- | --- | --- |
| s1 | `configs/SPNv2_BiFPNConvNeXt_v1_261119_s1.yaml` | ImageNet ConvNeXt-T | 全帧 20 epoch |
| 统计 | `src/tools/bbox_error_stats.py` | s1 | mask 框误差分位数，定 s2 抖动 |
| s2 | `configs/SPNv2_BiFPNConvNeXt_v1_261119.yaml` | s1 best | 全帧 + 误差模拟裁剪混合 20 epoch |

## 复现

s1：

```bash
cd /opt/data/private/speNet
unset LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl \
/opt/data/private/venvs/zz_fyp/bin/python src/train.py \
  --cfg configs/SPNv2_BiFPNConvNeXt_v1_261119_s1.yaml \
  EXP_NAME SPNv2_BiFPNConvNeXt_v1_261119 AUTO_RESUME False VERBOSE True
```

s1 后备份权重：

```bash
cp checkpoints/SPNv2_BiFPNConvNeXt_v1_261119/model_best.pth.tar \
   checkpoints/SPNv2_BiFPNConvNeXt_v1_261119/stage1_model_best.pth.tar
```

统计框误差：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/data/private/venvs/zz_fyp/bin/python src/tools/bbox_error_stats.py \
  --cfg configs/SPNv2_BiFPNConvNeXt_v1_261119_s1.yaml \
  EXP_NAME SPNv2_BiFPNConvNeXt_v1_261119 TEST.TEST_CSV synthetic/labels/validation.csv
```

s2：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl \
/opt/data/private/venvs/zz_fyp/bin/python src/train.py \
  --cfg configs/SPNv2_BiFPNConvNeXt_v1_261119.yaml \
  EXP_NAME SPNv2_BiFPNConvNeXt_v1_261119 AUTO_RESUME False VERBOSE True
```

测试：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
/opt/data/private/venvs/zz_fyp/bin/python src/tools/test.py \
  --cfg configs/SPNv2_BiFPNConvNeXt_v1_261119.yaml EXP_NAME SPNv2_BiFPNConvNeXt_v1_261119 VERBOSE True \
  TEST.TEST_CSV lightbox/labels/lightbox.csv
```

sunlamp 同理替换 `TEST.TEST_CSV`。

## 判据

- 同期合成域：第 1 轮 `heat_eR` 应回到个位数（baseline 3.57°），第 8 轮接近 baseline 的 1.38°
- 最终目标：超过 baseline 的 `final_pose`（lightbox 0.153 / sunlamp 0.270）
- 观察 `segm_det_iou`、`ref_pose` 相对 `heat_pose` 的增益、`# fallback` 比例
