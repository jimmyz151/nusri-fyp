# SPNv2_UDP_v1_261119

- 模型：SPNv2（EfficientDet-D3，phi=3，BN backbone / GN heads）——与 baseline 完全一致
- 实验类型：ablation（baseline + UDP 亚像素热图编码/解码）
- 日期：2026-11-19
- 配置快照：`config.yaml`（源：`configs/SPNv2_UDP_v1_261119.yaml`）
- 权重：`checkpoints/SPNv2_UDP_v1_261119/`
- 日志：`logs/SPNv2_UDP_v1_261119/`
- 结果：`results/tables/SPNv2_UDP_v1_261119/{lightbox,sunlamp}/`
- 指标汇总：`metrics.csv`
- 说明：全帧输入 768x512，训练 20 epoch。相对 baseline 的唯一变量是 `DATASET.UDP: True`：
  - 训练端：`HeatmapGenerator` 用连续亚像素位置作为高斯中心（不再 `int()` 截断）
  - 测试端：`solve_pose_from_heatmaps` 用 3 点抛物线做半像素修正（不再是整数 argmax）
- 对照组：`SPNv2_baseline_v1_261113`（同配置，`DATASET.UDP: False`）

## 有效训练 run

`logs/SPNv2_UDP_v1_261119/train_20261120_22_29_05`（2026-11-20 22:29 ~ 2026-11-21 07:00，20 epoch）。

- 双卡 DDP，`TRAIN.IMAGES_PER_GPU=8`
- 合成验证集最优（epoch 20，即 `checkpoints/SPNv2_UDP_v1_261119/model_best.pth.tar`）：
  `heat_eR 0.652 deg` / `effi_eT 0.048 m` / `final_pose 0.020`
- 对比 baseline 同口径验证集：`heat_eR 0.830 deg` / `effi_eT 0.046 m` / `final_pose 0.023`

## 真实域结果

测试权重均为 `checkpoints/SPNv2_UDP_v1_261119/model_best.pth.tar`（2026-11-21 07:00 保存）；
lightbox 于 2026-11-21 12:49 测（当前代码），sunlamp 于 2026-11-21 10:18 测。

| 指标 | baseline 261113 | UDP v1 261119 |
| --- | --- | --- |
| lightbox heat_eR [deg] | 6.92 | 12.61 |
| lightbox heat final_pose | 0.153 | 0.279 |
| lightbox heat reject | 2 | 12 |
| lightbox efficientpose pose | 0.180 | 0.318 |
| lightbox segmentation IoU | 0.898 | 0.864 |
| sunlamp heat_eR [deg] | 12.97 | 15.17 |
| sunlamp heat final_pose | 0.270 | 0.309 |
| sunlamp efficientpose pose | 0.294 | 0.326 |
| sunlamp segmentation IoU | 0.879 | 0.879 |

## 结论与注意事项

- **本次对比不是干净的单变量对比**：该 run 的命令行/实际配置为 `DIST.SYNCBN: False`，
  见 `logs/SPNv2_UDP_v1_261119/train_20261120_22_29_05/config.txt` 与 `train_rank0.log`
  中的 `SyncBN disabled: BatchNorm uses per-GPU statistics`；而 baseline 261113 是
  `SyncBN activated`（`train_rank0.log`）。两者 BN 统计量口径不同（每卡 batch=8 vs 跨卡等效 16）。
- 因此不能据此判定“UDP 拖累跨域泛化”。合成域验证集 UDP 反而更好（0.652 vs 0.830 deg），
  真实域却全面变差，且不消费热图目标的 segmentation IoU 也下降，符合 BN 统计量跨域偏移的特征，
  SyncBN 是主要混淆变量。
- UDP 的收益上限本身很小：热图 192x128（stride 4），一个 cell 对应原图 10 px；
  抛物线最多修正半个 cell，即原图约 5 px。
- 干净对比需用 `configs/SPNv2_UDP_v2_261121.yaml`（`DIST.SYNCBN: True`，其余不变）重训。

## 复现

训练（v1 原始命令，含污染项，仅留档）：

```bash
cd /opt/data/private/speNet
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl \
/opt/data/private/venvs/zz_fyp/bin/python src/train.py \
  --cfg configs/SPNv2_UDP_v1_261119.yaml \
  EXP_NAME SPNv2_UDP_v1_261119 AUTO_RESUME False VERBOSE True DIST.SYNCBN False
```

测试：

```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl \
/opt/data/private/venvs/zz_fyp/bin/python src/tools/test.py \
  --cfg configs/SPNv2_UDP_v1_261119.yaml \
  EXP_NAME SPNv2_UDP_v1_261119 VERBOSE True TEST.TEST_CSV lightbox/labels/lightbox.csv
```

sunlamp 把 `TEST.TEST_CSV` 换成 `sunlamp/labels/sunlamp.csv`。

## 合成域结果来源

`metrics.csv` 中的 `synthetic` 块来自训练期验证日志，不是独立的全量测试：

- 数据：`synthetic/labels/validation.csv`，训练期只取 `TRAIN.VALID_FRACTION=0.1`（约前 1,199 张）
- 权重：`model_best.pth.tar` 对应的 best epoch（按合成域 `final_pose` 选优）
- best epoch：20（2026/11/21 07:00:40）
- 日志：`logs/SPNv2_UDP_v1_261119/train_20261120_22_29_05/train_rank0.log`
- 可用指标：`heat_eR`、`effi_eT`（efficientpose 头）、`final_pose`；未记录 eT/HIL/IoU/reject/segm_iou，故 `synthetic` 行不完整
- `final_pose` 定义：多任务模型为 `heat_eR_HIL + effi_eT_norm_HIL`
- 数值精度：日志保留 3 位小数
