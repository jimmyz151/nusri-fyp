# SPNv2_baseline_v1_261113

- 模型：SPNv2（EfficientDet-D3，phi=3，BN backbone / GN heads）
- 实验类型：baseline（online style 在前，像素增强在后）
- 日期：2026-11-13
- 配置快照：`config.yaml`（源：`configs/baseline.yaml`）
- 权重：`checkpoints/SPNv2_baseline_v1_261113/`
- 日志：`logs/SPNv2_baseline_v1_261113/`
- 结果：`results/tables/SPNv2_baseline_v1_261113/{lightbox,sunlamp}/`
- 指标汇总：`metrics.csv`
- 说明：训练 20 epoch，输入 768x512，测试 SPEED+ HIL 的 lightbox / sunlamp 两个域。

## 合成域结果来源

`metrics.csv` 中的 `synthetic` 块来自训练期验证日志，不是独立的全量测试：

- 数据：`synthetic/labels/validation.csv`，训练期只取 `TRAIN.VALID_FRACTION=0.1`（约前 1,199 张）
- 权重：`model_best.pth.tar` 对应的 best epoch（按合成域 `final_pose` 选优）
- best epoch：20（2026/11/14 04:35:25）
- 日志：`logs/SPNv2_baseline_v1_261113/train_20261113_10_41_27/train_rank0.log`
- 可用指标：`heat_eR`、`effi_eT`（efficientpose 头）、`final_pose`；未记录 eT/HIL/IoU/reject/segm_iou，故 `synthetic` 行不完整
- `final_pose` 定义：多任务模型为 `heat_eR_HIL + effi_eT_norm_HIL`
- 数值精度：日志保留 3 位小数
