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
