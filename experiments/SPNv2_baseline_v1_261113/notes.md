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
