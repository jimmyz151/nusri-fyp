# official_offline

- 模型：SPNv2 官方离线权重 `spnv2_efficientnetb3_fullconfig_offline.pth.tar`
- 实验类型：official baseline 评测（不训练）
- 配置快照：`config.yaml`（源：`configs/official_phi3_bn.yaml`；评测时以 `TEST.MODEL_FILE` 指向官方权重）
- 权重：`checkpoints/pretrained/spnv2_efficientnetb3_fullconfig_offline.pth.tar`
- 日志：`logs/official_offline/`
- 结果：`results/tables/official_offline/{synthetic,lightbox,sunlamp}/`
- 指标汇总：`metrics.csv`
- 说明：synthetic 为验证集，lightbox / sunlamp 为 SPEED+ HIL 域。
- 备注：本实验的 `synthetic` 为完整验证集（11,994 张）独立测试结果；其余训练实验 `metrics.csv` 中的 `synthetic` 块来自训练期 10% 验证日志，口径不同，不可直接比较。
