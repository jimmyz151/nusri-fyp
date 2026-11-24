# probe_bnneck（BiFPN GroupNorm → BatchNorm）

- 父实验：`SPNv2_BiFPNConvNeXt_v2_261122`
- 目的：判断 v2 neck 的 GN 归一化是否是真实域退化的主因
- 配置：`configs/SPNv2_BiFPNConvNeXt_v2_261122_probe_bnneck.yaml`（快照 `config.yaml`）
- 权重：`checkpoints/SPNv2_BiFPNConvNeXt_v2_261122_probe_bnneck/`
- 训练日志：`logs/SPNv2_BiFPNConvNeXt_v2_261122_probe_bnneck/`
- 结果：`results/tables/SPNv2_BiFPNConvNeXt_v2_261122_probe_bnneck_sub400/sunlamp/`
- 记录：`experiments/probes/probe_bnneck/`（不再挂在父实验目录下）
- 评测集：`sunlamp/labels/sunlamp_probe400.csv`（400 张固定子集）

## 设置

- 初始化：v2 best，冻结 ConvNeXt，仅训练 neck / heads 1 epoch
- 唯一变量：BiFPN 内 GroupNorm 改为 BatchNorm

## 结果（sunlamp 400 子集）

| 指标 | heat | effi | seg | final |
| --- | ---: | ---: | ---: | ---: |
| pose | 0.54033 | 0.47978 | — | **0.49444** |
| eR [deg] | 24.652 | 23.812 | — | — |
| IoU | — | 0.87684 | 0.84395 | — |

合成域训练期 10% 验证：`heat_eR 1.291°`、`effi_eT 0.099 m`、`final_pose 0.040`。

## 判读

- 相对 `probe_ctl`（0.36731），GN→BN 后 `final_pose` 变差 34.6%，合成域也从 0.025 掉到 0.040。
- 更像是整体拟合能力下降，而不是单纯的跨域鲁棒性差异；不能据此把 v2 退化归因于 GN neck。
