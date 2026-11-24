# probe_ctl（v2 微调探针对照）

- 父实验：`SPNv2_BiFPNConvNeXt_v2_261122`
- 目的：扣除“从 v2 best 出发、冻结 ConvNeXt、只训练 neck/heads 1 epoch”本身的收益/损失
- 配置：`configs/SPNv2_BiFPNConvNeXt_v2_261122_probe_ctl.yaml`（快照 `config.yaml`）
- 权重：`checkpoints/SPNv2_BiFPNConvNeXt_v2_261122_probe_ctl/`
- 训练日志：`logs/SPNv2_BiFPNConvNeXt_v2_261122_probe_ctl/`
- 结果：`results/tables/SPNv2_BiFPNConvNeXt_v2_261122_probe_ctl_sub400/sunlamp/`
- 记录：`experiments/probes/probe_ctl/`（不再挂在父实验目录下）
- 评测集：`sunlamp/labels/sunlamp_probe400.csv`（400 张固定子集）

## 设置

- 初始化：`checkpoints/SPNv2_BiFPNConvNeXt_v2_261122/model_best.pth.tar`
- `TRAIN.FREEZE_BACKBONE: True`，仅 neck / heads 可训练，1 epoch
- 其余结构与训练配方同 v2

## 结果（sunlamp 400 子集）

| 指标 | heat | effi | seg | final |
| --- | ---: | ---: | ---: | ---: |
| pose | 0.35776 | 0.40411 | — | **0.36731** |
| eR [deg] | 17.582 | 19.650 | — | — |
| IoU | — | 0.88433 | 0.85361 | — |

合成域训练期 10% 验证：`heat_eR 0.924°`、`effi_eT 0.051 m`、`final_pose 0.025`。

## 判读

- 相对 v2 原模型（同子集 `final_pose 0.36260`），额外 1 epoch 微调后为 0.36731，仅变差 1.3%，属于对照噪声量级。
- 该对照用于解释 `probe_noln` / `probe_bnneck` 的差值，不能与全量 sunlamp（2,791 张）结果横向比较。
- 2026-11-23 02:06~02:21 的 5 次 probe 全量 sunlamp 测试均未产出 summary，已废弃；只保留 400 子集结果。
