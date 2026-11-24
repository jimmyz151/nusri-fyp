# v1_recheck_probe400（v1 stage-1 在同一 400 子集上的复核）

- 参考实验：`SPNv2_BiFPNConvNeXt_v1_261119`（stage-1 权重）
- 目的：用与 v2 探针完全相同的 `sunlamp_probe400` 子集复核 v1，判断“v2 相对 v1 退化”是否成立
- 配置：`configs/SPNv2_BiFPNConvNeXt_v1_261119_s1.yaml`（快照 `config.yaml`，仅覆盖 `TEST.MODEL_FILE` 与 `TEST.TEST_CSV`）
- 权重：`checkpoints/SPNv2_BiFPNConvNeXt_v1_261119/stage1_model_best.pth.tar`
- 日志：`logs/v1_recheck_probe400/`
- 结果：`results/tables/v1_recheck_probe400/sunlamp/`
- 记录：`experiments/probes/v1_recheck_probe400/`（不再挂在父实验目录下）
- 评测集：`sunlamp/labels/sunlamp_probe400.csv`（400 张固定子集）

## 结果（sunlamp 400 子集）

| 指标 | heat | seg | final |
| --- | ---: | ---: | ---: |
| pose | 0.22675 | — | **0.22675** |
| eR [deg] | 10.870 | — | — |

该实验只有 heatmap + segmentation 两头，没有 efficientpose，因此没有 effi 指标。

## 判读

- 同一 400 子集上，v1 stage-1 为 **0.22675**，v2 原模型为 **0.36260**；v1 好 37.5%。
- 这说明 v2 的退化不是“子集口径”造成的，v1 的骨干/颈部组合在该子集上确实明显更强。
- 由此，探针结论应表述为：v2 的退化主要来自上游 ConvNeXt 特征/训练配方的整体差异，而不是输出 LN 或 GN neck 单一模块。
