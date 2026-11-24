# sub400（v2 原模型，sunlamp 400 子集）

- 父实验：`SPNv2_BiFPNConvNeXt_v2_261122`
- 目的：为全部微调探针提供同一 400 张 sunlamp 子集上的原模型对照
- 配置：`configs/SPNv2_BiFPNConvNeXt_v2_261122.yaml`（快照 `config.yaml`，仅覆盖 `TEST.TEST_CSV`）
- 权重：`checkpoints/SPNv2_BiFPNConvNeXt_v2_261122/model_best.pth.tar`
- 日志：`logs/SPNv2_BiFPNConvNeXt_v2_261122_sub400/`
- 结果：`results/tables/SPNv2_BiFPNConvNeXt_v2_261122_sub400/sunlamp/`
- 记录：`experiments/probes/sub400/`（不再挂在父实验目录下）
- 评测集：`sunlamp/labels/sunlamp_probe400.csv`（400 张固定子集）

## 结果（sunlamp 400 子集）

| 指标 | heat | effi | seg | final |
| --- | ---: | ---: | ---: | ---: |
| pose | 0.36358 | 0.38890 | — | **0.36260** |
| eR [deg] | 17.422 | 18.930 | — | — |
| IoU | — | 0.88601 | 0.85646 | — |

合成域数值沿用 v2 训练期 10% 验证：`heat_eR 0.900°`、`effi_eT 0.048 m`、`final_pose 0.024`。

## 判读

- 该行是探针表的基准行；400 子集比全量 sunlamp 更容易（全量 `final_pose 0.38552`），只能与同子集数字比较。
