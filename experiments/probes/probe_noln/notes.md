# probe_noln（去掉 4 个输出 LayerNorm）

- 父实验：`SPNv2_BiFPNConvNeXt_v2_261122`
- 目的：判断 v2 的输出 LayerNorm 是否是真实域退化的主因
- 配置：`configs/SPNv2_BiFPNConvNeXt_v2_261122_probe_noln.yaml`（快照 `config.yaml`）
- 权重：`checkpoints/SPNv2_BiFPNConvNeXt_v2_261122_probe_noln/`
- 训练日志：`logs/SPNv2_BiFPNConvNeXt_v2_261122_probe_noln/`
- 结果：`results/tables/SPNv2_BiFPNConvNeXt_v2_261122_probe_noln_sub400/sunlamp/`
- 记录：`experiments/probes/probe_noln/`（不再挂在父实验目录下）
- 评测集：`sunlamp/labels/sunlamp_probe400.csv`（400 张固定子集）

## 设置

- 初始化：v2 best，冻结 ConvNeXt，仅训练 neck / heads 1 epoch
- 唯一变量：移除 v2 的 4 个输出 LayerNorm（`backbone.input_norms`）

## 结果（sunlamp 400 子集）

| 指标 | heat | effi | seg | final |
| --- | ---: | ---: | ---: | ---: |
| pose | 0.41104 | 0.41357 | — | **0.39907** |
| eR [deg] | 19.396 | 20.226 | — | — |
| IoU | — | 0.88626 | 0.85797 | — |

合成域训练期 10% 验证：`heat_eR 0.949°`、`effi_eT 0.051 m`、`final_pose 0.026`。

## 判读

- 相对 `probe_ctl`（0.36731），去掉输出 LN 后 `final_pose` 变差 8.6%。
- 说明输出 LN 不是 v2 退化的原因，反而略有帮助；v2 相对 baseline 的退化应继续向上游骨干或训练配方查找。
