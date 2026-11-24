# SPNv2_UDP_v2_261123（UDP 干净对照）

- 目的：在 BN 统计口径一致的前提下，单独验证 UDP（亚像素热图编解码）是否有效
- 配置：`configs/SPNv2_UDP_v2_261123.yaml`（快照 `config.yaml`）
- 权重：`checkpoints/SPNv2_UDP_v2_261123/`
- 日志：`logs/SPNv2_UDP_v2_261123/`
- 结果：`results/tables/SPNv2_UDP_v2_261123/`

## 与 baseline 的差异

- 与 `SPNv2_baseline_v1_261113` 逐项一致，唯一变量是 `DATASET.UDP=True`
- 训练端使用亚像素高斯中心（不再 `int()` 截断），测试端使用 3 点抛物线半像素解码
- 双卡 DDP 下 baseline 与本实验都由 `train.py` 自动转 SyncBN，BN 统计口径一致

## 训练

- 2026-11-23 22:59 启动，20 epoch 正常跑完（2026-11-24 14:48），无 NaN / 无 NCCL timeout
- best 权重对应合成域验证 `final_pose 0.02`（`heat_eR 0.60°`、`effi_eT 0.05 m`）
- 训练日志：`logs/SPNv2_UDP_v2_261123/train_20261123_22_59_44/`

## 结果

- 两个域均加载 `model_best.pth.tar`；sunlamp 日志 `logs/SPNv2_UDP_v2_261123/test_20261124_14_49_16/`，lightbox 日志 `logs/SPNv2_UDP_v2_261123/test_20261124_14_53_06/`

### lightbox 全量（6,740 张）

| 指标 | heat | effi | final |
| --- | ---: | ---: | ---: |
| pose | 0.16958 | 0.20967 | **0.17441** |
| eR [deg] | 7.916 | 9.924 | — |
| IoU | — | 0.90651（seg 0.88610） | — |

### sunlamp 全量（2,791 张）

| 指标 | heat | effi | final |
| --- | ---: | ---: | ---: |
| pose | 0.30042 | 0.30923 | **0.29107** |
| eR [deg] | 14.276 | 15.309 | — |
| IoU | — | 0.90999（seg 0.86949） | — |

## 判读（负结果，UDP 方向结项）

- 与 baseline 相比：lightbox final_pose 0.17441 vs 0.15281（+14.1%），sunlamp 0.29107 vs 0.26977（+7.9%），两个真实域都变差；heat 头 eR 同步变差（6.92° → 7.92°，12.97° → 14.28°）。
- 这是 UDP 第一次干净对照：v1 的 `SYNCBN=False` 口径污染问题、v2_261121 的发散问题都不存在，训练全程稳定。
- 合成域 0.02 与 baseline 0.023 同量级，但该值来自训练期 10% 验证（1,199 张），不是独立合成域测试。
- 结论：在 BN 口径一致的前提下，UDP **没有收益**，单变量差异为负；**UDP 方向结项为负结果，不再作为候选方向**。跨域差距应优先从前景分辨率（GTBBox 上界）入手。
