# SPNv2 实验结果汇总（截至 2026-11-24）

本文档汇总 `results/tables/`、`experiments/*/metrics.csv`、训练/测试日志与 `configs/` 中的全部已跑结果。
原始数值优先取自 `results/tables/<EXP>/<domain>/mean_performance.txt`，合成域取自训练日志。
探针实验的记录统一放 `experiments/probes/<探针名>/`；日志、结果、权重按各自 `EXP_NAME` 存放，不再挂在模型实验目录下。

## 0. 速览

- **真实域最好结果**来自 `SPNv2_GTBBox_v1_261118`（GT 框裁剪到 512×512）：lightbox final_pose **0.0695**、sunlamp **0.1584**；但这是"给定完美检测"的上界，不可直接部署。
- 官方离线权重复现为 lightbox **0.1415** / sunlamp **0.2267**；自训 baseline 为 **0.1528** / **0.2698**，final_pose 高 8.0% / 19.0%。
- ConvNeXt-T 替换：`v1 s1`（双头 + 原版 BN BiFPN + 全帧）与 baseline 基本持平（lightbox +13.6%，sunlamp −5.7%）；`v2`（三头 + 输出 LN + GN neck）和 `v3`（双头）在真实域明显变差，sunlamp 分别 +42.9% / +29.0%。
- `v1` 的两阶段 mask 裁剪推理**显著劣化**（lightbox 97 次 fallback，final 0.393）；`v1` 的 stage-2 裁剪混合训练**从未实际运行**，相关配置/代码已删除，现有 v1 结果全部来自 stage-1 权重。
- UDP 的第一次干净对照（`SPNv2_UDP_v2_261123`，SyncBN 口径一致）已完成 lightbox / sunlamp 全量：final_pose **0.1744 / 0.2911**，比 baseline 的 0.1528 / 0.2698 差 14.1% / 7.9%，**没有收益**；UDP 方向结项为负结果，不再作为候选方向。
- 微调探针（v2 best 出发，冻结 ConvNeXt 1 epoch）未定位到 v2 退化根因：去掉输出 LN 变差 8.6%，GN→BN 变差 34.6%；同一 400 子集上 v1 stage-1 为 **0.2268**，比 v2 原模型 0.3626 好 37.5%，说明差异在上游 ConvNeXt 特征/训练配方。
- ConvNeXt v3 训练早期不稳定：sunlamp 400 子集 final_pose 在 e5/e8/e12/e16/e20 为 0.711 / 1.163 / 0.620 / 0.326 / 0.315。
- GTBBox 裁剪框敏感性已完成 lightbox / sunlamp 各 400 张全量扫描，结果在 `results/tables/SPNv2_GTBBox_v1_261118/probes/box_sensitivity/`。

## 1. 评测口径

| 项目 | 说明 |
| --- | --- |
| 真实域 | SPEED+ HIL `lightbox`（6,740 张）、`sunlamp`（2,791 张） |
| 合成域 | `synthetic/labels/validation.csv`（11,994 张）。除 `official_offline` 与 400 子集外，训练实验只跑了训练期 10% 验证（前约 1,199 张），**不可与官方全量验证直接比较** |
| pose | `eR_HIL + eT_norm_HIL`（SPEED+ 阈值归一化） |
| final_pose | 三头模型：`heat_eR_HIL + effi_eT_norm_HIL`；双头/无 efficientpose：`heat_eR_HIL + heat_eT_norm_HIL`；heatmap 被拒的样本不参与统计 |
| reject | heatmap 头被 SPEED+ 阈值判为失败的样本数 |
| 热图头 | 表中 eR/eT 均取 heatmap 头 |

## 2. 真实域全量结果

| 实验 | 关键改动 | lightbox heat eR [deg] | lightbox heat eT [m] | lightbox seg IoU | lightbox final_pose (reject) | sunlamp heat eR [deg] | sunlamp heat eT [m] | sunlamp seg IoU | sunlamp final_pose (reject) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `official_offline` | 官方 phi=3 BN 权重，三头 | 6.442 | 0.1738 | 0.9044 | **0.1415** (3) | 10.871 | 0.2386 | 0.8929 | **0.2267** (4) |
| `SPNv2_baseline_v1_261113` | EfficientDet-D3，三头，全帧 | 6.921 | 0.2016 | 0.8981 | 0.1528 (2) | 12.965 | 0.2502 | 0.8788 | 0.2698 (1) |
| `SPNv2_GTBBox_v1_261118` | GT 框裁剪 512×512，双头 | **3.154** | **0.0866** | **0.9299** | **0.0695** (1) | **7.280** | **0.1826** | **0.9033** | **0.1584** (0) |
| `SPNv2_UDP_v1_261119` | baseline + UDP，`SYNCBN=False` | 12.611 | 0.3324 | 0.8638 | 0.2791 (12) | 15.175 | 0.3291 | 0.8786 | 0.3086 (0) |
| `SPNv2_UDP_v2_261121` | UDP + SyncBN，训练发散 | — | — | — | — | 26.100 | 0.6964 | 0.7911 | 0.5562 (7) |
| `SPNv2_UDP_v2_261123` | UDP 干净对照，SyncBN 口径一致 | 7.916 | 0.2056 | 0.8861 | 0.1744 (4) | 14.276 | 0.3226 | 0.8695 | 0.2911 (1) |
| `BiFPNConvNeXt_v1 s1` | ConvNeXt-T + BN BiFPN，双头，全帧 | 7.969 | 0.2220 | 0.8804 | 0.1736 (4) | 12.069 | 0.2782 | 0.8552 | 0.2543 (4) |
| `BiFPNConvNeXt_v1 s1` 两阶段 | 同上 + mask 框裁剪二次推理 | 7.969 | 0.2220 | 0.8804 | 0.3928 (4, fallback 97) | 12.069 | 0.2782 | 0.8552 | 0.3338 (4, fallback 13) |
| `BiFPNConvNeXt_v2_261122` | ConvNeXt-T + 输出 LN + GN BiFPN，三头 | 8.570 | 0.2143 | 0.8955 | 0.1827 (7) | 18.606 | 0.4202 | 0.8557 | 0.3855 (9) |
| `BiFPNConvNeXt_v3_261123` | v2 去 efficientpose，双头，`model_final` | 9.781 | 0.2502 | 0.8843 | 0.2118 (7) | 16.483 | 0.4149 | 0.8603 | 0.3479 (1) |

补充：

- `v1` 两阶段行中的 eR/eT/seg 是 stage-1 全帧结果；final_pose 为 refine 结果（refine pose：lightbox 0.3905，sunlamp 0.3330）。
- `v1` 另有 `det_iou`：lightbox 0.8753，sunlamp 0.8789。
- `v2` 的 efficientpose：lightbox IoU 0.9076 / pose 0.2088；sunlamp IoU 0.8783 / pose 0.4060。
- `UDP v2_261121` 为发散 run 的中间 best 权重，不可作为 UDP 结论；且只测了 sunlamp。`UDP v2_261123` 为干净对照，lightbox / sunlamp 均已测。

### 相对 baseline 的 final_pose 变化

| 实验 | lightbox | sunlamp |
| --- | ---: | ---: |
| `official_offline` | −7.4% | −16.0% |
| `GTBBox`（oracle 框） | −54.5% | −41.3% |
| `BiFPNConvNeXt v1 s1` | +13.6% | −5.7% |
| `BiFPNConvNeXt v1 s1` 两阶段 | +157.0% | +23.7% |
| `UDP v1` | +82.6% | +14.4% |
| `UDP v2_261123`（干净对照） | +14.1% | +7.9% |
| `BiFPNConvNeXt v2` | +19.5% | +42.9% |
| `BiFPNConvNeXt v3` | +38.6% | +29.0% |

## 3. 合成域验证

| 实验 | 口径 | heat eR [deg] | effi eT [m] | final_pose | seg IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| `official_offline` | 全量 11,994 张独立测试 | 0.987 | 0.0423 | 0.0267 | 0.9579 |
| `baseline` | 训练期 10% 验证（1,199） | 0.830 | 0.046 | 0.023 | — |
| `GTBBox v1` | 同上 | 0.291 | — | 0.007 | — |
| `UDP v1` | 同上 | 0.652 | 0.048 | 0.020 | — |
| `UDP v2_261123` | 同上 | 0.60 | 0.05 | 0.02 | — |
| `UDP v2_261121` | 训练发散，无有效值 | — | — | — | — |
| `BiFPNConvNeXt v1 s1` | 同上 | 0.698 | — | 0.016 | — |
| `BiFPNConvNeXt v2` | 同上 | 0.900 | 0.048 | 0.024 | — |
| `BiFPNConvNeXt v3` | 同上 | 0.86 | — | 0.02 | — |

注意：`v2`/`v3`/`v1` 的合成域数值来自日志 2~3 位小数，精度低于 `metrics.csv` 中的真实域结果；`v3` 未跑独立合成域测试。

## 4. 消融与诊断

### 4.1 GTBBox 上界（`SPNv2_GTBBox_v1_261118`）

- GT 框中心 + 1.2 倍边长裁剪，原图 1920×1200 上裁切后 resize 到 512×512；物体像素约 267 px → 427 px。
- 热图 128×128（stride 4），UDP 解码 + EPnP；GT 位姿只用于评估。
- 结果：lightbox 0.0695 / sunlamp 0.1584，相对 baseline 降低 55% / 41%；说明**分辨率与裁剪是当前最大的单项收益来源**。
- 该结果是 oracle 上界：不包含检测框误差，不能作为部署指标。

### 4.2 UDP（`SPNv2_UDP_v1/v2_261121/v2_261123`）

- v1：唯一变量是 UDP（亚像素热图编解码），但实际配置为 `DIST.SYNCBN=False`，baseline 为 SyncBN，BN 统计量口径不同。真实域全面变差（lightbox eR 6.92→12.61），合成域反而更好（0.830→0.652），符合 BN 口径混淆特征，**不能归因于 UDP**。
- v2_261121：`SYNCBN=True` 的首次干净重训，但训练不稳定：hmap 从 95.6 起跳，随后 NaN，2026-11-22 03:06 NCCL collective timeout；best 权重只到早期，sunlamp 26.1°/0.556，**作废**。
- v2_261123：重做的干净对照（与 baseline 逐项一致，唯一变量 `DATASET.UDP=True`；训练端亚像素高斯中心 + 测试端 3 点抛物线解码），20 epoch 正常跑完。lightbox 全量 final_pose **0.1744**（heat eR 7.916° / effi pose 0.2097 / seg IoU 0.8861），sunlamp 全量 **0.2911**（heat eR 14.276° / effi pose 0.3092 / seg IoU 0.8695），分别比 baseline 差 14.1% / 7.9%。
- 结论：在 BN 口径一致的前提下，UDP 在两个真实域都**没有收益**；**UDP 方向结项为负结果，不再作为候选方向**。跨域差距应优先从前景分辨率（4.1 GTBBox 上界）入手。

### 4.3 ConvNeXt-T + BiFPN 系列

| 版本 | 骨干/neck | 头 | 归一化 | 真实域表现 |
| --- | --- | --- | --- | --- |
| v1 s1 | ConvNeXt-T + 原版 BiFPN | heat + seg | BN neck | 与 baseline 持平 |
| v2 | ConvNeXt-T + BiFPN | heat + effi + seg | 输出 LN + GN neck + GN heads | lightbox +19.5%，sunlamp +42.9% |
| v3 | 同 v2 | heat + seg | 同 v2 | lightbox +38.6%，sunlamp +29.0% |

- v1 的 neck 保留 BiFPN 是必要的：无 neck 对照的同期验证 heat_eR 在 e1/e4/e8 为 103.1°/56.6°/28.7°，而 baseline 为 3.57°/1.69°/1.38°，热图分支基本没学到关键点。
- v2/v3 相对 v1 同时改了「输出 LN、neck 归一化、头组合、训练配方」，**不是单变量对比**，不能据此单独判断 ConvNeXt 骨干优劣。
- v3 去掉 efficientpose 后，lightbox final_pose 反而比 v2 差 16%，sunlamp 好 9.7%；由于 final_pose 口径从混合头变为纯 heat 头，不能直接与 v2 等价比较。

### 4.4 微调探针（v2 best 出发，冻结 ConvNeXt，训练 neck/heads 1 epoch）

统一在 `sunlamp_probe400.csv`（400 张子集）上评测，**只能与同一子集内的数字比较**：

| 配置 | 改动 | heat eR [deg] | heat pose | effi pose | seg IoU | final_pose | 合成域 final_pose |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 原模型（子集） | 对照 | 17.422 | 0.3636 | 0.3889 | 0.8565 | 0.3626 | 0.024 |
| `probe_ctl` | 仅微调 1 epoch | 17.582 | 0.3578 | 0.4041 | 0.8536 | 0.3673 | 0.025 |
| `probe_noln` | 去掉 4 个输出 LayerNorm | 19.396 | 0.4110 | 0.4136 | 0.8580 | 0.3991 | 0.026 |
| `probe_bnneck` | BiFPN GroupNorm → BatchNorm | 24.652 | 0.5403 | 0.4798 | 0.8440 | 0.4944 | 0.040 |
| `v1_recheck_probe400` | v1 stage-1，同一子集复核 | 10.870 | 0.2268 | — | 0.8547 | **0.2268** | — |

判读：

- 额外 1 epoch 微调几乎无影响（0.3626 → 0.3673，+1.3%）。
- 去掉输出 LN 使最终位姿变差 8.6%，说明输出 LN **不是** v2 退化的主因，反而略有帮助。
- GN→BN neck 使结果变差 34.6%，且合成域也从 0.025 掉到 0.040；这更像整体拟合能力下降，而不是单纯的跨域鲁棒性差异。
- v1 stage-1 在同一 400 子集上为 0.2268，比 v2 原模型 0.3626 好 37.5%；探针没有定位到 v2 相对 v1/baseline 退化的根因，差异在上游 ConvNeXt 特征与训练配方，仍需拆分验证。
- 探针记录统一归档：`experiments/probes/`；日志/结果/权重仍按各自 `EXP_NAME` 放 `logs/`、`results/tables/`、`checkpoints/`，不再挂在模型实验目录下。

### 4.5 v3 epoch ladder（sunlamp 400 子集）

| epoch | heat eR [deg] | heat pose | seg IoU | reject | final_pose |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 34.115 | 0.7111 | 0.7845 | 0 | 0.7111 |
| 8 | 53.326 | 1.1627 | 0.5270 | 7 | 1.1627 |
| 12 | 29.244 | 0.6200 | 0.8049 | 0 | 0.6200 |
| 16 | 15.997 | 0.3259 | 0.8681 | 0 | 0.3259 |
| 20 | 14.964 | 0.3146 | 0.8729 | 0 | 0.3146 |

- e5~e12 仍在明显不稳定区间，e16 后才进入正常量级；**v3 对 epoch 选择敏感**，需要依赖 best checkpoint 或更稳的 warmup/LR。
- 同一权重在全量 sunlamp 上为 eR 16.48 / final 0.3479，400 子集更容易（14.96 / 0.3146）。
- 产物统一收在主干实验目录下：结果 `results/tables/SPNv2_BiFPNConvNeXt_v3_261123/ladder/e{05,08,12,16,20}/sunlamp/`，日志 `logs/SPNv2_BiFPNConvNeXt_v3_261123/ladder/e{05,08,12,16,20}/`，逐 epoch 指标 `experiments/SPNv2_BiFPNConvNeXt_v3_261123/epoch_ladder.csv`。

### 4.6 GTBBox 裁剪框敏感性（lightbox / sunlamp 各 400 张）

用 `SPNv2_GTBBox_v1_261118` 权重，比较 GT 框与带误差的框。全量扫描已完成，结果在
`results/tables/SPNv2_GTBBox_v1_261118/probes/box_sensitivity/{lightbox,sunlamp}_summary.csv`。
下表是同一 400 张子集内的框误差敏感性，**不是**模型的整体指标。

| 域 | 框配置 | HIL pose 均值 | HIL pose 中位数 | 优于 GT 比例 | 11 点全部在框内 | 中心误差（GT 边长归一） | 边长比（/1.2×GT） |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lightbox | GT | 0.0809 | 0.0284 | — | 96.5% | 0 | 1.000 |
| lightbox | mask 框 | 0.1431 | 0.0288 | 45.3% | 88.0% | 0.036 | 0.979 |
| lightbox | p50 扰动 | 0.0982 | 0.0274 | 44.8% | 95.3% | 0.016 | 0.979 |
| lightbox | p90 扰动 | 0.0952 | 0.0285 | 46.8% | 95.3% | 0.085 | 1.006 |
| lightbox | p95 扰动 | 0.1013 | 0.0311 | 40.0% | 83.5% | 0.160 | 1.063 |
| sunlamp | GT | 0.1574 | 0.0438 | — | 95.3% | 0 | 1.000 |
| sunlamp | mask 框 | 0.2467 | 0.0456 | 47.5% | 85.3% | 0.080 | 0.985 |
| sunlamp | p50 扰动 | 0.1687 | 0.0424 | 50.0% | 94.8% | 0.020 | 0.982 |
| sunlamp | p90 扰动 | 0.1791 | 0.0449 | 50.3% | 95.0% | 0.075 | 1.018 |
| sunlamp | p95 扰动 | 0.1727 | 0.0447 | 48.3% | 95.5% | 0.106 | 1.051 |

- 各配置的中位数都与 GT 接近（差距 < 10%），均值恶化主要来自尾部：mask 框使 all-in 从 96.5%→88.0%（lightbox）、95.3%→85.3%（sunlamp），均值分别从 0.081→0.143、0.157→0.247。
- p50/p90 扰动与 GT 同量级；p95 在 lightbox 上开始出现明显尾部损失，sunlamp 上仍接近 GT。
- 结论：mask 框的均值恶化由关键点被裁出/尾部样本导致，中位数基本不受影响；这也解释了 v1 两阶段裁剪推理为什么反而更差。
- 早先 50 张 pilot（`/tmp/gtbbox_box_sensitivity_pilot*`）已被本次 400 张全量结果替代。

## 5. 作废 / 未完成 / 易混淆项

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| `SPNv2_UDP_v2_261121` | 作废 | 训练发散（NaN + NCCL timeout），best 权重不可用；只测了 sunlamp |
| UDP 方向（`SPNv2_UDP_v2_261123`） | 已结项（负结果） | 干净对照下 lightbox 0.1744 / sunlamp 0.2911，均劣于 baseline；后续不再开 UDP 实验 |
| `BiFPNConvNeXt_v1` stage-2 | 未运行 | 原 `configs/SPNv2_BiFPNConvNeXt_v1_261119.yaml` 指向 `stage1_model_best.pth.tar`，但训练日志 `train_20261119_21_26_14/config.txt` 为 `PRETRAIN_FILE: None`、`CROP.ENABLE: False`，即只跑了 s1；该 stage-2 配置与裁剪代码已删除，现有 v1 结果均为 stage-1 |
| v1 两阶段推理 | 已跑但不可用 | mask 框驱动裁剪，lightbox fallback 97 次，final 0.393，劣于单阶段 |
| probe 全量 sunlamp 测试 | 中断 | 2026-11-23 02:06~02:21 的 5 次 probe 全量测试均无 summary 输出；只有 400 子集结果 |
| v2/v3 独立合成域测试 | 未跑 | 只有训练期 10% 验证日志 |
| `official_offline` 合成域 | 口径不同 | 全量 11,994 张，不能与训练实验的 10% 验证横向比较 |

## 6. 结果文件索引

| 实验 | 结果目录 | 备注 |
| --- | --- | --- |
| official_offline | `results/tables/official_offline/{synthetic,lightbox,sunlamp}/` | 含全量合成域 |
| baseline v1 | `results/tables/SPNv2_baseline_v1_261113/{lightbox,sunlamp}/` | |
| GTBBox v1 | `results/tables/SPNv2_GTBBox_v1_261118/{lightbox,sunlamp}/` | |
| UDP v1 | `results/tables/SPNv2_UDP_v1_261119/{lightbox,sunlamp}/` | |
| UDP v2_261121 | `results/tables/SPNv2_UDP_v2_261121/sunlamp/` | 发散 run |
| UDP v2_261123 | `results/tables/SPNv2_UDP_v2_261123/{lightbox,sunlamp}/` | 干净对照 |
| ConvNeXt v1 s1（两阶段） | `results/tables/SPNv2_BiFPNConvNeXt_v1_261119_s1/{lightbox,sunlamp}/` | 含 refine/fallback |
| ConvNeXt v1 s1（全帧） | `results/tables/SPNv2_BiFPNConvNeXt_v1_261119_s1_fullframe/{lightbox,sunlamp}/` | |
| ConvNeXt v2 | `results/tables/SPNv2_BiFPNConvNeXt_v2_261122/{lightbox,sunlamp}/` | |
| ConvNeXt v2 探针 | 记录 `experiments/probes/{sub400,probe_ctl,probe_noln,probe_bnneck,v1_recheck_probe400}/`；结果按各自 EXP_NAME 在 `results/tables/` | 同一 400 张 sunlamp 子集 |
| ConvNeXt v3 | `results/tables/SPNv2_BiFPNConvNeXt_v3_261123/{lightbox,sunlamp}/` | model_final |
| ConvNeXt v3 ladder | `results/tables/SPNv2_BiFPNConvNeXt_v3_261123/ladder/e{05,08,12,16,20}/sunlamp/` | 400 子集；日志在 `logs/SPNv2_BiFPNConvNeXt_v3_261123/ladder/e{05,08,12,16,20}/` |
| 失败可视化 | `results/visualizations/<EXP>/longtail_failures.{png,pdf}` | baseline / GTBBox / UDP v1 / UDP v2 / official / v1 s1 / v1 s1_fullframe |
| 裁剪框敏感性 | `results/tables/SPNv2_GTBBox_v1_261118/probes/box_sensitivity/{lightbox,sunlamp}_{summary,per_sample}.csv` | 各 400 张全量 |
