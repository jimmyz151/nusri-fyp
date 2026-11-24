# SPNv2_BiFPNConvNeXt_v2_261122

- 模型：ImageNet 预训练 ConvNeXt-T + BiFPN，`heatmap` / `efficientpose` / `segmentation` 三头
- 实验类型：骨干替换 + 颈部归一化改造（BN → GN），验证 ConvNeXt-T 与 LN/GN 混合归一化
- 日期：2026-11-22
- 配置：`configs/SPNv2_BiFPNConvNeXt_v2_261122.yaml`（快照 `config.yaml`）
- 权重：`checkpoints/SPNv2_BiFPNConvNeXt_v2_261122/`
- 日志：`logs/SPNv2_BiFPNConvNeXt_v2_261122/`
- 结果：`results/tables/SPNv2_BiFPNConvNeXt_v2_261122/{synthetic,lightbox,sunlamp}/`

## 结构

- 骨干：`convnext_bifpn_tiny`
  - ConvNeXt-T `features` 的 index 1/3/5/7 → C2/C3/C4/C5（96/192/384/768，stride 4/8/16/32）
  - 各级输出先过各自的 LayerNorm（`backbone.input_norms`，eps 1e-6），只作用于送往 BiFPN 的分支
  - 骨干内部原有 LayerNorm 全部保留，不替换、不改动
  - BiFPN 拓扑与 baseline 一致：C2~C5 → P2~P7，phi=3，宽度 160，6 层
  - 1x1 通道投影（`Conv1x1Block`）与 BiFPN 内部 separable conv 统一使用 GroupNorm（16 通道/组）
- 头部：`heatmap` + `efficientpose` + `segmentation`，输入 P2 / P3~P7，超参与损失沿用 baseline
- 数据：全帧输入 768x512；不启用 UDP、不启用裁剪、测试不做二次推理，均为单次前向
- 并行：双卡 DDP，`TRAIN.IMAGES_PER_GPU: 8`，全局 batch = 16；训练 20 epoch，AdamW + step LR

## 与 baseline 的差异

| 项目 | baseline | v2 |
| --- | --- | --- |
| 骨干 | EfficientNet-D3 | ConvNeXt-T |
| 颈部输入 | backbone 直出 | 各级独立 LayerNorm |
| 颈部归一化 | BN | GN |
| 头部 / 损失 / 增广 / 优化器 | baseline | 不变 |

## 冒烟测试（2026-11-22）

512x768 单卡 eval：

- 模块统计：LayerNorm 26、GroupNorm 216、BatchNorm 0
- 输出 LN 4 个，BiFPN 6 层
- 输出形状：heatmap `(1,11,128,192)`、efficientpose 4 个张量（73,656 anchors）、segmentation `(1,1,128,192)`
- train 模式 loss 正常返回：`hmap / cls / box / pose / seg` 均有值

结论：结构、通道、归一化与三头输出均符合设计，可以开始训练。

## 训练（双卡，手动启动）

```bash
cd /opt/data/private/speNet
unset LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl \
/opt/data/private/venvs/zz_fyp/bin/python src/train.py \
  --cfg configs/SPNv2_BiFPNConvNeXt_v2_261122.yaml \
  AUTO_RESUME False VERBOSE True
```

后台运行版本：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl \
/opt/data/private/venvs/zz_fyp/bin/python src/train.py \
  --cfg configs/SPNv2_BiFPNConvNeXt_v2_261122.yaml \
  AUTO_RESUME False VERBOSE True \
  > logs/SPNv2_BiFPNConvNeXt_v2_261122.console.log 2>&1 &
tail -f logs/SPNv2_BiFPNConvNeXt_v2_261122.console.log
```

续训：去掉 `AUTO_RESUME False`，保持双卡。

## 测试

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
/opt/data/private/venvs/zz_fyp/bin/python src/tools/test.py \
  --cfg configs/SPNv2_BiFPNConvNeXt_v2_261122.yaml \
  EXP_NAME SPNv2_BiFPNConvNeXt_v2_261122 VERBOSE True \
  TEST.TEST_CSV lightbox/labels/lightbox.csv
```

sunlamp / 合成域同理替换 `TEST.TEST_CSV` 为 `sunlamp/labels/sunlamp.csv`、`synthetic/labels/validation.csv`。

## 判据

- 同期合成域：第 1 轮 `heat_eR` 应回到个位数（baseline 3.57°），第 8 轮接近 baseline 的 1.38°
- 真实域目标：超过 baseline 的 `final_pose`（lightbox 0.153 / sunlamp 0.270）
- 同时观察 `effi_iou`、`segm_iou` 与 `heat_eR` 是否同步正常收敛，判断 LN → GN 桥接是否稳定

## 探针实验（已归档到 `probes/`）

所有探针从本实验 best 出发，冻结 ConvNeXt，只训练 neck / heads 1 epoch，统一在 `sunlamp/labels/sunlamp_probe400.csv`（400 张）上评测。

| 子实验 | 改动 | final_pose（400 子集） | 记录 |
| --- | --- | ---: | --- |
| `probes/sub400` | 本实验原模型对照 | 0.36260 | `experiments/probes/sub400/` |
| `probes/probe_ctl` | 仅额外微调 1 epoch | 0.36731 | `experiments/probes/probe_ctl/` |
| `probes/probe_noln` | 去掉 4 个输出 LayerNorm | 0.39907 | `experiments/probes/probe_noln/` |
| `probes/probe_bnneck` | BiFPN GroupNorm → BatchNorm | 0.49444 | `experiments/probes/probe_bnneck/` |
| `probes/v1_recheck_probe400` | v1 stage-1 在同一子集复核 | 0.22675 | `experiments/probes/v1_recheck_probe400/` |

产物路径：探针记录统一放 `experiments/probes/<子实验>/`；训练/测试日志 `logs/<子实验 EXP_NAME>/`，结果 `results/tables/<子实验 EXP_NAME>/`，权重 `checkpoints/<子实验 EXP_NAME>/`。不再放在本实验目录下。

