# SPNv2_BiFPNConvNeXt_v3_261123

- 模型：ImageNet 预训练 ConvNeXt-T + BiFPN，`heatmap` / `segmentation` 双头训练
- 实验类型：在 v2 结构基础上裁剪头部，只保留 Heat + Seg，验证双头是否优于三头
- 日期：2026-11-23
- 配置：`configs/SPNv2_BiFPNConvNeXt_v3_261123.yaml`（快照 `config.yaml`）
- 权重：`checkpoints/SPNv2_BiFPNConvNeXt_v3_261123/`
- 日志：`logs/SPNv2_BiFPNConvNeXt_v3_261123/`
- 结果：`results/tables/SPNv2_BiFPNConvNeXt_v3_261123/{synthetic,lightbox,sunlamp}/`

## 结构

- 骨干 / 颈部 / 数据 / 增广 / 优化器与 v2 完全一致（ConvNeXt-T、输出 LN、GN BiFPN、全帧 768x512）
- 头部只构建 `heatmap` + `segmentation`；两者都只取 P2，损失为 `hmap` + `seg`
- `MODEL.FIND_UNUSED_PARAM: True`：两个头只消费 P2，BiFPN 中仅服务 P3~P7 的
  `conv*_down` 与权重参数不参与损失，DDP 必须允许未使用参数，否则第二轮迭代会报
  `Expected to have finished reduction in the prior iteration` 并卡住
- 并行：双卡 DDP，`TRAIN.IMAGES_PER_GPU: 8`，全局 batch = 16；20 epoch，AdamW + step LR
- 测试：`TEST.HEAD: ['heatmap', 'segmentation']`，pose 指标由 heatmap 头给出

## 与 v2 的差异

| 项目 | v2 | v3 |
| --- | --- | --- |
| 头部 | heatmap + efficientpose + segmentation | heatmap + segmentation |
| 训练损失 | hmap / cls / box / pose / seg | hmap / seg |
| DDP | FIND_UNUSED_PARAM False | FIND_UNUSED_PARAM True |
| 其他（骨干、颈部、增广、优化器、轮数） | - | 与 v2 相同 |

## 训练（双卡，前台）

```bash
cd /opt/data/private/speNet
unset LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl \
/opt/data/private/venvs/zz_fyp/bin/python src/train.py \
  --cfg configs/SPNv2_BiFPNConvNeXt_v3_261123.yaml \
  AUTO_RESUME False VERBOSE True
```

后台运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl \
/opt/data/private/venvs/zz_fyp/bin/python src/train.py \
  --cfg configs/SPNv2_BiFPNConvNeXt_v3_261123.yaml \
  AUTO_RESUME False VERBOSE True \
  > logs/SPNv2_BiFPNConvNeXt_v3_261123.console.log 2>&1 &
tail -f logs/SPNv2_BiFPNConvNeXt_v3_261123.console.log
```

续训：去掉 `AUTO_RESUME False`，保持双卡与相同 `EXP_NAME`。

## 测试

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
/opt/data/private/venvs/zz_fyp/bin/python src/tools/test.py \
  --cfg configs/SPNv2_BiFPNConvNeXt_v3_261123.yaml \
  EXP_NAME SPNv2_BiFPNConvNeXt_v3_261123 VERBOSE True \
  TEST.TEST_CSV lightbox/labels/lightbox.csv
```

sunlamp / 合成域替换 `TEST.TEST_CSV` 为 `sunlamp/labels/sunlamp.csv`、`synthetic/labels/validation.csv`。

## 日志格式

- 训练：`Epoch 01/20 train [ 245/2997]  8.2% | 01:42<19:16 | 420.1 ms/batch | hmap ... | seg ...`
- 终端内单行原地刷新；重定向到文件时每约 60 条输出一行，不再刷屏
- 每轮结束额外输出一行汇总：`Epoch 01/20 train summary | batches ... | hmap ... | seg ...`
- 验证：`Validation 01/20 ...`，汇总行同样以 `summary |` 开头

## 判据

- 首轮 `hmap`、`seg` 均正常下降，验证集 `heat_eR` 与 `final_pose` 回到 v2 同量级
- 与 v2 对比：`final_pose`（lightbox / sunlamp）不劣于三头版本，`segm_iou` 同步收敛
- 双卡吞吐参考：约 380~450 ms/batch
