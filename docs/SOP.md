# SPNv2 训练 / 测试 SOP

## 0. 环境

```bash
export PY=/opt/data/private/venvs/zz_fyp/bin/python
unset LD_LIBRARY_PATH          # 必须；否则 torch 报 GLIBCXX_3.4.29 错误
cd /opt/data/private/speNet
$PY -c "import torch, yacs, cv2, albumentations; print(torch.__version__, torch.cuda.is_available())"
```

venv 用 `--system-site-packages` 创建；重建见 `requirements.txt` 与 `docs/README.md`。TensorBoard 已随镜像安装（2.20+）。

## 1. 目录约定

```
configs/                     实验配置
experiments/<模型实验>/       config.yaml / metrics.csv / notes.md
experiments/probes/<探针名>/  探针、子集评测、复查实验（不要放进模型实验目录）
logs/<EXP>/                   train_* / test_* 日志
results/tables/<EXP>/         测试结果
results/visualizations/<EXP>/ 可视化
results/tensorboard/<EXP>/    实时 TensorBoard
checkpoints/<EXP>/            checkpoint / model_best / model_final
```

**探针实验只放 `experiments/probes/<探针名>/`**，不要放 `experiments/<模型实验>/probes/`，也不要直接放 `experiments/<探针名>/`。

## 2. 训练

```bash
export CFG=configs/SPNv2_BiFPNConvNeXt_v3_261123.yaml
export EXP=SPNv2_BiFPNConvNeXt_v3_261123
export GPUS=0,1

CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl \
$PY src/train.py --cfg $CFG EXP_NAME $EXP AUTO_RESUME False VERBOSE True
```

后台运行（把输出重定向）：

```bash
CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl \
$PY src/train.py --cfg $CFG EXP_NAME $EXP AUTO_RESUME False VERBOSE True \
  > logs/$EXP.console.log 2>&1 &
tail -f logs/$EXP.console.log
```

单卡：追加 `DIST.MULTIPROCESSING_DISTRIBUTED False`。
续训：保持 `EXP_NAME` 不变，去掉 `AUTO_RESUME False`，卡数必须与首次一致。

## 3. 训练内监控（TensorBoard + probe）

TB 默认开启（`TRAIN.TENSORBOARD: True`），只由 rank 0 写 `results/tensorboard/<EXP>/`。写入失败只告警，不影响训练。

```bash
env -u LD_LIBRARY_PATH tensorboard --logdir results/tensorboard \
  --port 6006 --bind_all --reload_interval 5
```

历史日志回填（跳过已有 live TB 的 run）：

```bash
$PY src/tools/export_tensorboard.py
$PY src/tools/export_tensorboard.py --exp <EXP>          # 只导一个
$PY src/tools/export_tensorboard.py --exp <EXP> --force  # 覆盖 live 目录
```

### 内嵌 probe（目标域小子集，仅监控）

默认开启（`TEST.PROBE_ENABLE: True`）；不需要时设 `False`。可覆盖：

```bash
TEST.PROBE_ENABLE True \
TEST.PROBE_CSV sunlamp/labels/sunlamp_probe400.csv \
TEST.PROBE_FREQ 4 \
TEST.PROBE_MIN_FREE_MB 1024
```

- 每 N 轮在 rank 0 上跑一次，`batch=1`、`no_grad`、`num_workers=0`；
- 只写指标：`experiments/probes/<EXP>/online_probe.csv` + TB `probe/*`；
- **不保存权重、不保存预测/逐样本 .mat**；
- 显存不足会跳过并告警；任何异常只告警，**绝不中止训练**；
- 是否停止训练由人工判断，代码不会自动 kill。

probe 指标：`heat_eR/eT`、`eR_HIL/eT_norm_HIL`、`final_pose`、`segm_iou`、`reject`、`fail30`、`median_eR`。

## 4. 测试

```bash
# lightbox
CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 \
$PY src/tools/test.py --cfg $CFG EXP_NAME $EXP VERBOSE True \
  TEST.TEST_CSV lightbox/labels/lightbox.csv

# sunlamp
CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 \
$PY src/tools/test.py --cfg $CFG EXP_NAME $EXP VERBOSE True \
  TEST.TEST_CSV sunlamp/labels/sunlamp.csv

# synthetic validation
CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 \
$PY src/tools/test.py --cfg $CFG EXP_NAME $EXP VERBOSE True \
  TEST.TEST_CSV synthetic/labels/validation.csv
```

`experiments/<EXP>/metrics.csv` 至少包含 `synthetic`、`lightbox`、`sunlamp` 三个 domain。

## 5. 可视化

```bash
# 预测姿态
$PY src/tools/vis_pose.py --cfg $CFG

# 长尾失败样本（每域 4 个误差区间，输出 results/visualizations/<EXP>/longtail_failures.{png,pdf}）
$PY src/tools/vis_longtail.py --run <EXP>
$PY src/tools/vis_longtail.py --run <EXP> --domains sunlamp --head heatmap
```

## 6. 常用覆盖项

```bash
TRAIN.IMAGES_PER_GPU 4        # 显存不足时降 batch
TRAIN.WORKERS 2               # /dev/shm 小时降 worker
TRAIN.END_EPOCH 40
TRAIN.VALID_FREQ 5
TRAIN.TENSORBOARD False       # 关闭 TB
AUTO_RESUME False
--dist-url tcp://127.0.0.1:29577
```

## 7. 输出位置

```
logs/$EXP/train_*/train_rank0.log
checkpoints/$EXP/{checkpoint,model_best,model_final}.pth.tar
results/tables/$EXP/<domain>/mean_performance.txt
results/tensorboard/$EXP/                     # 实时 TB
experiments/probes/$EXP/online_probe.csv      # 内嵌 probe
results/visualizations/$EXP/
```

不保留逐 epoch 的 ladder 权重；需要看真实域曲线时用内嵌 probe + TB。

## 8. 命名规范

`{模型}_{实验类型}_{改动点}_{数据集}_{版本}_{日期}`，日期 `YYMMDD`，例如 `SPNv2_baseline_v1_261113`。
探针/子集实验同样命名，但记录统一放 `experiments/probes/<EXP>/`。

## 9. TensorBoard 看什么

| tag | 含义 |
|---|---|
| `train/*` | 训练损失（hmap/seg/cls/box/pose） |
| `val/heat_eR`、`val/final_pose`、`val/effi_eT` | 合成域验证 |
| `probe/*` | 目标域小子集（`final_pose`、`eR_HIL`、`eT_norm_HIL`、`fail30`、`median_eR`） |
| `opt/lr` | 学习率 |
| `perf/ms_per_batch`、`perf/epoch_seconds` | 吞吐/耗时 |
| Text → `config/key_params`、`config/yaml`、`final/full_test` | 关键超参、完整配置、最终测试值 |

合成域只能看收敛；**跨域判断看 `probe/*`**。单点会抖，至少看两个连续 epoch。
