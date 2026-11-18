# SPNv2 训练 / 测试 SOP

## 0. 运行环境（venv 复用镜像包）

| 项目 | 位置 / 版本 |
| --- | --- |
| 解释器 | `/opt/data/private/venvs/zz_fyp/bin/python` |
| 基础镜像自带 | Python 3.12、torch 2.10、torchvision 0.25、numpy 2.1、scipy 1.16、matplotlib、pandas、tqdm、Pillow、PyYAML |
| venv 内额外安装 | yacs 0.1.8、opencv-python-headless、albumentations 1.3.1、plyfile、torchinfo、styleaug（源码 `-e` 安装） |

venv 用 `--system-site-packages` 创建，torch/torchvision 等直接借用镜像里的安装，
不重复占盘，镜像升级 torch 后 venv 自动跟随。

需要重建时（例如镜像换代）：

```bash
/usr/bin/python3.12 -m venv --system-site-packages /opt/data/private/venvs/zz_fyp
/opt/data/private/venvs/zz_fyp/bin/pip install \
  "yacs==0.1.8" "opencv-python-headless>=4.10" "albumentations==1.3.1" \
  "plyfile>=1.0" "torchinfo>=1.8"
/opt/data/private/venvs/zz_fyp/bin/pip install -e \
  /opt/data/private/speNet/src/style-augmentation-master/style-augmentation-master
```

自检：

```bash
/opt/data/private/venvs/zz_fyp/bin/python -c \
"import torch, torchvision, yacs, cv2, albumentations, plyfile, torchinfo, styleaug; \
print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

> 若终端报 `ImportError: ... GLIBCXX_3.4.29 not found`，说明 shell 里残留了
> Trae 自带的 `LD_LIBRARY_PATH`，先执行 `unset LD_LIBRARY_PATH` 再跑。

## 1. 目录结构

```
configs/        实验配置（configs/*.yaml）
experiments/    实验记录（每个实验一个目录：config.yaml / notes.md / metrics.csv）
checkpoints/    模型权重（checkpoints/<EXP_NAME>/，官方预训练在 checkpoints/pretrained/）
logs/           训练/测试日志（logs/<EXP_NAME>/<phase>_<时间>/）
results/        评估结果（results/tables/、results/visualizations/）
docs/           设计与操作文档
src/            源码（train.py、tools/ 及各 Python 包）
```

## 2. 每次使用前：改这 4 行

```bash
export PY=/opt/data/private/venvs/zz_fyp/bin/python
export CFG=configs/baseline.yaml
export EXP=SPNv2_baseline_v1_$(date +%y%m%d)   # {模型}_{实验类型}_{改动点}_{数据集}_{版本}_{日期}
export GPUS=0,1
```

进入工程目录，并确认有哪些实验配置：

```bash
cd /opt/data/private/speNet
ls configs/*.yaml
```

`CFG` 直接填 `configs/` 下的文件名即可，新增 yaml 后无需改本文件。

## 3. 训练（默认双卡，前台运行，带进度条）

```bash
CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl \
$PY src/train.py --cfg $CFG EXP_NAME $EXP AUTO_RESUME False VERBOSE True
```

进度条说明：每完成一个 batch 原地刷新一次，显示 `Training 001 [i/N] [Time] hmap cls box pose seg`；每轮结束再打印一行汇总。由 `VERBOSE True` 开启。

单卡跑同一份配置（去掉分布式包装）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl \
$PY src/train.py --cfg $CFG EXP_NAME $EXP AUTO_RESUME False VERBOSE True \
  DIST.MULTIPROCESSING_DISTRIBUTED False
```

## 4. 后台训练（进度实时可见）

```bash
CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl \
  $PY src/train.py --cfg $CFG EXP_NAME $EXP AUTO_RESUME False VERBOSE True \
  > logs/$EXP.console.log 2>&1 &
```

看进度：

```bash
tail -f logs/$EXP.console.log
```

## 5. 续训

```bash
CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl \
$PY src/train.py --cfg $CFG EXP_NAME $EXP VERBOSE True
```

`EXP_NAME` 保持不变、不要加 `AUTO_RESUME False`；卡数必须与首次训练一致。

## 6. 测试（带进度条）

lightbox：

```bash
CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 \
$PY src/tools/test.py --cfg $CFG EXP_NAME $EXP VERBOSE True \
  TEST.TEST_CSV lightbox/labels/lightbox.csv
```

sunlamp：

```bash
CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 \
$PY src/tools/test.py --cfg $CFG EXP_NAME $EXP VERBOSE True \
  TEST.TEST_CSV sunlamp/labels/sunlamp.csv
```

进度显示 `Testing 001 [i/N] [Time] heat_eR effi_eT final_pose`，跑完打印 `Mean performance`。

## 7. 可视化

```bash
$PY src/tools/vis_pose.py --cfg $CFG
$PY src/tools/vis_failures.py --run results/tables/$EXP --baseline results/tables/official_offline
```

## 8. 常用覆盖项

以 `KEY VALUE` 形式追加到命令末尾，可叠加多条：

```bash
TRAIN.IMAGES_PER_GPU 4              # 显存不足时降 batch
TRAIN.WORKERS 2                     # /dev/shm 较小时降 worker（可为 0）
TRAIN.END_EPOCH 40                  # 训练轮数
TRAIN.VALID_FREQ 5                  # 每 5 轮验证一次
AUTO_RESUME False                   # 新实验，不接旧 checkpoint
EXP_NAME my_exp_01                  # 指定实验名
--dist-url tcp://127.0.0.1:29577    # 23456 端口被占用时
```

## 9. 输出位置

```
logs/$EXP/train_*/train_rank0.log
logs/$EXP/test_*/test_rank0.log
checkpoints/$EXP/checkpoint.pth.tar
checkpoints/$EXP/model_best.pth.tar
checkpoints/$EXP/model_final.pth.tar
results/tables/$EXP/<domain>/mean_performance.txt
results/visualizations/$EXP/
```

## 10. 命名规范

实验名（`EXP_NAME`）统一格式，全英文，日期用 `YYMMDD`：

```
{模型}_{实验类型}_{改动点}_{数据集}_{版本}_{日期}
```

基准实验没有改动点/数据集时可省略对应字段，例如 `SPNv2_baseline_v1_261113`。

目录与文件命名：

| 类型 | 规则 | 示例 |
| --- | --- | --- |
| 实验配置 | `configs/<实验名>.yaml` | `configs/transformer_v01.yaml` |
| 实验记录 | `experiments/<EXP_NAME>/{config.yaml,notes.md,metrics.csv}` | `experiments/SPNv2_baseline_v1_261113/` |
| 模型权重 | `checkpoints/<EXP_NAME>/` | `checkpoints/SPNv2_baseline_v1_261113/model_best.pth.tar` |
| 训练日志 | `logs/<EXP_NAME>/<phase>_<时间>/` | `logs/SPNv2_baseline_v1_261113/train_20261113_104127/` |
| 实验结果 | `results/tables/<EXP_NAME>/<domain>/` | `results/tables/SPNv2_baseline_v1_261113/lightbox/` |
| 可视化结果 | `results/visualizations/<EXP_NAME>/` | `results/visualizations/SPNv2_baseline_v1_261113/sunlamp_failures.png` |
| 架构设计 | `design_模块名_版本.md` | `design_pose_decoder_v02.md` |
| 实验笔记 | `experiments/<EXP_NAME>/notes.md` | `experiments/SPNv2_baseline_v1_261113/notes.md` |

导出/对外交付的单个文件仍按 `train_<实验名>.log`、`<模型>_<实验名>_epoch.pth`、`result_<实验名>.csv`、`vis_<实验名>_<样本ID>.png` 命名。
