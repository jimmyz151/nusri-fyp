# SPNv2_GTBBox_v1_261118

- 模型：SPNv2（EfficientDet-D3，phi=3，BN backbone / GN heads），仅 heatmap + segmentation 两头
- 实验类型：SPNv3 式 GTBBox 裁剪（Stage-1 假设"给定完美检测"），验证裁剪-缩放管线的几何与收益
- 日期：2026-11-18
- 配置快照：`config.yaml`（源：`configs/SPNv2_GTBBox_v1_261118.yaml`）
- 权重：`checkpoints/SPNv2_GTBBox_v1_261118/`
- 日志：`logs/SPNv2_GTBBox_v1_261118/`
- 结果：`results/tables/SPNv2_GTBBox_v1_261118/{lightbox,sunlamp}/`（测试后生成）
- 指标汇总：`metrics.csv`

## 管线

- 输入 512×512。裁剪窗口以 GT 框中心为心，边长 `max(h,w)×1.2`，越界 reflect，再从**原图 1920×1200** 上裁切后 resize（不用 `images_768x512_RGB`，否则等效分辨率反而低于全图）
- 训练抖动：scale ×U[0.85,1.15]，中心平移 U[±0.1]×side；验证/测试不抖动，固定 expand 1.2
- 真值：11 关键点用同一仿射映射到裁剪系，128×128 热图（stride 4，σ=2，高斯中心为真实亚像素位置）
- 分割：全分辨率 `masks/` 同步裁剪后 resize 到 128×128
- 解码：逐通道 argmax + UDP 抛物线半像素修正 → 逆仿射回原图 px → EPnP（内参、3D 关键点不变）
- GT 位姿只用于计算 eR/eT 与 SPEED

## 与全图 baseline 的差异

- 物体像素：267 px（占画面 35%）→ 427 px（占 83%）；热图 1 格对应原图 10.0 px → 5.4 px
- 离线风格图（`styles_768x512_RGB`）与裁剪框不对齐，已强制关闭；在线风格化保留（PROB 0.5 / ALPHA 0.5），但只作用于裁剪区域，背景纹理不再被风格化
- 去掉 efficientpose 后 BiFPN P3~P7 输出无人使用，DDP 需 `FIND_UNUSED_PARAM: True`

## 复现

```bash
cd /opt/data/private/speNet
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl \
/opt/data/private/venvs/zz_fyp/bin/python src/train.py \
  --cfg configs/SPNv2_GTBBox_v1_261118.yaml \
  EXP_NAME SPNv2_GTBBox_v1_261118 AUTO_RESUME False VERBOSE True
```

测试：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
/opt/data/private/venvs/zz_fyp/bin/python src/tools/test.py \
  --cfg configs/SPNv2_GTBBox_v1_261118.yaml EXP_NAME SPNv2_GTBBox_v1_261118 VERBOSE True \
  TEST.TEST_CSV lightbox/labels/lightbox.csv
```

## 预期与判据

- 上界口径（GT 框）：与 `SPNv2_baseline_v1_261113` 的 heat 头比较；注意 baseline 是三头训练，存在多任务差异
- 几何下限（GT 关键点直接解码）：0.06 px / eR 0.03° / eT 0.0004 m，若实测远离该值即为网络误差而非坐标变换误差
- 尚无检测器，本轮不给部署绝对值；框误差敏感性留待 P1 合成扰动扫描

## 备注

- 吞吐：双卡 bs8（全局 16）约 560 ms/iter，2997 iter/epoch，约 28 min/epoch
- 环境：本机后台执行拿不到 GPU，必须前台运行
