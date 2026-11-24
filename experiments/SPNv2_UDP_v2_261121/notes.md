# SPNv2_UDP_v2_261121（作废）

- 目的：UDP + SyncBN 的首次干净重训尝试
- 配置：`configs/SPNv2_UDP_v2_261121.yaml`（快照 `config.yaml`）
- 状态：**作废**，训练发散（hmap 95 → NaN，2026-11-22 03:06 NCCL collective timeout）
- 权重：`checkpoints/SPNv2_UDP_v2_261121/`（仅早期 best，不可用）
- 日志：`logs/SPNv2_UDP_v2_261121/`
- 结果：`results/tables/SPNv2_UDP_v2_261121/sunlamp/`（只测了 sunlamp）

## 结果（发散 run 的中间 best，仅作记录）

sunlamp 全量：heat `eR 26.100°` / `pose 0.56029`，effi `pose 0.59301`，seg IoU `0.79113`，`final_pose 0.55622`。

## 说明

- 该行不能作为 UDP 结论，仅用于说明发散 run 的失败量级。
- 干净 UDP 对照改由 `SPNv2_UDP_v2_261123` 完成。
