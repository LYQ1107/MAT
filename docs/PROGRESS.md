# MAT 进度（2026-09-08）

## 状态摘要

`RESEARCH_PIPELINE_READY / PARTIALLY_EVALUATED`。M0 审计已执行；M1/M2 的代码和小型契约已实现；真实 P1 数据、姿态/身份权重及跨日姿态真值因直连路由未核验而未下载/未运行。没有训练、没有正式 benchmark、没有人工耗时或测试指标可报告。

## 已完成

- 重核查 `/data2/usr_for_deadline/MAT` 原不存在、GitHub `main` 为空；建立本地 `main` 和 `origin`，不覆盖其他目录。
- 记录机器/网络/磁盘/GPU 只读审计；应用层代理可在子进程清除，透明出口未证明，设置 `DIRECT_ROUTE_UNVERIFIED`。
- 写入 `AGENTS.md`、`docs/MASTER_PLAN.md`、`doctor.json`、`ASSET_PLAN.json`、网络/数据/上游审计和锁文件；敏感值未写入。
- 锁定已核查上游 commit；记录 DLC/ByteTrack/WildlifeTools 实际签名，未编造不在源码中的接口。
- 实现 DirectOnlyPolicy、限额 probe/fetch、重定向白名单、Range 续传保护、checksum/SHA-256、zip-slip 检查、原子 receipt 和授权本地 import。
- 实现中性数据契约、Rat/PigReID/PigTracking/MultiCamCows 适配器、冻结 split、ByteTrack 检测索引边界、冻结身份特征接口、部位/冲突匹配、anchor/quarantine/commit registry、固定 S0 mapping 评价和标注包边界。
- 研究分支 `research/longitudinal-mat` 当前本地/远端 SHA：`088367923698bfcd6090d9ee8d6651000856d65c`（`git ls-remote origin refs/heads/research/longitudinal-mat` 已核对）；main 保留安全基线 `78e7ae4`。

## 未运行/真实结果

| 项目 | status | 事实 |
|---|---|---|
| 大数据/权重下载 | `BLOCKED_DIRECT_ROUTE` | 0 bytes；无 `.part` 被发布 |
| P1 B0 pilot | `BLOCKED_MISSING_ASSET` | 无真实 Rat/PigReID 原件，未生成实验分母/准确率 |
| DLC/姿态 | `BLOCKED_MISSING_ASSET` | 没有本地 checkpoint/config |
| source/B2 训练 | `BLOCKED_MISSING_ASSET` | optimizer steps/checkpoint 均 `null`；未占 GPU |
| H-human | `BLOCKED_NEEDS_REFERENCE_REVIEW` | 没有人工操作，不编造分钟数 |
| O1/O2/外部基线 | `NOT_APPLICABLE`/blocked | 需要真实视频/session 和权重 |
| 联合纵向姿态 | `BLOCKED_NEEDS_POSE_GT` | 没有人工连续姿态真值 |

小型官方网页/源码文本核验已通过控制通道和受限 source-control 请求完成；Zenodo API 直连请求超时，未保存或猜测文件直链，故 provider 字节数/license 仍以待现场 API 为准。

## 下一条可执行命令

```bash
export MAT_WORK_ROOT=/data2/usr_for_deadline/MAT_workspace
PYTHONPATH=/data2/usr_for_deadline/MAT/src python -m mat.cli doctor --work-root "$MAT_WORK_ROOT"
PYTHONPATH=/data2/usr_for_deadline/MAT/src python -m mat.cli assets plan --catalog /data2/usr_for_deadline/MAT/configs/assets/catalog.yaml --phase P1
# 获得直连/授权文件后：
PYTHONPATH=/data2/usr_for_deadline/MAT/src python -m mat.cli assets import --asset rat_id_v1 --source /authorized/Dataset_128_16x9x10K_Color.zip
PYTHONPATH=/data2/usr_for_deadline/MAT/src python -m mat.cli data inspect --dataset rat_id --work-root "$MAT_WORK_ROOT"
```
