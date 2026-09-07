# 数据资产审计（截至 2026-09-08）

这是资源能力和阻塞表，不是已下载声明。大小/checksum 仅在 provider metadata 已取得并复核后锁定；`null` 表示未知。

| 资产 | 官方入口 | 当前文件状态 | `has_original_video` | `has_verified_persistent_ids` | `has_pose_ground_truth` | A/H/纵向姿态 |
|---|---|---|---:|---:|---:|---|
| Rat ID | [Zenodo 15112879](https://zenodo.org/records/15112879) | `BLOCKED_DIRECT_ROUTE`; provider MD5 `bb58d79cee8d87ed0c10b3aa84eff3f0`，字节数/直链待 API | false | true（crop 标签，待现场复核） | false | A=false/H=false/pose=false |
| PigReID | [Zenodo 18224572](https://zenodo.org/records/18224572) | `BLOCKED_DIRECT_ROUTE`; 少量组和文件 checksum 待 API | false | true（预处理 crop 标签，待映射复核） | false | A=false/H=false/pose=false |
| PigTracking | [Zenodo 18155637](https://zenodo.org/records/18155637) | `BLOCKED_DIRECT_ROUTE`; 1/2/5 FPS 选择待 metadata | true（发布序列） | null（不能从 gt.txt 猜跨日 EID） | false | 端到端待核验 |
| MultiCamCows2024 | [data.bris record](https://data.bris.ac.uk/data/dataset/2inu67jru7a6821kkgehxg3cv2) | `BLOCKED_DIRECT_ROUTE`; observed archive 39,270,266,920 bytes，逐文件目录/映射待核验 | null | null | false | crop 检索可候选；原视频纵向待映射 |

Rat/PigReID 的实验输入明确是 `prelocalized_crops`；标签只供私有评估器，不能证明自动检测、空间共现约束或 A 建档。PigTracking 的 local MOT 可在原件到位后评估，但持久 ID 关系必须有作者证据。所有数据的相机标定、关键点真值和连续姿态目前均未确认。

