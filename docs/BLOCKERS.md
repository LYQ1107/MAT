# 当前阻塞

| blocker | 状态 | 恢复入口 |
|---|---|---|
| 外部透明代理/TUN 未核验 | `BLOCKED_DIRECT_ROUTE` | 网络管理员确认直连出口/核验镜像；更新 `route_status=VERIFIED_DIRECT` 后才 fetch |
| P1 原始 Rat/PigReID | `BLOCKED_MISSING_ASSET` | 用户提供授权本地文件，执行 `mat assets import --asset ... --source ...` |
| MegaDescriptor 权重/config | `BLOCKED_MISSING_ASSET` | 导入完整官方 revision/checksum；离线加载前不运行身份模型 |
| DLC pose/detector/backbone | `BLOCKED_MISSING_ASSET` | 导入已校验本地 checkpoint/config；不使用 model zoo 自动下载 |
| H-human 操作 | `BLOCKED_NEEDS_REFERENCE_REVIEW` | 仅 S0 contact sheet 完成分组 JSON；未操作不填人工时间 |
| PigTracking 跨日 map | `BLOCKED_NEEDS_PERSISTENT_TRUTH` | 核验作者 localID→EID 说明/私有补标 |
| 联合纵向姿态 | `BLOCKED_NEEDS_POSE_GT` | 按 `annotation/export` 生成连续片段并人工导入；预测不能当 GT |

已实现的逻辑、fixture 契约和本地导入不依赖以上外部状态；不会以合成 fixture 代替 P1 真实结果。

