# 实验矩阵（预注册草案）

| ID | 固定前端 | 身份证据 | memory | 输入/状态 |
|---|---|---|---|---|
| B0 | DLC/SuperAnimal+ByteTrack | 冻结 global | static anchor | pilot/full |
| B1 | 同 B0 | global | 开发期固定规则 EMA/exemplars | pilot/full |
| B2 | 同缓存 | source-trained projection | static | 需真实 source labels |
| B3 | 同缓存 | global+part | static | 需姿态 cache |
| O1 | 同缓存 | 部位+冲突约束 | quarantine/commit | 主方法 |
| O2 | 同 O1 | 安全无标注适配 | versioned | 仅有有效学习信号才启用 |
| E1 | 官方 idtracker.ai | 官方 matcher | 外部 | 真实 session only |
| E2 | 作者 CowIDentifier/适用 ReID | 按源码 | 外部 | 泄漏审计后 |

B0–O1 共用检测/姿态/tracklet cache、source/dev/test、S0 budget；每个消融只改一个因素。预定 seeds 为 17/42/2026，冻结推断共同 cache 不重复伪造随机差异。

