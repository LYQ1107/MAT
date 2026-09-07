# MAT 总计划与执行记录

版本基于 `MAT_CODEX_MASTER_PROMPT_ZH.md`（2026-09-08）和 `MAT_RESOURCE_SEEDS.json`，不是已完成实验报告。目标是：给同一群动物在参考录制 S0 建立一次 H/A 身份档案，随后在真实日期/条件的独立录像中按时间因果保持持久 ID，同时输出原图坐标二维关键点和可核验纵向运动记录。鱼、牛、猪、鼠切换时使用不同 `SpeciesSpec`、骨架和姿态先验，并建立新的 cohort registry，绝不迁移生物身份。

## 不可改变的边界

1. 数据集、权重、依赖大包不经过 Codex 代理；只由 `DirectOnlyPolicy` 子进程下载。父会话的代理、路由、系统文件和他人训练不改。应用代理已发现，默认路由虽指向 `eno1`，透明代理/TUN 尚未证明不存在，因此大资产先阻塞，允许小元数据和用户授权本地导入。
2. 远端 Git 仓库先重核查；代码路径 `/data2/usr_for_deadline/MAT`，大资产路径 `/data2/usr_for_deadline/MAT_workspace`。数据、私有 GT、凭据和权重永不 Git。提交只走研究分支，不 force push。
3. 主协议是 chronological、`offline_within_session`、`end_of_session` 更新：Ss 只能读 Ss 开始的 gallery 快照；已发布预测不可由未来 session 重写。
4. H-human 需要真实人工确认和时长；oracle 只叫 H-oracle-reference。A 不读取测试身份目录、框、mask 或姿态真值。预定位 Rat/PigReID crop 实验永远标 `input=prelocalized_crops`、`protocol=prelocalized_reid`，不代替端到端 A。
5. 预测、GT、来源和状态分离。缺少跨日身份映射、人工核验、姿态真值或标定时使用 `BLOCKED_*`/`null`，不编造指标、训练 step、人工分钟数或毫米单位。

## 交付链和状态

每个阶段写 `docs/PROGRESS.md`，产出 run/asset receipt；状态统一为 `PLANNED`、`DOWNLOADED`、`VERIFIED`、`IMPLEMENTED`、`SMOKE_PASSED`、`RUNNING`、`SUCCEEDED`、`FAILED`、`BLOCKED_*`、`NOT_APPLICABLE`。每个 run 记录 commit、dirty hash、实验/seed/scope、split hash、输入模式、H/A 模式、模型/环境 hash、registry 版本、网络策略、时间、实际 optimizer steps/checkpoint、metrics 和 blockers。

## M0 — 安全、机器、仓库与上游审计（当前已执行）

**动作**：重查远端和本地工作树；建立代码/资产目录；检查用户、Python、Git、curl/wget、网卡/路由、磁盘/inode、GPU 和活动进程；仅做小请求；检查应用 proxy、URL 级 Git 配置、curl/wget 配置、LD_PRELOAD/TUN/透明代理迹象；锁定上游源码 commit 和函数签名。

**交付**：`doctor.json`、`docs/NETWORK_AUDIT.md`、`docs/UPSTREAM_API_MAP.md`、`configs/assets/catalog.yaml`、`locks/upstream.lock.yaml`、`locks/assets.lock.json`、`docs/BLOCKERS.md`。

**现场结果**：GitHub `main` 通过小型 `ls-remote` 重核查为空；应用层 `http_proxy/https_proxy/all_proxy`（大小写）存在，本地有 7890/7891 监听；`LD_PRELOAD` 未设置，接口仅 `eno1/eno2/lo`，默认路由经 `202.205.84.1`，没有可见 TUN；直连 GitHub 30 秒超时。故 `application_proxy_bypass=true`，`route_status=DIRECT_ROUTE_UNVERIFIED`，大资产 `BLOCKED_DIRECT_ROUTE`。GPU 0 有 A100 40GB 且正在使用，不启动训练，不碰其他 GPU 进程。

**通过条件**：代理/路由风险写清；没有秘密；每个拟下载资产有入口、许可证、大小/哈希未知标记和预算；不把规划字段写成已下载。

## M1 — 无代理资产层、元数据和离线资产（已实现代码，资产受阻）

实现 `DirectOnlyPolicy`、`AssetCatalog`、限额 `AssetDownloader.probe/fetch`、安全重定向、Range/续传、sha256/官方校验、zip-slip 检查、原子 receipt 和 `import_local`。先拿作者 metadata/README 和源代码版本；不进行 9–50GB 数据或未知权重传输。只有直连路由得到管理员核验，或用户给出授权原件，才进入 P1 下载。依赖用隔离 wheelhouse；推断设离线变量并缺权重失败。

**P1 选择规则**：Rat ID（MD5 由官方 API 实时锁定）＋ PigReID 少量跨日组＋一套轻量身份权重。先索引而不全量解压；四组的源/dev/封存划分由实际 metadata 固化，绝不利用 EID 路径给模型喂标签。

## M2 — 适配、匿名清单、冻结划分与真实 B0 pilot（代码就绪，数据受阻）

`DatasetAdapter` 分离 observations/source labels/reference annotations/private truth；Rat/PigReID/PigTracking/MultiCamCows 适配器只暴露中性 UID；schema 断言、重复/同事件泄漏检查和按 group/identity/session 的冻结 split。B0 为冻结全局 descriptor＋静态 gallery＋固定 S0 mapping，支持 unknown 和 scope=pilot。没有真实 P1 原件时不运行伪造结果，run manifest 记 `BLOCKED_MISSING_ASSET`。

## M3 — 姿态/检测、ByteTrack 索引补丁和源 head

锁定 DeepLabCut PyTorch/SuperAnimal 真实兼容版本，使用本地 pose/detector/backbone checkpoint；核验 h5/JSON 字段、骨架语义、xyxy↔xywh 和 crop→全图变换。补丁只给 ByteTrack `STrack` 传播原始 detection index（并固定 dtype），保存 patch 和上游 commit；块处理保持 session tracker 状态，跨 session reset。B2 只训练冻结 backbone 的 source projection/fusion，真实反传才记录 step/checkpoint。

## M4 — H/A 建档与跨录制 B0/B1

实现 `ManualRegistrar`（contact sheet、中性 UID、S0-only verification JSON）与 `AutoRegistrar`（质量/覆盖代表帧、cannot-link、保守约束凝聚聚类）；同一 `EnrollmentResult` 进入 registry。CohortRunner 逐 session 读取快照并输出 tracklets/assignments。B1 仅作开发集固定质量 EMA/有限 exemplar 的朴素更新；更新污染计入指标。

## M5 — B3 部位证据与 O1 隔离式 memory

实现冻结 MegaDescriptor 全局＋可见语义部位 crop、无共同部位时 global fallback、姿态只对齐身份外观；`ConflictGraphBuilder` 和可解释受约束分配允许不重叠碎片共享 ID、同相机同时冲突，unknown 是每轨迹自己的选项。`GalleryStore` 的 anchors/quarantine/committed、expected-version commit、rollback、事件日志和 encoder fingerprint 完整可审计。B0/B1/B2/B3/O1 共用缓存，每次消融只变一个因素。

## M6 — O2 与强外部基线（条件执行）

仅 O1 有可靠数据和自监督信号才尝试 O2；保留 anchor 编码器或从所有历史 crop 重编码后原子切换 gallery，严禁跨 fingerprint 直接 cosine。独立环境真实运行 idtracker.ai/idmatcherai 和适用 CowIDentifier；缺原视频/session 或许可不符写 `NOT_APPLICABLE`，不伪造 session。

## M7 — 封存正式实验与联合姿态评价

冻结 split/阈值/预算后运行至少预设 seeds（17/42/2026），共同缓存可复用。局部 MOT（TrackEval）与持久 ID 评价分开；S0 mapping 永久冻结，报告漏检/unknown/错认/拒绝/连接延迟/持续时长/建档重复混合。PoseEvaluator 只用真实人工/辅助 GT；IdentityAwarePoseEvaluator 同时要求实例、持久 ID、部位和预定误差。缺姿态 GT 则交可执行 annotation manifest 和 `BLOCKED_NEEDS_POSE_GT`。

## M8 — 论文级可复现交付

清理仅 MAT 中间产物（不删除用户原数据）；补齐许可证、依赖锁、RUNBOOK/METHOD/claim-evidence、预测导出和图表脚本。研究分支分阶段提交，暂存审计、远端 SHA 核验；最终状态仅在必要阶段实际满足时叫 `COMPLETE`，否则为 `RESEARCH_PIPELINE_READY / PARTIALLY_EVALUATED`。

## 当前执行顺序

1. 完成 M0 文档、doctor 和真实上游 commit/API 小核验。
2. 完成 M1 资产策略、catalog、receipts 和本地导入；直连未核验不下大文件。
3. 完成 M2 适配/匿名清单/冻结 split/B0 代码和 fixture 契约测试；等待 P1 原件后立即跑真实 pilot。
4. 在不依赖大资产的范围继续 M4/M5 核心 registry、匹配和评价逻辑；M3/M6/M7 的外部运行保持明确阻塞。

