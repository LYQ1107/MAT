# MAT 工作约束

本仓库实现一次参考录制建档、跨录制持久身份、二维姿态和可核验纵向测量。以下约束适用于所有自动化工作：

## 资产和网络

- 数据集、权重、Python/CUDA 大包不得经过 Codex 代理。下载只能由 `mat` 的隔离子进程执行；父会话和全局配置不可修改。
- 下载前检查应用代理、命令包装、URL 级 Git 配置、网卡和路由。发现透明代理/TUN 无法排除时，`route_status=DIRECT_ROUTE_UNVERIFIED`，大文件必须 `BLOCKED_DIRECT_ROUTE`，等待核验镜像或授权本地导入。
- 只允许作者入口、记录文件列表或已核验镜像；禁止猜 URL、静默代理回退、打印凭据、将 token/Cookie/签名 URL 写入日志或 Git。
- 所有下载使用 `DirectOnlyPolicy`；流式限额、重定向白名单、`.part` 续传、官方 checksum 与本地 SHA-256、原子发布均是必需条件。已有有效 hash 直接复用。
- 原始数据、私有真值、权重、缓存、环境、SQLite 注册表和大日志只能放在 `MAT_WORK_ROOT`（默认 `/data2/usr_for_deadline/MAT_workspace`），不进入 Git。

## 科研协议

- S0 参考录制建立一次固定身份映射；后续 session 按真实时间因果读取 session 开始快照，session 结束才提交 memory。
- H-human 必须有真实人工核验记录；使用参考真值模拟只能叫 H-oracle-reference。A 不读取测试身份、框、mask 或姿态真值来建档。
- ByteTrack 的 `source_detection_index` 必须随轨迹保留；不得用输出 Kalman 框最近 IoU 猜检测/姿态对应关系。局部轨迹 ID 不等于持久 ID。
- DeepFeatures 的 `forward_batch` 是 `no_grad` 冻结特征接口，不能冒充训练 API。DLC、MegaDescriptor 和其他权重离线加载，缺资产必须显式失败。
- anchor 不可覆盖；quarantine 不参加强匹配；commit 使用 expected version 和原子事件。不同特征空间禁止直接比较。
- `gt_id`、处理组、行为标签只存在私有评估器，模型输入和推断日志使用中性 observation UID。预测不是 GT；缺标签的指标写 `BLOCKED_*` 或 `null`。
- 预定位 Rat/PigReID crop 只能标为 `prelocalized_reid`，不能声称 A 端到端建档。跨日持久 ID、姿态真值、标定缺失要明确报告。

## Git 和运行

- 保留其他工作，不使用 `reset --hard`、`git clean -fdx` 或 force push。提交前检查暂存文件、体积、二进制和疑似凭据。
- 长任务记录命令、PID、日志、seed、step、checkpoint 和恢复点；未运行使用 `null` 加原因。
- 每个阶段更新 `docs/PROGRESS.md`；状态只用 `PLANNED/DOWNLOADED/VERIFIED/IMPLEMENTED/SMOKE_PASSED/RUNNING/SUCCEEDED/FAILED/BLOCKED_*/NOT_APPLICABLE`。

