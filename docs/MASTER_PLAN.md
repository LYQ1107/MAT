# MAT 主计划与阶段执行记录

本计划依据 `MAT_CODEX_MASTER_PROMPT_ZH.md`、`MAT_RESOURCE_SEEDS.json` 及本轮续执行指令。目标是：在一次参考录制建档后，对同一 cohort 的不同日期/条件录像保持持久身份，同时输出原图坐标二维身体关键点和可核验运动记录。鱼、牛、猪、鼠切换时只切换 `SpeciesSpec`、骨架和姿态先验，并创建新的 cohort registry；绝不把一个物种的身份迁移给另一个物种。

## 不可改变的边界

1. 大数据、权重和依赖不经 Codex 代理。当前续执行指令只授权官方 SLEAP gerbils 五个数据对象使用现有服务器代理，且只允许 `storage.googleapis.com`；代理仅在下载子进程中保留。系统代理、bashrc、路由、iptables 和其他进程未改动；凭据不打印、不写 receipt、不入 Git。其余依赖尝试使用直连和已存在本地环境；透明代理/TUN 无法从用户态完全排除时，不继续未知大资产。
2. 代码仓库为 `/data2/usr_for_deadline/MAT`，大资产和运行时为 `/data2/usr_for_deadline/MAT_workspace`；研究分支为 `research/longitudinal-mat`。不 force push；数据、权重、私有 GT、账号和 token 不入 Git。
3. gallery 采用 session-start snapshot、`end_of_session` 更新和版本化提交。anchor 永不覆盖/删除；pending 不可作为确认另一个 pending 的主要证据。
4. 预测、来源和私有真值严格分离。没有人工跨日映射、人工核验或姿态真值时使用 `BLOCKED_*`/`null`，不把预测当 GT，不编造训练时长、step 或指标。provided SLEAP tracking 文件不是人工 GT。

## M0 — 机器、网络、源码和 metadata 审计

**状态：`SMOKE_PASSED`。** 已重查实际仓库、分支、`AGENTS.md`、应用代理变量、监听端口、网卡/默认路由、磁盘、Python 环境和 GPU。`nvidia-smi` 显示所有 A100 正被其他任务使用，因此没有触碰 GPU；真实 SLEAP 训练选择 `CUDA_VISIBLE_DEVICES=''` 的 CPU。透明代理/TUN 没有可见证据但不能证明不存在，故未知大资产仍暂停。上游 commit 和实际函数签名写入 `locks/upstream.lock.yaml`、`docs/UPSTREAM_API_MAP.md`；不编造私有 API。

## M1 — 资产层和真实 P1 数据

**状态：`IMPLEMENTED / DOWNLOADED / VERIFIED`（仅 SLEAP P1）。** `AuthorizedProxyPolicy` 与 `DirectOnlyPolicy` 分离，session 的 `trust_env`、host allowlist、重定向和失败收据脱敏均有测试；archive `.part` 后缀校验已修复。CLI 使用 `--network-mode direct-only|authorized-proxy`，默认 direct-only。catalog 中五个 SLEAP URL 为官方入口且 `download_url_verified: true`，provider checksum/expected bytes 未猜测，实际 bytes/SHA 写入 `locks/assets.lock.json` 和工作区 receipts。旧 Rat/Pig/Cow 资产仍未下载。

SLEAP 固定路径：`MAT_workspace/datasets/sleap_gerbils/`。实际对象为 train/val/test `.pkg.slp`、`example_5min.mp4` 和 `example_tracking.slp`；总字节数为 **964,739,339**。下载使用授权代理且 receipt 的 `route_status=AUTHORIZED_PROXY`；所有文件已 SHA-256 复核并以 hardlink 暴露到固定目录。SLEAP-NN/sleap-io、WildlifeTools 和运行时依赖放在工作区隔离环境，不进 Git；当前版本 wheel 已重新用去代理直连保存到 `upstream_audit` wheelhouse。早期启动 runtime 的历史 `pip_download_proxy.log`/安装日志也保留（仅轻量依赖、无数据/权重），不把它当作大资产合规证明，后续禁止复用代理安装。

## M2 — 适配、匿名 manifest、冻结边界和 B0 条件

**状态：`SMOKE_PASSED / BLOCKED_MISSING_IDENTITY_ASSET`。** `SleapGerbilsAdapter` 使用公开 `sleap_io.load_slp`，保持 source filename/video index，生成中性 `frames/sessions/observations` 与仅供评价的 `private_pose_identity_truth`；`observations.jsonl` 不含身份、keypoint GT 或 visibility GT。已对 23 个 source-video sessions 用 seed 17 实际冻结 `13/4/6` source/development/sealed 划分（`f1a04319e89f90a6`），但 provider 文件本身仍是 random frame split，不能解释为 strict longitudinal split。B0 的 global-only static gallery/固定 S0 mapping 链保留；真实 MegaDescriptor-T-224 checkpoint/config 未在本机 cache，HF 直连不可达且未获权重代理授权，所以不以 ImageNet Swin 或随机特征冒充 B0 结果。

## M3 — 真实 SLEAP pose baseline、predict 和 tracking smoke

**状态：`SMOKE_PASSED / PARTIALLY_EVALUATED`。** 只调用官方 `sleap-nn` CLI；首次 `--version`、`config/train/predict/eval --help` 原文和 command receipts 在 `MAT_workspace/upstream_audit/sleap_nn/`。实际版本为 `sleap-nn 0.3.3`。官方 `config --auto --pipeline bottomup` 生成 training config；修复了 val list、typed validation_fraction 和 skia API 兼容性后，CPU 2-epoch smoke 真正完成：`best.ckpt`/`last.ckpt` 在 `runs/sleap_gerbils_pose_smoke/models/260908_181311.bottomup.n=383/`，checkpoint `global_step=2`，optimizer step=2，epoch=0/1，training log 的第 1 行记录 train loss 0.019036、val loss 0.018798。CLI 随后的全量 train/val post-eval 在 CPU 上长时间运行，已有 checkpoint 后被终止，原始 receipt 保留 `return_code=-15`，不记为完整 CLI 成功。

使用真实 checkpoint 对 test labeled frames 的官方 predict 已返回 0，42 labels/0 instances；官方 eval 明确输出 `SUCCEEDED_NO_PREDICTIONS` 且不产生 NPZ，不能报告 pose 数字。对真实 `example_5min.mp4`（1280×1024、25 FPS、2560 帧，实测约 102.4 s）连续帧 0–15 运行了官方 `predict --tracking`，输出 16 帧 SLP、0 instances；这是 **tracking format smoke，不是全片完成，也不是 GT 评价**。完整 2560 帧 CPU 推理未运行，标记 `BLOCKED_CPU_BUDGET`。

## M4 — H/A 建档和跨 session runner

**状态：`IMPLEMENTED / BLOCKED_NEEDS_REFERENCE_REVIEW`。** `ManualRegistrar`、`AutoRegistrar` 共用 `EnrollmentResult` 后端；A 聚类现在在每次加入后重算 prototype，并优先检查 cannot-link。`LongitudinalCohortRunner` 固定读取 session-start snapshot，所有 proposals 在 session 结束后一次性提交。真实 SLEAP 没有人工 S0 核验/跨日生物映射，H-human 和 strict longitudinal accuracy 不运行。

## M5 — 部位证据与安全 memory

**状态：`IMPLEMENTED / BLOCKED_MISSING_IDENTITY_ASSET`。** 新增 `PosePartCropper`（真正的 `torchvision.ops.roi_align`、全局/部位 ROI、finite/score/valid 门控、part quality）和不复制 global embedding 的 `PartAwareIdentityEncoder`；B0 matcher 严格 global-only，新增固定 B1 fusion 与 trainable `EvidenceMatcher`。新增 `LongitudinalGalleryStore`（anchor/confirmed/pending、多 exemplar top-k、fingerprint）和确定性的 `MemoryCommitGate`。没有经过核验的 MegaDescriptor，未生成正式 identity 指标。

## M6 — O2 和外部基线

**状态：`NOT_APPLICABLE`。** O1/真实跨 session 身份数据、许可和姿态/检测输入不足，不启动 O2、idtracker.ai、idmatcherai 或 CowIDentifier。

## M7 — strict split 和联合评价

**状态：`BLOCKED_NEEDS_POSE_GT / BLOCKED_NEEDS_CROSS_DAY_MAPPING`。** SLEAP 官方 train/val/test 只用于 `SLEAP_OFFICIAL_RANDOM_SPLIT_BASELINE`。strict longitudinal split、人工跨日 ID 映射、连续人工 pose GT 尚不存在；`PoseEvaluator`、固定 S0 mapping、unknown/confusion 输出边界已保留，未填假指标。

## M8 — 封存和复现交付

**状态：`RESEARCH_PIPELINE_READY / PARTIALLY_EVALUATED`。** `mat baseline sleap-gerbils --stage all` 和四个 `mat experiment` 入口已生成真实 run receipts；7 个阶段提交已非 force fast-forward 推送，最终远端 SHA 为 `ea91c28975b6ff71f05f418755c9b817e4202d17`。工作区数据、checkpoint、wheelhouse、私有 truth 不提交。最终 `docs/PROGRESS.md` 和 `docs/RESULTS_GERBILS_V0_V2.md` 明列实际事实、阻塞和下一条命令。

## 下一条可执行命令

```bash
export MAT_WORK_ROOT=/data2/usr_for_deadline/MAT_workspace
PYTHONPATH=/data2/usr_for_deadline/MAT/src python -m mat.cli data inspect --dataset sleap_gerbils --work-root "$MAT_WORK_ROOT"
# 有授权的 MegaDescriptor-T-224 config + checkpoint 原件后，先用 GlobalIdentityBackend.from_local 校验架构/SHA/keys，
# 再从 SLEAP 中性 observations 生成 S0/query crops，运行真实 B0_global_static_gallery。
PYTHONPATH=/data2/usr_for_deadline/MAT/src python -m mat.cli split freeze \
  --manifest "$MAT_WORK_ROOT/prepared/sleap_gerbils/manifests/sessions.jsonl" \
  --field session_uid --seed 17
```
