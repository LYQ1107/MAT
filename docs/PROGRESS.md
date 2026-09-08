# MAT 进度（2026-09-08）

当前总状态：`RESEARCH_PIPELINE_READY / PARTIALLY_EVALUATED`。本轮没有占用他人 GPU，没有把预测当 GT，也没有提交数据、权重或凭据。

## 已推送阶段提交

`e7ffa71` asset policy → `64b025b` gerbil adapter → `6ad87b3` SLEAP baseline → `4d33481` pose-conditioned features → `39dd3cf` evidence matcher → `5fec8b9` safe memory → `ea91c28` report/configuration；随后仅有文档收尾提交。全部提交均为非 force fast-forward，SHA 核验命令为：`git rev-parse HEAD && git ls-remote origin refs/heads/research/longitudinal-mat`。

## M0—M8

| 阶段 | 状态 | 证据/阻塞 |
|---|---|---|
| M0 | `SMOKE_PASSED` | 实际仓库/AGENTS、代理、路由、GPU、上游 commit/API 审计已写入 `doctor.json`、`docs/NETWORK_AUDIT.md`、`locks/upstream.lock.yaml`；所有 A100 忙，透明代理无法完全排除 |
| M1 | `DOWNLOADED / VERIFIED` | 仅授权 `storage.googleapis.com` 的 SLEAP gerbils 五个对象；其余 Rat/Pig/Cow/权重未下载 |
| M2 | `SMOKE_PASSED / BLOCKED_MISSING_IDENTITY_ASSET` | SLEAP adapter、匿名 manifest、公开 API schema audit、23-session 冻结划分 (`f1a04319e89f90a6`, seed 17) 和 global-only B0 链就绪；缺 MegaDescriptor-T-224 原件/跨日映射 |
| M3 | `SMOKE_PASSED / PARTIALLY_EVALUATED` | 官方 SLEAP-NN 0.3.3 2-epoch CPU fit、test predict/eval、连续 clip tracking prefix smoke 已执行；完整 clip 与有实例 pose 指标未完成 |
| M4 | `IMPLEMENTED / BLOCKED_NEEDS_REFERENCE_REVIEW` | H/A 共用 EnrollmentResult、prototype 修复、end-of-session runner 已实现；无人工 S0 核验 |
| M5 | `IMPLEMENTED / BLOCKED_MISSING_IDENTITY_ASSET` | ROIAlign part encoder、B1 fusion、EvidenceMatcher、三层 gallery、commit gate 已实现；无核验 identity checkpoint |
| M6 | `NOT_APPLICABLE` | O2/idtracker.ai/idmatcherai/CowIDentifier 缺输入或许可 |
| M7 | `BLOCKED_NEEDS_POSE_GT / BLOCKED_NEEDS_CROSS_DAY_MAPPING` | 没有连续人工 pose GT 或跨日 biological-ID 映射 |
| M8 | `SUCCEEDED` | 7 个阶段提交及后续文档收尾均为非 force fast-forward push；每次 push 后均执行 `git rev-parse HEAD` 与 `git ls-remote origin refs/heads/research/longitudinal-mat` 核验（当前值以该命令为准） |

## 已实际执行的 SLEAP 数据链

固定目录：`/data2/usr_for_deadline/MAT_workspace/datasets/sleap_gerbils/`。

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `train.pkg.slp` | 617,972,466 | `12afa74d4f1ddf8e5bb5ee6cd8e6909380d03f091d0964e974a069181ff0a8d9` |
| `val.pkg.slp` | 77,860,441 | `36998385d9aefed214b314d031b0ffc8423added29fab61d6513dcf0af6df0be` |
| `test.pkg.slp` | 76,468,071 | `9baec728ba185c5c50c60c6ac36f15df65c9e84320052c5d4ad662f351202aae` |
| `example_5min.mp4` | 189,346,985 | `e3e994382db85636cfd4a5acae1319413de346cd5637ab96f5d000cba23cc081` |
| `example_tracking.slp` | 3,091,376 | `feeedc56b7527b460d068d0b0e2e7c07c3a210f3fc093c71c430202740b58eaa` |

合计 **964,739,339 bytes**。实际 schema 计数：train 340 labeled frames/1,249 instances，val 43/159，test 42/153；合并后 425 unique labeled frames、1,561 instances、23 source-video slots/sessions（其中 11 个有随机 split 标签）、4 个 source identities、14 个 canonical nodes。`example_tracking.slp` 计 2,560 frames/9,744 instances，但 `is_ground_truth=false`，只用于格式/追踪 sanity check。

session-level 冻结划分已实际写入 `MAT_workspace/assets/manifests/splits/f1a04319e89f90a6.json`：13 source、4 development、6 sealed-test session UIDs，manifest SHA-256 为 `8f8aef448e64227b9a38cf1c8ababd023487ac6760157473135361e7fdc44459`。这是按 provider 的 23 个 source-video slots 冻结；不能解释为跨日期 biological-ID split。

合规备注：早期为使 SLEAP runtime 启动而留下了 `pip_download_proxy.log`/`pip_install_*`（轻量 wheel，未含数据或模型权重）；随后已用去代理直连重新取得同版本 wheel 并保存在 `upstream_audit/sleap_nn/wheels/`，后续安装/下载必须只用该直连 wheelhouse。该历史依赖安装不作为大资产合规证明，已在审计目录保留原始日志。

## 真实运行收据

- 统一入口 `mat baseline sleap-gerbils --stage all --device cpu --clip-frames 0-15` 已复用现有 checkpoint，生成 `MAT_workspace/runs/sleap_gerbils_baseline/run_manifest.json`。manifest 记录 hostname、UTC 起止时间、git 状态、五个数据文件 SHA、split SHA、SLEAP-NN/torch 版本、pose checkpoint SHA、2 epochs/2 optimizer steps、官方 test/eval/clip 输出和 gallery/identity 字段；状态为 `PARTIALLY_EVALUATED`，阻塞为 `pose_evaluation_has_no_predicted_instances` 与 `clip_tracking_is_prefix_smoke_only`。四个 `mat experiment b0|b1|b2|o1` 入口均生成独立 receipt，因缺 verified MegaDescriptor-T-224 checkpoint 保持 `BLOCKED_MISSING_IDENTITY_ASSET`。

- 运行时审计：`MAT_workspace/upstream_audit/sleap_nn/version.txt` 为 `sleap-nn 0.3.3`；`config_help.txt`、`train_help.txt`、`predict_help.txt`、`eval_help.txt` 为本机 CLI 原文。
- 训练：`MAT_workspace/runs/sleap_gerbils_pose_smoke/`。官方 stdout 明确 `max_epochs=2 reached`；checkpoint `global_step=2`、optimizer steps=2、epoch=0/1；`training_log.csv` 第 1 个 epoch train loss=0.019036374986171722、val loss=0.018797585740685463；best checkpoint 为 `models/260908_181311.bottomup.n=383/best.ckpt`（104,729,166 bytes）。官方随后全量 train/val post-eval 在 CPU 长时间运行后被终止，`train.command.json` 的 `return_code=-15` 保留，不把它写成完整 CLI 成功。
- test predict：`runs/sleap_gerbils_pose_predict_eval/test_predictions.slp` 42 labels、0 predicted instances（默认 peak threshold 0.2）；0.15+max_instances=4 的复核同样 0 instances。官方 eval 输出 `eval_peak015_max4/metrics_official.json`：`SUCCEEDED_NO_PREDICTIONS`、metrics/NPZ 为 null，故没有 pose 数字可报告。
- continuous tracking smoke：`runs/sleap_gerbils_clip_tracking_smoke/example_5min.predictions.slp`，官方 `predict --tracking --frames 0-15`，真实 16 连续帧、0 instances、return code 0。视频实测 1280×1024、25 FPS、2,560 帧（约 102.4 s）；完整全片 CPU 运行未完成，状态 `BLOCKED_CPU_BUDGET`，不使用 `example_tracking.slp` 计算 HOTA/ID 指标。

## B0/B1/B2、H/A 和 strict protocol

| 项目 | 状态 | 实际事实 |
|---|---|---|
| B0_global_static_gallery | `BLOCKED_MISSING_IDENTITY_ASSET` | 代码和 TEST_FIXTURE 链存在；没有 MegaDescriptor-T-224 官方 config+checkpoint，HF 直连超时，不能用缓存 ImageNet Swin 冒充；persistent accuracy/unknown/confusion 全为 `null` |
| B1_part_static / EvidenceMatcher | `IMPLEMENTED / NOT_RUN` | ROIAlign 部位证据、固定 fusion 和 trainable head 已有契约测试；无 identity 训练数据 |
| B2 safe memory | `IMPLEMENTED / NOT_RUN` | `LongitudinalGalleryStore` 的 anchor/confirmed/pending、multi-exemplar score、`MemoryCommitGate` 和 end-of-session commit 已有；无真实跨 session 结果 |
| H-human | `BLOCKED_NEEDS_REFERENCE_REVIEW` | 未进行人工核验，不填人工秒数 |
| A-auto | `DIAGNOSTIC_ONLY` | 可在中性 rows 上运行，不能替代人工 S0；真实 SLEAP 只有 provider source track names 的 private truth |
| strict longitudinal | `BLOCKED_NEEDS_CROSS_DAY_MAPPING` | 官方 split 是 random frame split，不宣称跨日期泛化 |
| ID-aware pose | `BLOCKED_NEEDS_POSE_GT` | 没有连续人工姿态真值；不把 SLEAP predictions 当 GT |

## 验证

源码与契约测试：`pytest -q` → **20 passed, 2 skipped**。新增测试覆盖首次 session quarantine、第二 session promotion、anchor descriptor 不被 legacy static store 覆盖，以及私有 pose truth 字段的 leakage rejection；目标 runtime 中还做了 ROIAlign、detach、EvidenceMatcher 的最小真实 forward（global `[1,6]`、parts `[1,2,6]`，坐标无梯度），但没有把它写成论文指标。

## 下一条可执行命令

```bash
export MAT_WORK_ROOT=/data2/usr_for_deadline/MAT_workspace
export MEGA_CHECKPOINT=/data2/usr_for_deadline/MAT_workspace/assets/identity/megadescriptor_t_224.ckpt
# 取得并核验与 checkpoint 同目录的 config.json 后，再执行；当前文件不存在，故保持 BLOCKED。
PYTHONPATH=/data2/usr_for_deadline/MAT/src python -m mat.cli experiment b0 \
  --config configs/experiments/gerbils/B0_global_static.yaml \
  --identity-checkpoint "$MEGA_CHECKPOINT" --work-root "$MAT_WORK_ROOT"
```
