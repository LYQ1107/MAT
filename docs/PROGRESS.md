# MAT 进度（2026-09-08）

当前总状态：`RUNNING / FORMAL_POSE_TRAINING`。代码/配置和小型脱敏报告在研究分支；数据、权重、私有 GT、代理凭据均只在 `MAT_workspace`，没有把预测当 GT，也没有触碰其他作业。

## 已推送阶段提交

`e7ffa71` asset policy → `64b025b` gerbil adapter → `6ad87b3` SLEAP baseline → `4d33481` pose-conditioned features → `39dd3cf` evidence matcher → `5fec8b9` safe memory → `ea91c28` report/configuration → `61d6e73` executable gerbil pilot → `3334e82` bounded B0/preprocess → `e80361e` scoped route lock → `5c1cf8e` upstream audit → `2c233cc` detection index/device → `fcf090e` formal receipt retention → `7d56083` query guard → `f55a72b` offline identity runtime → `c646afc` executable pilot commits → `cd3070d` SHA-addressed full artifacts → `6e1012e` formal artifact documentation → `d061af9` blocker/next-command refresh → `2abfa2d` progress receipt → `31334c8` formal epoch-3 record。全部提交均为非 force fast-forward，SHA 核验命令为：`git rev-parse HEAD && git ls-remote origin refs/heads/research/longitudinal-mat`。

## M0—M8

| 阶段 | 状态 | 证据/阻塞 |
|---|---|---|
| M0 | `SMOKE_PASSED` | 实际仓库/AGENTS、代理、路由、GPU、上游 commit/API 审计已写入 `doctor.json`、`docs/NETWORK_AUDIT.md`、`locks/upstream.lock.yaml`；透明代理/TUN 仍无法由用户态完全排除 |
| M1 | `DOWNLOADED / VERIFIED` | SLEAP 五个对象及官方 MegaDescriptor-T-224 config/weights 已在隔离下载子进程中核验；HF commit、bytes、SHA、MIT、redirect origin 写入 `locks/assets.lock.json`/receipts；Rat/Pig/Cow 未下载 |
| M2 | `SMOKE_PASSED / IMPLEMENTED` | SLEAP adapter、匿名 manifest、公开 API schema audit、23-session 冻结划分 (`f1a04319e89f90a6`, seed 17)、真实文件型 B0 runner 和两份 B0 配置就绪 |
| M3 | `RUNNING` | smoke 与 full 命令/检查点已隔离；formal 50 epoch 在 GPU 1 运行，已到 epoch 46 / `global_step=9400`，尚未完成 test/metrics |
| M4 | `IMPLEMENTED / BLOCKED_NEEDS_REFERENCE_REVIEW` | H/A 共用 EnrollmentResult、全参考质量池化、end-of-session runner 已实现；尚无人工 H-human 核验 |
| M5 | `IMPLEMENTED / VERIFIED` | 统一 Mega runtime/preprocess、ROIAlign 部位证据、B1 fusion、EvidenceMatcher、三层 gallery、pending promotion gate 已实现并有测试 |
| M6 | `NOT_APPLICABLE` | O2/idtracker.ai/idmatcherai/CowIDentifier 缺输入或许可 |
| M7 | `BLOCKED_NEEDS_POSE_GT / BLOCKED_NEEDS_CROSS_DAY_MAPPING` | 没有连续人工 pose GT 或跨日 biological-ID 映射 |
| M8 | `RUNNING` | 代码和文档变更待正式训练/test/B0 结果后分阶段提交；此前提交均为非 force fast-forward，推送后核对本地/远端 SHA |

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

- `mat baseline sleap-gerbils --stage train-full --device auto --gpu-index 1 --full-epochs 50` 已启动；父 PID 13938，SLEAP 子 PID 14046，子进程 `CUDA_VISIBLE_DEVICES=1`。启动收据在 `runs/sleap_gerbils_pose_full/formal_training_launch.json`，当前观测到 epoch 46 / `global_step=9400`，实时 command receipt 在子进程结束时补齐。

- 运行时审计：`MAT_workspace/upstream_audit/sleap_nn/version.txt` 为 `sleap-nn 0.3.3`；`config_help.txt`、`train_help.txt`、`predict_help.txt`、`eval_help.txt` 为本机 CLI 原文。`train_help.txt` 已核验 `trainer_config.resume_ckpt_path`。
- MegaDescriptor-T-224：HF revision `3ea58ff6c6195bc748bb86c111ff40c32bdddcba`；config 609 B/SHA `27ef9cc22f677980785e0778fada1bbc03a9f6a294333756f308638f2e83b86c`，weights 204,267,588 B/SHA `62f53e6335d5f8ea4d764c91b442f8daa5ce7f316d388cc890e06c943218190c`，均 `AUTHORIZED_PROXY` 收据，MIT。`IdentityPreprocessSpec` 与 torchvision BCHW bicubic reference 的 audit 为 `EQUIVALENT`（max/mean abs error 0）。
- smoke 训练仍为 `MAT_workspace/runs/sleap_gerbils_pose_smoke/`，官方 stdout `max_epochs=2 reached`；checkpoint `global_step=2`、optimizer steps=2、epoch=0/1，明确只是 smoke。formal 目录为 `MAT_workspace/runs/sleap_gerbils_pose_full/`，已写到 `global_step=9400`/`epoch=46`，50 epoch 仍在运行；不以中间 checkpoint 充当正式终点。
- test predict：`runs/sleap_gerbils_pose_predict_eval/test_predictions.slp` 42 labels、0 predicted instances（默认 peak threshold 0.2）；0.15+max_instances=4 的复核同样 0 instances。官方 eval 输出 `eval_peak015_max4/metrics_official.json`：`SUCCEEDED_NO_PREDICTIONS`、metrics/NPZ 为 null，故没有 pose 数字可报告。
- continuous tracking smoke：`runs/sleap_gerbils_clip_tracking_smoke/example_5min.predictions.slp`，官方 `predict --tracking --frames 0-15`，真实 16 连续帧、0 instances、return code 0。视频实测 1280×1024、25 FPS、2,560 帧（约 102.4 s）；完整全片 CPU 运行未完成，状态 `BLOCKED_CPU_BUDGET`，不使用 `example_tracking.slp` 计算 HOTA/ID 指标。

## B0/B1/B2、H/A 和 strict protocol

| 项目 | 状态 | 实际事实 |
|---|---|---|
| B0_global_static_gallery | `BLOCKED_POSE_TEST_NOT_SUCCEEDED` | 官方 MegaDescriptor-T-224 已核验，file-backed H_oracle_reference runner 已就绪但尚未运行；当前必须等待 full pose test 非零 predictions/metrics，故 persistent accuracy/unknown/confusion 仍为 `null` |
| B1_part_static / EvidenceMatcher | `IMPLEMENTED / NOT_RUN` | ROIAlign 部位证据、固定 fusion 和 trainable head 已有契约测试；无 identity 训练数据 |
| B2 safe memory | `IMPLEMENTED / NOT_RUN` | `LongitudinalGalleryStore` 的 anchor/confirmed/pending、multi-exemplar score、`MemoryCommitGate` 和 end-of-session commit 已有；无真实跨 session 结果 |
| H-human | `BLOCKED_NEEDS_REFERENCE_REVIEW` | 未进行人工核验，不填人工秒数 |
| A-auto | `DIAGNOSTIC_ONLY` | 可在中性 rows 上运行，不能替代人工 S0；真实 SLEAP 只有 provider source track names 的 private truth |
| strict longitudinal | `BLOCKED_NEEDS_CROSS_DAY_MAPPING` | 官方 split 是 random frame split，不宣称跨日期泛化 |
| ID-aware pose | `BLOCKED_NEEDS_POSE_GT` | 没有连续人工姿态真值；不把 SLEAP predictions 当 GT |

## 验证

源码与契约测试：`pytest -q` → **28 passed, 4 skipped**。新增测试覆盖 full/smoke checkpoint 隔离、显式 train-step 参数、authorized-proxy host scope、Mega preprocess、全参考池化、真实 bbox/S0 选择和中性 sample 合约；目标 runtime 已加载官方 Mega 权重并完成 BCHW 预处理等价审计与 ROI forward，但没有把它写成论文指标。

## 下一条可执行命令

```bash
export MAT_WORK_ROOT=/data2/usr_for_deadline/MAT_workspace
export MEGA_CHECKPOINT=/data2/usr_for_deadline/MAT_workspace/assets/identity/megadescriptor_t_224/pytorch_model.bin
# full pose 完成后先运行独立 full test；只有非零实例和官方 metrics 后，才跑 oracle crop diagnostic。
PYTHONPATH=/data2/usr_for_deadline/MAT/src python -m mat.cli baseline sleap-gerbils \
  --stage test-full --device auto --gpu-index 1 \
  --work-root "$MAT_WORK_ROOT" --run-dir "$MAT_WORK_ROOT/runs/sleap_gerbils_baseline"
# 随后才允许严格 H_oracle_reference 的 B0（预测 pose 版本仍需另行审计）。
PYTHONPATH=/data2/usr_for_deadline/MAT/src python -m mat.cli experiment b0 \
  --config configs/experiments/gerbils/B0_oracle_crop_diagnostic.yaml \
  --identity-checkpoint "$MEGA_CHECKPOINT" --work-root "$MAT_WORK_ROOT"
```
