# MAT 进度（2026-09-09）

当前总状态：`SUCCEEDED / STRICT_B0_B1_SEALED_RECEIPTS`。正式姿态训练、full test、严格 B0 oracle/predicted 与 B1 oracle/predicted 均已真实完成；H-human、独立跨日 biological-ID 映射、连续姿态 GT 和 B2/O1 正式运行仍按实际缺口保留阻塞。代码/配置和小型脱敏报告在研究分支；数据、权重、私有 GT、代理凭据均只在 `MAT_workspace`，没有把预测当 GT，也没有触碰其他作业。

最新更正与 sealed 结果见 [`RESULTS_PROTOCOL_CORRECTION.md`](RESULTS_PROTOCOL_CORRECTION.md)。旧 `0.599315` F1 仅保留为 `INVALID_NONSTANDARD_F1` legacy receipt。

## 已推送阶段提交

`e7ffa71` asset policy → `64b025b` gerbil adapter → `6ad87b3` SLEAP baseline → `4d33481` pose-conditioned features → `39dd3cf` evidence matcher → `5fec8b9` safe memory → `ea91c28` report/configuration → `61d6e73` executable gerbil pilot → `3334e82` bounded B0/preprocess → `e80361e` scoped route lock → `5c1cf8e` upstream audit → `2c233cc` detection index/device → `fcf090e` formal receipt retention → `7d56083` query guard → `f55a72b` offline identity runtime → `c646afc` executable pilot commits → `cd3070d` SHA-addressed full artifacts → `6e1012e` formal artifact documentation → `d061af9` blocker/next-command refresh → `2abfa2d` progress receipt → `31334c8` formal epoch-3 record → `428960c` heterogeneous B0 ROI batching + final diagnostic。全部提交均为非 force fast-forward，SHA 核验命令为：`git rev-parse HEAD && git ls-remote origin refs/heads/research/longitudinal-mat`。

## M0—M8

| 阶段 | 状态 | 证据/阻塞 |
|---|---|---|
| M0 | `SMOKE_PASSED` | 实际仓库/AGENTS、代理、路由、GPU、上游 commit/API 审计已写入 `doctor.json`、`docs/NETWORK_AUDIT.md`、`locks/upstream.lock.yaml`；透明代理/TUN 仍无法由用户态完全排除 |
| M1 | `DOWNLOADED / VERIFIED` | SLEAP 五个对象及官方 MegaDescriptor-T-224 config/weights 已在隔离下载子进程中核验；HF commit、bytes、SHA、MIT、redirect origin 写入 `locks/assets.lock.json`/receipts；Rat/Pig/Cow 未下载 |
| M2 | `SMOKE_PASSED / IMPLEMENTED` | SLEAP adapter、匿名 manifest、公开 API schema audit、11 个有标签 session 的冻结纵向 protocol (`92dd746577047d74`, seed 17；1/6/2/2 roles)、真实文件型 B0 runner 和严格配置就绪 |
| M3 | `SUCCEEDED` | smoke 与 full 命令/检查点已隔离；formal 50 epoch 已在 GPU 1 完成（最后日志 epoch 49 / `global_step=10000`）；full test 产生 42 labeled frames/197 instances，官方 eval `SUCCEEDED`（mOKS 0.4020、OKS mAP 0.1087、mPCK 0.2567）；Popen 与 NPZ object-format 兼容问题均已修复 |
| M4 | `IMPLEMENTED / BLOCKED_NEEDS_REFERENCE_REVIEW` | H/A 共用 EnrollmentResult、全参考质量池化、end-of-session runner 已实现；尚无人工 H-human 核验 |
| M5 | `IMPLEMENTED / VERIFIED / B1_RUN` | 统一 Mega runtime/preprocess、ROIAlign 部位证据、B1 fixed fusion、EvidenceMatcher、三层 gallery、pending promotion gate 已实现并有测试；B1 oracle/predicted 均完成 sealed-only receipt |
| M6 | `NOT_APPLICABLE` | O2/idtracker.ai/idmatcherai/CowIDentifier 缺输入或许可 |
| M7 | `PARTIAL / BLOCKED_NEEDS_POSE_GT / BLOCKED_NEEDS_CROSS_DAY_MAPPING` | sparse provider-labeled pose/identity rows 可做 frame-instance 评价；dense continuous pose GT、独立跨日 mapping 不存在，不能把预测当 GT |
| M8 | `SUCCEEDED / SEALED` | strict B0/B1 四份 sealed receipt、协议修正、预测姿态契约、几何 evaluator、raw-video/over-detection audit 已封存；B1 oracle 仍是诊断，不是端到端 H/A |

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

历史 session-level 审计划分 `f1a04319e89f90a6.json`（13/4/6）仍保留作 receipt，但不用于本轮 strict 结果。当前 strict protocol 写入 `MAT_workspace/assets/manifests/splits/gerbils_longitudinal_v1.json`：1 reference、6 source、2 development、2 sealed-test（protocol `92dd746577047d74`；session manifest SHA-256 `a13ffe4025c111f9c34c52c02e3cf633636edf0ec6fcf02feefae2f4a7fa85f7`，truth SHA-256 `329d7d63af61f86ba5d58bc2d1f8d4fcb77a28f744fe70b68aaec8054600e29f`）。provider datetime 全缺失，ordering basis 为 UID fallback；不能解释为跨日期 biological-ID split。

合规备注：早期为使 SLEAP runtime 启动而留下了 `pip_download_proxy.log`/`pip_install_*`（轻量 wheel，未含数据或模型权重）；随后已用去代理直连重新取得同版本 wheel 并保存在 `upstream_audit/sleap_nn/wheels/`，后续安装/下载必须只用该直连 wheelhouse。该历史依赖安装不作为大资产合规证明，已在审计目录保留原始日志。

## 真实运行收据

- `mat baseline sleap-gerbils --stage train-full --device auto --gpu-index 1 --full-epochs 50` 已完成；父 PID 13938、SLEAP 子 PID 14046，子进程 `CUDA_VISIBLE_DEVICES=1`，训练 return code 0。收据在 `runs/sleap_gerbils_pose_full/formal_training_launch.json`，最后日志 epoch 49 / `global_step=10000`，正式 checkpoint SHA-256 为 `0399099f64a283656c86fc901753b0d835e33c29b11890ff95120f98badf1d10`。
- 首次自动 full test 于训练退出后触发，但 `SleapNNBackend._run` 使用了 `Popen(capture_output=True)`，被 Python 拒绝而以 return code 1 结束；已改为显式 `stdout/stderr=subprocess.PIPE`，新增回归测试。
- 修复 Popen 后的 full test 已完成预测并生成官方 metrics，但后端以 `allow_pickle=False` 读取 SLEAP legacy object NPZ 时失败；已核验官方 JSON sidecar 格式并改为优先读取 sidecar（仅保留本地 evaluator 输出的兼容回退），新增 eval 回归测试。最终 full test 已成功完成：42 labeled frames、197 instances，验证阈值 0.10 时 43 frames/198 instances；checkpoint SHA 为 `0399099f64a283656c86fc901753b0d835e33c29b11890ff95120f98badf1d10`。
- full test 官方 metrics（仅 SLEAP 官方 random frame split，不是纵向泛化）：mOKS `0.4020083039400739`，OKS VOC mAP/mAR `0.1086789079250195/0.1392156862745098`，PCK VOC mAP/mAR `0.019913079220009913/0.023529411764705882`，平均关键点距离 `8.790849863678831 px`（p50 `4.6945187969428765`、p95 `21.483785319651368`），mPCK `0.25666023166023166`（PCK@5 `0.27075289575289574`、PCK@10 `0.4107142857142857`），visibility precision/recall `0.9832089552238806/0.6078431372549019`。

- 运行时审计：`MAT_workspace/upstream_audit/sleap_nn/version.txt` 为 `sleap-nn 0.3.3`；`config_help.txt`、`train_help.txt`、`predict_help.txt`、`eval_help.txt` 为本机 CLI 原文。`train_help.txt` 已核验 `trainer_config.resume_ckpt_path`。
- MegaDescriptor-T-224：HF revision `3ea58ff6c6195bc748bb86c111ff40c32bdddcba`；config 609 B/SHA `27ef9cc22f677980785e0778fada1bbc03a9f6a294333756f308638f2e83b86c`，weights 204,267,588 B/SHA `62f53e6335d5f8ea4d764c91b442f8daa5ce7f316d388cc890e06c943218190c`，均 `AUTHORIZED_PROXY` 收据，MIT。`IdentityPreprocessSpec` 与 torchvision BCHW bicubic reference 的 audit 为 `EQUIVALENT`（max/mean abs error 0）。
- smoke 训练仍为 `MAT_workspace/runs/sleap_gerbils_pose_smoke/`，官方 stdout `max_epochs=2 reached`；checkpoint `global_step=2`、optimizer steps=2、epoch=0/1，明确只是 smoke。formal 目录为 `MAT_workspace/runs/sleap_gerbils_pose_full/`，50 epoch 已完成，最后日志 epoch 49 / `global_step=10000`；不以 smoke 或中间 checkpoint 代替正式模型。
- test predict：旧 smoke checkpoint 在 0.2 与 0.15+max_instances=4 均为 0 instances，仍只作为 smoke 事实；正式 full test 使用 full checkpoint 并得到上述 197 instances/官方 metrics。官方 eval 输出 `eval_peak015_max4/metrics_official.json`：`SUCCEEDED_NO_PREDICTIONS`，不能把 smoke 当 formal 指标。
- continuous tracking smoke：`runs/sleap_gerbils_clip_tracking_smoke/example_5min.predictions.slp`，官方 `predict --tracking --frames 0-15`，真实 16 连续帧、0 instances、return code 0。视频实测 1280×1024、25 FPS、2,560 帧（约 102.4 s）；完整全片 CPU 运行未完成，状态 `BLOCKED_CPU_BUDGET`，不使用 `example_tracking.slp` 计算 HOTA/ID 指标。

## B0/B1/B2、H/A 和 strict protocol

| 项目 | 状态 | 实际事实 |
|---|---|---|
| B0_global_static_gallery | `SUCCEEDED / LEGACY_ORACLE_DIAGNOSTIC` | 历史 all-non-S0 诊断保留作 receipt；1428 query 的 `0.599315` F1 为 `INVALID_NONSTANDARD_F1`，不可与 strict sealed 指标比较 |
| B0/B1 strict static pilots | `SUCCEEDED / SEALED_ONLY` | B0 oracle/predicted 与 B1 oracle/predicted 均使用 protocol `92dd746577047d74`、同一 v0 reference gallery、development-only calibration；逐行数字见 `RESULTS_PROTOCOL_CORRECTION.md` |
| B1_part_static / EvidenceMatcher | `B1_FIXED_FUSION_SUCCEEDED / EvidenceMatcher_NOT_RUN` | ROIAlign 部位证据、固定 quality-weighted fusion 和 trainable head 均已实现；固定 fusion oracle/predicted sealed receipts 已完成，EvidenceMatcher 无 identity 训练数据 |
| B2 safe memory | `IMPLEMENTED / NOT_RUN` | `LongitudinalGalleryStore` 的 anchor/confirmed/pending、multi-exemplar score、`MemoryCommitGate` 和 end-of-session commit 已有；无真实跨 session 结果 |
| H-human | `BLOCKED_NEEDS_REFERENCE_REVIEW` | 未进行人工核验，不填人工秒数 |
| A-auto | `DIAGNOSTIC_ONLY` | 可在中性 rows 上运行，不能替代人工 S0；真实 SLEAP 只有 provider source track names 的 private truth |
| strict longitudinal | `BLOCKED_NEEDS_CROSS_DAY_MAPPING` | 官方 split 是 random frame split，不宣称跨日期泛化 |
| ID-aware pose | `BLOCKED_NEEDS_POSE_GT` | 没有连续人工姿态真值；不把 SLEAP predictions 当 GT |

## 验证

源码与契约测试：最终 `PYTHONPATH=src pytest -q` → **47 passed, 4 skipped**（2026-09-09）。测试覆盖 full/smoke checkpoint 隔离、显式 train-step 参数、authorized-proxy host scope、Mega preprocess、异形 ROI 批量编码、全参考池化、冻结 roles、预测姿态契约、geometry-only matching、B0/B1 sealed-only 查询、预测 UID/无歧义 session 映射和中性 sample 合约。目标 runtime 已加载官方 Mega 权重并完成 BCHW 预处理等价审计与 ROI forward，但没有把它写成论文指标。

## 下一条可执行命令

```bash
export MAT_WORK_ROOT=/data2/usr_for_deadline/MAT_workspace
export MEGA_CHECKPOINT=/data2/usr_for_deadline/MAT_workspace/assets/identity/megadescriptor_t_224/pytorch_model.bin
# 历史复现命令：正式训练/full test receipt 已封存；该命令不会自动重跑训练。
PYTHONPATH=/data2/usr_for_deadline/MAT/src python -m mat.cli baseline sleap-gerbils \
  --stage test-full --device auto --gpu-index 1 \
  --work-root "$MAT_WORK_ROOT" --run-dir "$MAT_WORK_ROOT/runs/sleap_gerbils_baseline"
# 严格 B0/B1 receipts 已封存；新增运行须继续使用冻结 protocol 和 development-only calibration。
PYTHONPATH=/data2/usr_for_deadline/MAT/src python -m mat.cli experiment b0 \
  --config configs/experiments/gerbils/B0_oracle_crop_diagnostic.yaml \
  --identity-checkpoint "$MEGA_CHECKPOINT" --work-root "$MAT_WORK_ROOT"
```

## 本轮续执行的实际收据（2026-09-09）

- P0/E0 标准指标代码与测试已推送：`4cd0ce00cbd4c3f9db803675ca1cdb544f9eab34`；P1/E1 冻结 protocol 已推送：`5695071`。reference/development/sealed 角色固定为 `1/2/2`（另有 6 个 source），ordering basis 为 UID fallback，不是 provider date order。
- 严格 B0 oracle：`runs/b0_oracle_strict_gerbils/`，sealed truth 252，accepted/correct/wrong/unknown=`238/108/130/14`，accuracy=`0.42857142857142855`，macro-F1=`0.4017191327896662`，micro-F1=`0.44081632653061226`，development threshold=`0.21345680952072144`。
- 严格 B0 predicted-pose：`runs/b0_predicted_pose_strict_gerbils/`，parser 读取 197 个预测实例，sealed geometry located=26，accepted/correct/wrong/unknown=`25/9/16/227`，accuracy=`0.03571428571428571`，macro-F1=`0.05581527347781218`，micro-F1=`0.06498194945848375`，development threshold=`0.1338246464729309`。
- 严格 B1 oracle part：`runs/b1_oracle_part_strict_gerbils/`，accepted/correct/wrong/unknown=`237/117/120/15`，accuracy=`0.4642857142857143`，macro-F1=`0.4218544051606998`，micro-F1=`0.4785276073619632`，development 选择 global weight=`0.0`、threshold=`0.3501112461090088`。
- 严格 B1 predicted part：`runs/b1_predicted_part_strict_gerbils/`，geometry located=26，accepted/correct/wrong/unknown=`26/12/14/226`，accuracy=`0.047619047619047616`，macro-F1=`0.07177033492822966`，micro-F1=`0.08633093525179857`，development 选择 global weight=`0.5`、threshold=`0.09584712237119675`。
- `pose_detection_diagnostic.json` 复核 official test 的 153 GT vs 197 predictions：42 frames 中 27 over-detect、1 under-detect、14 equal；完整解释见 `docs/runtime/GERBIL_POSE_DETECTION_DIAGNOSTIC.md`。原始 23-slot 视频可用性见 `docs/runtime/GERBIL_RAW_VIDEO_AUDIT.md`。
- 预测 pilot 收据已在 UID 契约修正后刷新：预测分配只保留 `prediction_uid`；仅几何 evaluator 产生的 `evaluator_match.truth_observation_uid` 可连接 sealed truth。`b0_predicted_pose_strict_gerbils` 与 `b1_predicted_part_strict_gerbils` 的 sealed 数字不变（分别 9/16/227 与 12/14/226 的 correct/wrong/unknown）。
- B2/O1 阻塞命令已实际执行并封存：`runs/b2_s0_matcher_strict_gerbils/run_manifest.json`、`runs/o1_static_memory_gerbils/run_manifest.json` 均为 `BLOCKED_EXPERIMENT_NOT_IMPLEMENTED`；缺少独立跨日身份映射、source identity 标注和 dense pose GT，未伪造训练或测试指标。

## 当前下一条可执行命令

```bash
export MAT_WORK_ROOT=/data2/usr_for_deadline/MAT_workspace
# 仅在新增、授权且独立的 source identity 标注和连续 pose GT 到位后，才解除 B2/O1 阻塞；不得用 sealed 结果反向调参。
PYTHONPATH=/data2/usr_for_deadline/MAT/src python -m mat.cli experiment b2 \
  --config configs/experiments/gerbils/B2_s0_matcher_strict.yaml --work-root "$MAT_WORK_ROOT"
```

在这些输入到位前，上述命令应保持 `BLOCKED_EXPERIMENT_NOT_IMPLEMENTED`/`NOT_RUN`，不伪造训练 step 或纵向指标。
