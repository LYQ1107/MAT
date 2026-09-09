# SLEAP gerbils：V0–V2 结果表

本表只记录真实执行事实。官方 random frame split 结果不能解释为 strict longitudinal generalization；`example_tracking.slp` 不是人工 GT。`NOT_RUN`/`null` 表示没有满足数据、权重或标注条件，不是零分。

`B0_global_static_gallery` 旧行是历史 all-non-S0 diagnostic：其 F1
`0.5993150684931506` 明确为 `INVALID_NONSTANDARD_F1`，不可与下方标准
sealed-only 结果比较；更正说明和原始 receipt 位置见
[`RESULTS_PROTOCOL_CORRECTION.md`](RESULTS_PROTOCOL_CORRECTION.md)。

| Method | Pose source | ID feature | Matcher | Memory | Enrollment | Persistent ID | Unknown | ID-aware Pose |
|---|---|---|---|---|---|---|---|---|
| SLEAP_OFFICIAL_RANDOM_SPLIT_BASELINE (smoke) | SLEAP-NN 0.3.3 bottomup，2 epochs CPU，`global_step=2` | NOT_RUN（smoke 仅为姿态链路验证） | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| SLEAP_OFFICIAL_RANDOM_SPLIT_BASELINE (formal) | SLEAP-NN 0.3.3 bottomup，GPU 1，50 epochs；full test 42 labeled frames/197 instances，官方 eval `SUCCEEDED`：mOKS 0.4020、OKS mAP 0.1087/mAR 0.1392、mPCK 0.2567、平均距离 8.7908 px | NOT_RUN（pose baseline 之外未训练 identity） | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| B0_global_static_gallery | formal full pose checkpoint；oracle crops 仅用于诊断 | MegaDescriptor-T-224（官方本地文件已校验） | global cosine（static gallery） | static S0 snapshot（gallery `v0→v0`） | `H_oracle_reference`，S0=`0b1d08252e086969ba10`，anchors=`16/16/9/16` | accuracy `0.36764705882352944`（1428 query instances，correct 525） | `201` | NOT_RUN — `BLOCKED_NEEDS_POSE_GT` |
| B0 oracle strict (corrected) | oracle GT pose bbox；固定 protocol sealed-only | MegaDescriptor-T-224 | global cosine + deterministic greedy | frozen S0 | `H_oracle_reference`，sealed truth=252 | accuracy `0.428571` / macro-F1 `0.401719` / micro-F1 `0.440816`；accepted/correct/wrong/unknown=`238/108/130/14` | `14` | pose GT 仅为诊断上限 |
| B0 predicted-pose strict | SLEAP full-checkpoint predicted instances；末端 geometry evaluator | MegaDescriptor-T-224 | global cosine + deterministic greedy | frozen S0 | predicted 197；sealed truth=252，located=26 | accuracy `0.035714` / macro-F1 `0.055815` / micro-F1 `0.064982`；accepted/correct/wrong/unknown=`25/9/16/227` | `227` | geometry matched after ID |
| B1 oracle part strict | oracle GT pose + ROIAlign parts（诊断） | global + head/trunk/tail parts | fixed quality-weighted fusion；development weight=`0.0` | frozen S0 | sealed truth=252 | accuracy `0.464286` / macro-F1 `0.421854` / micro-F1 `0.478528`；accepted/correct/wrong/unknown=`237/117/120/15` | `15` | not end-to-end H/A |
| B1 predicted part strict | SLEAP predicted keypoints + ROIAlign parts | global + head/trunk/tail parts | fixed quality-weighted fusion；development weight=`0.5` | frozen S0 | predicted 197；sealed truth=252，located=26 | accuracy `0.047619` / macro-F1 `0.071770` / micro-F1 `0.086331`；accepted/correct/wrong/unknown=`26/12/14/226` | `226` | geometry matched after ID |
| B1 EvidenceMatcher | NOT_RUN | global/part pair MLP（实现但未训练） | trainable evidence head | static | NOT_RUN | NOT_RUN | null | NOT_RUN |
| B2 safe longitudinal memory | NOT_RUN | NOT_RUN | NOT_RUN | anchor/confirmed/pending + end-of-session（anchor update 不可在本轮伪造） | NOT_RUN | NOT_RUN | null | NOT_RUN |

## 可复核运行事实

- 数据：`MAT_workspace/datasets/sleap_gerbils/` 五个官方对象合计 964,739,339 bytes；SHA 和 split 计数见 [`docs/PROGRESS.md`](PROGRESS.md)。
- smoke checkpoint：`MAT_workspace/runs/sleap_gerbils_pose_smoke/models/260908_181311.bottomup.n=383/best.ckpt`；训练 log 的第 1 个 epoch train/val loss 分别为 0.019036374986171722/0.018797585740685463。两步 optimizer 确认于 checkpoint `global_step=2`，只用于 smoke。
- formal checkpoint 目录：`MAT_workspace/runs/sleap_gerbils_pose_full/`；GPU 1 的正式训练命令、PID 和实时 log 在 `formal_training_launch.json` 及其模型目录，50 epochs 已完成，最后日志 epoch 49 / `global_step=10000`；checkpoint SHA-256 为 `0399099f64a283656c86fc901753b0d835e33c29b11890ff95120f98badf1d10`。full test receipt 在 `runs/sleap_gerbils_baseline/run_manifest.json`，test prediction 为 42 labeled frames/197 instances，validation threshold 0.10 为 43 frames/198 instances。
- 旧 smoke test 官方 eval：模型在 peak threshold 0.2 与 0.15（max 4）均没有 predicted instances，官方 CLI 因此跳过 metric NPZ；不能把 smoke 当 formal 指标。formal test 首次因 `Popen(capture_output=True)` 失败，修复后又遇到 object-NPZ 读取失败；两项已由显式 pipes 和 JSON sidecar 优先读取修复，最终 full test 已成功。
- formal official metrics（random frame split only）：mOKS `0.4020083039400739`；OKS VOC mAP/mAR `0.1086789079250195/0.1392156862745098`；PCK VOC mAP/mAR `0.019913079220009913/0.023529411764705882`；平均距离 `8.790849863678831 px`（p50 `4.6945187969428765`、p95 `21.483785319651368`）；mPCK `0.25666023166023166`（PCK@5 `0.27075289575289574`、PCK@10 `0.4107142857142857`）；visibility precision/recall `0.9832089552238806/0.6078431372549019`。
- B0 oracle-crop diagnostic receipt 在 `runs/b0_gerbils/run_manifest.json`：status `SUCCEEDED`，S0 固定一个 session，1428 个有 provider 标签 query instances 中 correct 525、unknown 201、accuracy `0.36764705882352944`、F1 `0.5993150684931506`。该表不声称人工核验、A-auto enrollment、跨日 biological-ID 或 ID-aware pose。
- Tracking：真实视频连续帧 0–15 的 `--tracking` 命令成功，输出 16 帧 SLP；完整 2,560 帧未跑完，且没有可靠人工 tracking GT，故不计算 HOTA/ID switch。

## 解释边界

当前可称为：真实 SLEAP 数据适配、公开 CLI schema 审计、2-epoch CPU pose smoke、50-epoch formal GPU 训练、full test 官方 random-split 指标、严格门控的 B0 oracle-crop diagnostic 和 MAT 核心实现已落地。当前不能称为：跨日 persistent-ID accuracy、H/A 人工核验时长、B1/B2 改善、strict longitudinal generalization、连续姿态 GT 或论文级纵向结果。B0 的 query crop 使用 provider evaluator 框仅作为 `H_oracle_reference` 诊断，预测 pose 版本和人工 H 仍需独立真实输入。
