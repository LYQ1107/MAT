# SLEAP gerbils：V0–V2 结果表

本表只记录真实执行事实。官方 random frame split 结果不能解释为 strict longitudinal generalization；`example_tracking.slp` 不是人工 GT。`NOT_RUN`/`null` 表示没有满足数据、权重或标注条件，不是零分。

| Method | Pose source | ID feature | Matcher | Memory | Enrollment | Persistent ID | Unknown | ID-aware Pose |
|---|---|---|---|---|---|---|---|---|
| SLEAP_OFFICIAL_RANDOM_SPLIT_BASELINE (smoke) | SLEAP-NN 0.3.3 bottomup，2 epochs CPU，`global_step=2` | NOT_RUN（MegaDescriptor-T-224 缺官方 checkpoint） | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| B0_global_static_gallery | SLEAP test predict（42 labels、0 predicted instances） | MegaDescriptor-T-224 | global cosine（代码已实现） | static S0 snapshot（代码已实现） | H_oracle_reference/A_auto code path；真实 S0 未建档 | NOT_RUN — `BLOCKED_MISSING_IDENTITY_ASSET` | null | NOT_RUN — `BLOCKED_NEEDS_POSE_GT` |
| B1_part_static | SLEAP pose + pose-derived ROI（代码/契约 forward） | global + ROIAlign head/trunk/tail parts | fixed quality-weighted fusion（代码已实现） | static | NOT_RUN | NOT_RUN | null | NOT_RUN |
| B1 EvidenceMatcher | NOT_RUN | global/part pair MLP（实现但未训练） | trainable evidence head | static | NOT_RUN | NOT_RUN | null | NOT_RUN |
| B2 safe longitudinal memory | NOT_RUN | NOT_RUN | NOT_RUN | anchor/confirmed/pending + end-of-session（实现但未运行） | NOT_RUN | NOT_RUN | null | NOT_RUN |

## 可复核运行事实

- 数据：`MAT_workspace/datasets/sleap_gerbils/` 五个官方对象合计 964,739,339 bytes；SHA 和 split 计数见 [`docs/PROGRESS.md`](PROGRESS.md)。
- 训练 checkpoint：`MAT_workspace/runs/sleap_gerbils_pose_smoke/models/260908_181311.bottomup.n=383/best.ckpt`；训练 log 的第 1 个 epoch train/val loss 分别为 0.019036374986171722/0.018797585740685463。两步 optimizer 确认于 checkpoint `global_step=2`。
- Test 官方 eval：模型在 peak threshold 0.2 与 0.15（max 4）均没有 predicted instances，官方 CLI 因此跳过 metric NPZ；不能报告 mAP/PCK/像素误差。
- Tracking：真实视频连续帧 0–15 的 `--tracking` 命令成功，输出 16 帧 SLP；完整 2,560 帧未跑完，且没有可靠人工 tracking GT，故不计算 HOTA/ID switch。

## 解释边界

当前可称为：真实 SLEAP 数据适配、公开 CLI schema 审计、2-epoch CPU pose smoke、test/clip 格式链和 MAT 核心实现已落地。当前不能称为：完整 pose baseline 指标、跨日 persistent-ID accuracy、H/A 人工核验时长、B1/B2 改善、strict longitudinal generalization 或论文级结果。下一步必须先取得授权的 MegaDescriptor-T-224 官方 `config.json` + checkpoint，完成本地架构/SHA/state-key 校验，再按 S0-only mapping 运行 B0；没有该原件继续保持阻塞。
