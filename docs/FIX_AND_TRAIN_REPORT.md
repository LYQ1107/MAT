# MAT fix-and-train report

This report is updated from receipts after each real stage.  It deliberately
does not convert a smoke run into a formal result.

1. **Old smoke bug and fix.** The earlier two-epoch run used
   `train_steps_per_epoch=1`, producing exactly two optimizer steps
   (`global_step=2`).  `SleapNNBackend.train` now accepts an explicit,
   keyword-only `train_steps_per_epoch`; no value is inferred from
   `max_epochs`.
2. **Checkpoint isolation.** Smoke lives only under
   `runs/sleap_gerbils_pose_smoke`; formal training/evaluation resolves only
   `runs/sleap_gerbils_pose_full/**/best.ckpt`.  A resolver test proves that a
   smoke checkpoint and a non-`best.ckpt` cannot satisfy full evaluation.
3. **Formal command.** The generated plan is
   `MAT_workspace/runs/sleap_gerbils_pose_full/planned_command.json`: 50
   epochs, no step override, no smoke reference.  Actual command, PID and
   child `CUDA_VISIBLE_DEVICES` are recorded in the full run receipt.
4. **Global step/epochs.** The previous smoke values remain 2/2.  Formal
   `actual_epochs`, `optimizer_steps`, `global_step`, loss rows and checkpoint
   hash are filled only after the running full command exits.
5. **Prediction counts/metrics.** The old smoke checkpoint produced zero test
   instances and `SUCCEEDED_NO_PREDICTIONS`; this was not called a SLEAP
   failure.  Full validation sweeps thresholds 0.10/0.15/0.20/0.25 before one
   test/evaluation pass.  Full validation/test/clip paths include the full
   checkpoint SHA prefix and live below `sleap_gerbils_pose_full`; an old
   smoke artifact in the shared run directory cannot be reused.  Zero formal
   predictions stop B0.
6. **MegaDescriptor asset.** Official T-224 config and weights are pinned to
   Hugging Face commit `3ea58ff6c6195bc748bb86c111ff40c32bdddcba`, downloaded
   in an authorized-proxy child and recorded with local SHA-256/bytes/license
   and redirect origins in `locks/assets.lock.json` and workspace receipts.
7. **Preprocess equivalence.** `IdentityPreprocessSpec` uses RGB, bicubic
   224x224, float32 BCHW, ImageNet mean/std.  The local implementation and a
   torchvision reference transform matched exactly on the deterministic audit
   sample; the result is saved beside the local asset.
8. **ROI/part evidence.** The production part encoder calls the same frozen
   MegaDescriptor runtime for global and pose ROIs, has no positive-scaling
   `part_gates` or projection, and masks invalid parts.  A real-asset ROI
   integration check is skipped only when the local runtime is unavailable.
9. **Enrollment pooling.** H and A now share quality-weighted pooling across
   all references; no `refs[0]` shortcut remains.  Anchor descriptors stay
   immutable, and promoted pending exemplars are removed from the active
   pending tier while old snapshots remain auditable.
10. **Oracle B0.** The file-backed runner selects deterministic earliest S0
    sessions covering four provider labels and at most 16 visible/time-spread
    `H_oracle_reference` anchors per identity.  Query truth is evaluator-only.
11. **Predicted-pose B0.** `B0_predicted_pose.yaml` and the runner accept only
    full SLEAP predictions for query boxes/keypoints; identity is assigned by
    the model and evaluator matching never feeds query identity truth back.
12. **B1/B2/O1.** These are not reported as runs until B0 has a measured full
    pose input and identity result.  Missing implementations raise an explicit
    `NotImplementedError`; no fake missing-asset receipt is emitted.
13. **Full test and B0 completion.** The independent formal full test now
    succeeds with 42 labeled frames/197 instances (validation threshold 0.10:
    43 frames/198 instances) and official metrics are recorded in
    `runs/sleap_gerbils_baseline/run_manifest.json`.  The first B0 retry exposed
    a real implementation bug: oracle ROI crops have heterogeneous pixel
    shapes, so stacking them before preprocessing raised `ValueError: all input
    arrays must have the same shape`.  `GlobalIdentityBackend.encode_crops`
    now applies the audited 224/BCHW transform to each crop and concatenates
    only fixed tensors; the generic runner has a per-row fallback and a
    regression test.  The rerun succeeded as a static
    `H_oracle_reference/oracle_crop_diagnostic` (S0 one session, anchors
    16/16/9/16; 1428 labeled query instances, accuracy 0.367647, F1 0.599315,
    unknown 201).  The provider-label contact sheet remains
    `runs/data_sanity/gerbils_gt_contact_sheet.png` (eight seed-17 train
    frames).  Labels are only `true_for_provider_labeled_sessions` because of
    shaved appearance; dense continuous pose GT, human H review and
    independent cross-day biological-ID mapping remain unavailable.  No
    prediction is substituted for GT, and B1/B2 are not reported as runs.
