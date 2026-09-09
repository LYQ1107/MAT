# Results protocol correction (2026-09-09)

本文件更正早期 B0 诊断的协议和指标解释；旧 receipt 不删除。

## 更正事项

早期 `runs/b0_gerbils` 使用了所有非 S0 session、固定阈值 0.35，并输出了
非标准的 F1 `0.599315`。该值标记为
`INVALID_NONSTANDARD_F1 / LEGACY_ALL_NON_S0_DIAGNOSTIC`，不再作为 MAT
标准结果。新评估器以每个 sealed GT instance 为分母：unknown 也计入；
macro-F1 是四个固定 identity label 的 one-vs-rest F1 平均，micro-F1 按
全体 TP/FP/FN 计算，另报告 accepted accuracy、unknown rate 和
wrong-identity rate。

## 冻结协议

protocol `92dd746577047d74` 的 reference/development/sealed 角色和 manifest
hash 固定在 `MAT_workspace/assets/manifests/splits/gerbils_longitudinal_v1.json`。
本机 provider datetime 全部缺失，故 `deterministic_session_uid_fallback`；
不能把它表述为严格日期顺序。reference 为
`0b1d08252e086969ba10`，四类 anchor 数为 16/16/9/16；development 为
`c379e2e843756a24a7f4`、`cb8b8c122b53cce32df2`；sealed 为
`dc9fb278b67720cfe656`、`fc57e931dea60736a54e`。

## 真实 sealed-only 结果

| run | pose input | global weight | threshold source/value | truth | accepted/correct/wrong/unknown | accuracy | macro-F1 | micro-F1 |
|---|---|---:|---|---:|---|---:|---:|---:|
| B0 oracle strict | oracle GT pose bbox (diagnostic upper bound) | global only | development / 0.21345681 | 252 | 238/108/130/14 | 0.428571 | 0.401719 | 0.440816 |
| B0 predicted-pose strict | SLEAP predicted pose + geometry evaluator | global only | development / 0.13382465 | 252 | 25/9/16/227 | 0.035714 | 0.055815 | 0.064982 |
| B1 oracle part strict | oracle GT pose parts (diagnostic) | 0.0 (selected on development) | development / 0.35011125 | 252 | 237/117/120/15 | 0.464286 | 0.421854 | 0.478528 |
| B1 predicted part strict | SLEAP predicted pose parts | 0.5 (selected on development) | development / 0.09584712 | 252 | 26/12/14/226 | 0.047619 | 0.071770 | 0.086331 |

每行的 sealed truth 都由同一个 frozen reference mapping 评价；没有按
session 重新 Hungarian，也没有用 sealed 结果调 threshold 或 part weight。
predicted-pose 行先完成预测/身份分配，最后才用 geometry-only matcher 将
prediction UID 与 GT observation UID 对齐。

## 解释边界和未运行项

- B1 oracle 是“pose 完全正确时部位特征”的诊断，不是端到端自动建档；B1 predicted 才包含预测姿态误差。
- `H-human`、独立跨日 biological-ID mapping、连续原始 23 段录像和 dense pose GT 仍为阻塞；provider labels 不能冒充这些真值。
- 197 predicted vs 153 official test GT instance 的 over-detection 诊断见 `docs/runtime/GERBIL_POSE_DETECTION_DIAGNOSTIC.md`。
- B2 learned matcher、O1 memory 和正式论文级纵向运动记录未运行；不存在训练 step、人工时长或测试指标时保持 `null/NOT_RUN`。
