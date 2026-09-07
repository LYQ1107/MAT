# 复现实验运行手册（当前阶段）

1. 设置 `MAT_WORK_ROOT` 到工作区并运行 `mat doctor`；检查 `route_status`。只有外部核验为 `VERIFIED_DIRECT` 或有授权本地文件才 fetch/import 大资产。
2. 用 `assets plan` 生成 P1 队列；解析 provider metadata，补入真实直链、字节数、license 和 checksum，再由 `assets fetch --direct-only` 执行。每个 receipt 必须 `VERIFIED`。
3. `data inspect/prepare` 生成中性 observations、独立 source/reference/private truth 和 inventory；运行 neutral/leakage 检查后 `split freeze`。
4. 导入并校验离线 MegaDescriptor/DLC 资产；缺任何必需 checkpoint 时停止相应路径，不联网回退。
5. S0 先建 tracklet，按 A 或 S0-only H 生成 gallery v0；每个 query session 使用开始快照，结束时再 commit。
6. 评估器先在 S0 fit/freeze mapping，后续禁止重配；报告完整 truth 分母、unknown、漏检、错误持续和 blockers。

