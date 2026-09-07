# MAT

MAT（Multi-animal identity and 2-D pose over recordings）将一次参考录制建档与后续按日期/条件的独立录像连接起来。工程底座是 DeepLabCut/SuperAnimal 姿态、官方 ByteTrack 局部轨迹和 WildlifeTools/MegaDescriptor 冻结身份特征；MAT 自己负责中性数据契约、部位证据、冲突约束、持久 registry 和可核验评价。

当前仓库处于从零初始化阶段。大资产目录由 `MAT_WORK_ROOT` 指向，默认 `/data2/usr_for_deadline/MAT_workspace`，不随仓库提交。由于服务器存在应用层代理且透明路由未核验，M0/M1 只完成小元数据和本地导入路径；大资产状态以 `docs/NETWORK_AUDIT.md`、`docs/BLOCKERS.md` 和工作区 receipt 为准。

## 快速检查

```bash
export MAT_WORK_ROOT=/data2/usr_for_deadline/MAT_workspace
PYTHONPATH=src python -m mat.cli doctor --work-root "$MAT_WORK_ROOT"
PYTHONPATH=src python -m mat.cli assets plan --catalog configs/assets/catalog.yaml --phase P1
PYTHONPATH=src python -m pytest -q
```

`mat assets fetch --direct-only` 在路由未经批准时会拒绝大文件；通过 `mat assets import --asset ... --source ...` 可导入用户已经授权且本地完成校验的原件。没有权重时姿态/身份后端不会随机初始化或联网回退。

阶段计划见 [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md)，实际状态和下一条命令见 [docs/PROGRESS.md](docs/PROGRESS.md)。

