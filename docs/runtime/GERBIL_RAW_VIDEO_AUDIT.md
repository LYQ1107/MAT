# Gerbil 原始视频/嵌入帧审计

审计时间：2026-09-09；对象均为本地已校验文件，未联网。

`train.pkg.slp`、`val.pkg.slp`、`test.pkg.slp` 各自暴露 23 个
`labels.videos` 槽位，但 public `Video` 的 filename 仍是对应的
`.pkg.slp`，backend 为 `NoneType`；这些文件提供的是稀疏标注帧所需的
嵌入图像，不是 23 段可连续解码的原始录像。三份文件的标注计数分别为
340/1249、43/159、42/153（frames/instances）。因此不能据此构造完整
跨日 tracklet 或声称连续 20-day GT。

唯一可连续访问的本地示例是
`example_5min.mp4`（1280×1024、25 FPS、2560 帧，约 102.4 s），与
`example_tracking.slp` 的单视频引用相符。它没有独立的跨日生物 ID 真值，
当前只做格式/短帧 tracking smoke，不进入 HOTA、持久 ID 或 pose GT 评价。

## 当前合法评价范围

- B0/B1 使用 `frame-instance cross-recording evaluation`；reference、development、sealed 的角色来自冻结 protocol JSON。
- SLEAP provider track 名称只在私有 truth 中用于数据审计和 H-oracle-reference 建档；尚无独立跨日 biological-ID mapping。
- 没有将预测关键点写入 GT，也没有用 GT identity 预先生成 query tracklet。
- 若未来补齐 23 段连续原视频和独立人工标注，必须新建数据/协议版本，不能覆盖本审计。
