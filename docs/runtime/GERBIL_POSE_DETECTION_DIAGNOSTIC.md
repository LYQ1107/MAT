# Gerbil pose instance count diagnostic

输入为官方 SLEAP `test.pkg.slp` user instances 与同一 full-checkpoint
prediction SLP，均为 42 labeled frames；未调整 test threshold，也未重新训练。

| quantity | observed |
|---|---:|
| total GT instances | 153 |
| total predicted instances | 197 |
| mean GT count/frame | 3.642857 |
| mean predicted count/frame | 4.690476 |
| equal-count frames | 14 |
| over-detected frames | 27 |
| under-detected frames | 1 |

逐帧计数保存在工作区
`runs/sleap_gerbils_pose_full/test/pose_detection_diagnostic.json`。
解析器用 public `sleap_io.load_slp` 读取 197 个 predicted instances，
没有发现把一个实例重复计数为多个 track 的证据；因此当前将 197 vs 153
标记为 pose baseline 的 over-detection limitation，而不是修改 B1 的输入或
把预测当作 GT。完整细节仍受 official random-frame split 限定。
