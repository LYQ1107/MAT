# 上游接口锁定与核验

以下 commit 由 2026-09-08 的小型 `git ls-remote` 读取并写入 `locks/upstream.lock.yaml`；没有 clone 大仓库或下载 LFS。源码文本只存于临时审计目录，不进入 MAT Git。

| 上游 | commit | 实际核验 |
|---|---|---|
| DeepLabCut | `53e95879b9a090ccd095c3f9c724e1ade4be8fb4` | `video_inference_superanimal(videos, superanimal_name, model_name, detector_name=None, ..., customized_pose_checkpoint=None, customized_detector_checkpoint=None, customized_model_config=None, ..., max_individuals=10, device='auto')`；`apis/videos.py::video_inference(video, pose_runner, detector_runner=None, cropping=None, shelf_writer=None, robust_nframes=False, show_gpu_memory=False)`；`VideoIterator.set_context(context)` |
| ByteTrack | `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` | `STrack.__init__(tlwh, score)`、`BYTETracker(args, frame_rate=30)`、`update(output_results, img_info, img_size)`；上游有 `np.float`，需要最小检测索引/dtype patch |
| WildlifeTools | `2f22214c8c331ca4174ce231c23de0add7718ee2` | `DeepFeatures(model, batch_size=128, num_workers=1, device='cpu', cache_path=None)`；`forward_batch` 内 `torch.no_grad()`，仅冻结推断 |
| idtracker.ai | `14ff552d209d743e514e8acdeb1590777b9fef98` | 仅在独立环境、真实 session 目录和官方 `idmatcherai MASTER MATCHING...` CLI 满足后接入 |
| TrackEval | `12c8791b303e0a0b50f753af204249e622d0281a` | 计划使用官方指标实现；MAT 转换器尚未运行 |
| CowIDentifier | `5126e7f5af085779160e8d5cc986cc0cf07a4719` | 作为 MultiCamCows 作者代码候选，尚未下载/运行 |

未核验内容：DLC 本机安装版本、模型 zoo 权重、SuperAnimal 输出实际 h5/JSON 键、ByteTrack 补丁后未修改版等价性、WildlifeTools/timm 模型配置和所有数据的跨日映射。MAT 后端在缺少本地校验资产时明确失败，不猜函数或静默联网。

