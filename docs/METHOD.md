# MAT 方法边界（实现版）

前端产生检测、二维关键点和 session 内局部 tracklet；ByteTrack wrapper 将真实输入 detection index 带到 association，纯 Kalman 预测没有完整姿态。身份 encoder 只接中性 crop/关键点，global/part descriptor 带 fingerprint。跨 session matcher 对每个 tracklet允许 unknown；只对同相机同时重叠 tracklet 加 cannot-link，非重叠碎片可共享一个 persistent ID，多相机视图不被错误互斥。

registry 的 anchor 永不覆盖。session 开始固定快照用于打分，候选先进入 quarantine；开发集确定质量、margin、独立时间窗门槛后才可在 session 结束以 expected version 原子 commit。encoder fingerprint 改变时拒绝直接 cosine，必须重编码历史引用或保留固定 anchor 分支。

当前实现是可审计骨架和 fixture-capable 核心，不宣称收益；真实上游/权重/数据未到位的路径显式阻塞。

