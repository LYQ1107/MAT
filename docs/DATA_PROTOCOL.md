# 数据与信息隔离协议

模型入口只读取 observations JSONL：`observation_uid/frame_uid/session_uid/cohort_uid/camera_uid/frame_index/timestamp_s/image_ref`。`gt_id/EID/treatment/behavior` 仅在 `private_eval_truth.jsonl` 中，由评估器读取。`private_object_index.jsonl` 把中性对象句柄映射到只读原件，不能作为模型匹配键。

S0 参考信息和后续 query 真值权限分离；H-human 只允许 S0 的明确身份确认，H-oracle-reference 只能做上限诊断。按群体/个体/录制冻结 source/dev/sealed test，禁止随机图片切分、未来日期可见、测试 EID 计数先验。每次 run 保存 manifest hash、输入模式、seed、registry 版本和访问清单。

