# MAT 项目规范

MAT 的持久身份是 cohort 范围 UUID；它只在一次 S0 建档时与真实个体映射，后续 session 不重新 Hungarian。局部 tracker UID 每段录像重置，gallery 版本跨录像保留。二维关键点始终保存原图像素坐标，未标定相机不输出毫米速度或跨视角生物学比较。

系统边界、数据/真值隔离、协议、评价分母和状态码以 `AGENTS.md` 与 `MASTER_PLAN.md` 为准。本文仅描述代码对象关系，不宣称任何实验结论。

