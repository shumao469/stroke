# 操作说明（中文版）

## 一、适用对象

本说明适用于：

- WSL / Ubuntu 环境下运行小鼠行为学视频解码；
- 数据中包含 **旷场** 与 **Y迷宫** 两类视频；
- 采用 **人工蓝色 ROI 标注 + 背景差分 + 质心追踪** 的流程。

---

## 二、最推荐的使用顺序

### A. 旷场

1. 先从视频抽一张清晰背景图或静态帧；
2. 用蓝色把底部有效运动区域涂出来；
3. 保存为 `视频同名_ROI_debug.jpg`；
4. 运行 `run_open_field.py`；
5. 检查：
   - `*_Zone_Debug.jpg`
   - `*_trajectory.jpg`
   - `*_heatmap.jpg`
   - `OFT_Results.csv`

### B. Y 迷宫

1. 在背景图或静态帧上把整个 Y 迷宫区域涂成蓝色；
2. 保存为 `视频同名_YMaze_ROI_debug.jpg`；
3. 运行 `run_ymaze.py`；
4. 检查：
   - `*_Zone_Debug.jpg`
   - `*_trajectory.jpg`
   - `*_heatmap.jpg`
   - `YMaze_Results.csv`

---

## 三、WSL 中的典型命令

```bash
cd /mnt/data/ISTBI_Behavior_Decoder
source .venv/bin/activate
```

### 旷场

```bash
python scripts/run_open_field.py \
  --base-dir "/mnt/h/2024/VPL电生理/processed/Submit/1号鼠"
```

### Y 迷宫

```bash
python scripts/run_ymaze.py \
  --base-dir "/mnt/h/2024/VPL电生理/processed/Submit/1号鼠"
```

---

## 四、如何判断结果是否可信

### 1. 轨迹图

应满足：

- 红线基本连续；
- 轨迹主要位于 ROI 内；
- 起点 `S` 和终点 `E` 合理；
- 不应大面积跑到墙外或画面边角。

### 2. 热图

应满足：

- 热区与轨迹高频停留位置一致；
- 不应在 ROI 外出现高强度热点。

### 3. Zone Debug 图

#### 旷场
- 中心区应位于实验场中央；
- 中心区不应越过实际边界。

#### Y 迷宫
- Center 应位于三臂交汇处；
- Arm 1/2/3 应覆盖三条臂；
- 若分错，应先检查 ROI 标注是否过窄或断裂。

---

## 五、常见问题

### Q1. 为什么一定要手工蓝色 ROI？

因为真实实验视频中：

- 光照不均匀；
- 底板边界不一定总是高对比；
- 相机角度、反光、阴影都会影响自动边界检测。

所以对于你当前数据，人工 ROI 反而更稳，更适合论文前的批量分析。

### Q2. 为什么有时轨迹会短暂停住？

当某一帧没有成功检测到小鼠时，代码会沿用上一帧位置，以减少轨迹断裂。这是一个工程上的平滑处理，而不是重新估计真实位置。

### Q3. 距离为什么是像素？

因为原始 notebook 主要基于图像像素追踪。若你有比例尺，可以用 `--pixel-to-cm` 转成厘米。

---

## 六、建议的实验记录方式

建议每只动物、每个时间点保留：

- 原始视频
- ROI 标注图
- 输出轨迹图
- 输出热图
- CSV 指标表
- 实验日志（动物编号、时间点、是否剔除）

这样后续写论文或补充材料时会更方便。

---

## 七、版权与署名

**Copyright (c) ISTBI, Fudan University, Xu Lab. All rights reserved.**  
**Contact:** shumaoxu@fudan.edu.cn

