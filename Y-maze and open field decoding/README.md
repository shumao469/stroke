# ISTBI Behavior Decoder

用于 **小鼠旷场（Open Field Test, OFT）** 与 **Y 迷宫（Y-maze）** 视频解码的 Python 项目。该仓库由上传 notebook 代码整理而成，已经重构为更适合 GitHub 上传、协作维护和后续扩展的目录结构。

## 版权说明

**Copyright (c) ISTBI, Fudan University, Xu Lab. All rights reserved.**  
**Contact:** shumaoxu@fudan.edu.cn

本仓库中的代码、文档、图示和衍生输出默认归属于 **ISTBI, Fudan University, Xu Lab**。如需转载、复用、再分发或商业合作，请联系上方邮箱。

---

## 1. 项目目标

本项目面向以下两类行为学视频：

1. **旷场实验（OFT）**
   - 目标：追踪小鼠在开放区域中的运动轨迹。
   - 典型输出：轨迹图、热图、速度/区域时间线、总距离、平均速度、中心区停留时间、中心进入次数。

2. **Y 迷宫实验（Y-maze）**
   - 目标：追踪小鼠在 3 个臂和中心区中的探索过程。
   - 典型输出：轨迹图、热图、速度/区域时间线、各臂停留时间、进入次数、自发交替率（SAP）、进臂顺序。

---

## 2. 当前代码的核心思路

本仓库保留了你 notebook 中最关键、最稳妥的一条技术路线：

- **中值背景建模**：从视频中均匀抽帧，生成无鼠的背景图。
- **手动蓝色 ROI 标注**：不强依赖亮度，而是让用户直接用蓝色标出有效区域。
- **背景差分 + 形态学清理**：从前景中提取小鼠轮廓。
- **质心追踪**：以轮廓质心作为小鼠当前位置。
- **轨迹重建**：绘制红色轨迹线，并标记起点 `S`、终点 `E`。
- **行为量化**：输出距离、速度、分区停留时间、分区进入次数等指标。

这条路线特别适合你前面提到的情形：

- 视频底部是一个明确边界的方框或迷宫结构。
- 不能单纯依赖光照明暗来识别底板。
- 更强调 **先精确限定实验区域，再在区域内追踪小鼠**。

---

## 3. 仓库结构

```text
ISTBI_Behavior_Decoder/
├─ README.md
├─ COPYRIGHT.txt
├─ LICENSE.txt
├─ requirements.txt
├─ pyproject.toml
├─ .gitignore
├─ docs/
│  └─ OPERATION_GUIDE_CN.md
├─ notebooks/
│  └─ 小鼠旷场和Y迷宫视频解码.ipynb
├─ scripts/
│  ├─ run_open_field.py
│  └─ run_ymaze.py
└─ src/
   └─ istbi_behavior_decoder/
      ├─ __init__.py
      ├─ common.py
      ├─ open_field.py
      └─ ymaze.py
```

---

## 4. 环境配置（WSL 推荐）

### 4.1 创建虚拟环境

```bash
cd /path/to/ISTBI_Behavior_Decoder
python3 -m venv .venv
source .venv/bin/activate
```

### 4.2 安装依赖

```bash
pip install -r requirements.txt
```

---

## 5. 数据组织建议

建议目录结构如下：

```text
/mnt/h/2024/VPL电生理/processed/Submit/1号鼠/
├─ TI刺激前/
│  ├─ 小鼠A_旷场.mp4
│  ├─ 小鼠A_ROI_debug.jpg
│  ├─ 小鼠A_Y迷宫.mp4
│  └─ 小鼠A_YMaze_ROI_debug.jpg
├─ TI刺激后1天/
│  ├─ 小鼠B_旷场.mp4
│  ├─ 小鼠B_ROI_debug.jpg
│  ├─ 小鼠B_Y迷宫.mp4
│  └─ 小鼠B_YMaze_ROI_debug.jpg
└─ ...
```

### 关键命名要求

#### 旷场
- 视频文件名中应包含：`旷场`
- ROI 图片建议命名为：`视频同名_ROI_debug.jpg`

例如：
- `mouse01_旷场.mp4`
- `mouse01_旷场_ROI_debug.jpg`

#### Y 迷宫
- 视频文件名中应包含：`Y迷宫`
- ROI 图片建议命名为：`视频同名_YMaze_ROI_debug.jpg`

例如：
- `mouse01_Y迷宫.mp4`
- `mouse01_Y迷宫_YMaze_ROI_debug.jpg`

---

## 6. ROI 手工标注规则

### 6.1 为什么推荐手工蓝色标注

对于你的数据，直接自动识别底板或迷宫边界可能受以下因素影响：

- 反光
- 阴影
- 边缘不均匀照明
- 实验者手部短暂进入画面
- 装置底板颜色变化

因此当前方案采用 **蓝色手工 ROI 标注**，优点是：

- 可控性高
- 对复杂光照更稳
- 便于快速人工校正
- 更接近“先保证有效分析区域，再做追踪”的思路

### 6.2 标注方法

在背景图或任意静态帧上：

- 用画图工具把 **有效实验区域** 涂成明显蓝色。
- 旷场：涂整个底部活动区域。
- Y 迷宫：涂整个 Y 形活动区域。
- 只需保证蓝色区域覆盖有效区域，不必追求艺术化边缘。

程序会自动提取 HSV 中的蓝色区域作为 mask。

---

## 7. 旷场实验解码逻辑

### 7.1 输入

- `*.mp4` 视频
- 对应蓝色 ROI 图

### 7.2 处理流程

1. **背景估计**  
   从视频中跳过前几秒，均匀抽取若干帧，取中值生成背景图。

2. **ROI 提取**  
   从 `_ROI_debug.jpg` 中提取蓝色区域，得到旷场底板 mask。

3. **中心区定义**  
   在 ROI 的外接矩形内，取中心矩形作为 `Center`。默认 `center_ratio=0.5`，即中心区宽高各为整体的一半。

4. **小鼠检测**  
   对每帧做：
   - 背景差分
   - 阈值分割
   - 形态学开运算
   - 轮廓面积筛选
   - 质心计算

5. **轨迹重建**  
   按帧连接质心，生成轨迹线。

6. **行为量化**  
   输出：
   - 总路程
   - 平均速度
   - 中心区停留时间
   - 边缘区停留时间
   - 中心区进入次数
   - 中心区占比

### 7.3 输出文件

每个视频所在目录会生成：

- `*_trajectory.jpg`：红色轨迹图，带 `S`/`E`
- `*_heatmap.jpg`：热图
- `*_session_summary.png`：速度 + 区域时间线
- `*_Zone_Debug.jpg`：ROI 与中心区检查图

根目录会生成：

- `OFT_Results.csv`
- `OFT_Cohort_Summary.png`

---

## 8. Y 迷宫解码逻辑

### 8.1 输入

- `*.mp4` 视频
- 对应蓝色 Y 迷宫 ROI 图

### 8.2 处理流程

1. **背景估计**  
   同旷场，采用中值背景。

2. **Y 迷宫 ROI 提取**  
   从 `_YMaze_ROI_debug.jpg` 中提取蓝色区域。

3. **自动分区**  
   在整个 Y 迷宫 mask 上：
   - 使用凸包缺陷（convexity defects）寻找中心交汇区域
   - 若凸包缺陷不足，则退化为距离变换圆形中心估计
   - 将剩余区域按连通域拆成 3 个臂
   - 按相对中心角度排序为 `Arm 1/2/3`

4. **小鼠检测与追踪**  
   同样使用：
   - 背景差分
   - 阈值分割
   - ROI 内轮廓筛选
   - 质心追踪

5. **区域判定**  
   根据质心落点，判断当前位于 `Center` 或某个 `Arm`。

6. **序列与 SAP 计算**  
   记录进臂序列，例如：

   ```text
   Arm 1 > Arm 2 > Arm 3 > Arm 1 > Arm 3
   ```

   以连续三个互不重复臂组成一次有效交替，计算自发交替率：

   ```text
   SAP = 有效交替次数 / (总进臂次数 - 2) × 100%
   ```

### 8.3 输出文件

每个视频目录会生成：

- `*_trajectory.jpg`：红色轨迹图，带 `S`/`E`
- `*_heatmap.jpg`：热图
- `*_session_summary.png`：速度 + 区域时间线
- `*_Zone_Debug.jpg`：Center 与 3 个 arm 的自动分区检查图

根目录会生成：

- `YMaze_Results.csv`
- `YMaze_Cohort_Summary.png`

---

## 9. 命令行运行方式

### 9.1 运行旷场分析

```bash
python scripts/run_open_field.py \
  --base-dir "/mnt/h/2024/VPL电生理/processed/Submit/1号鼠" \
  --skip-seconds 5 \
  --analyze-seconds 300
```

如果已经知道像素与厘米换算关系，可加入：

```bash
python scripts/run_open_field.py \
  --base-dir "/mnt/h/2024/VPL电生理/processed/Submit/1号鼠" \
  --skip-seconds 5 \
  --analyze-seconds 300 \
  --pixel-to-cm 0.045
```

### 9.2 运行 Y 迷宫分析

```bash
python scripts/run_ymaze.py \
  --base-dir "/mnt/h/2024/VPL电生理/processed/Submit/1号鼠" \
  --skip-seconds 5 \
  --analyze-seconds 300
```

同样可加入：

```bash
python scripts/run_ymaze.py \
  --base-dir "/mnt/h/2024/VPL电生理/processed/Submit/1号鼠" \
  --skip-seconds 5 \
  --analyze-seconds 300 \
  --pixel-to-cm 0.045
```

---

## 10. 结果解释建议

### 旷场

常用解释方向：

- **Total Distance**：总体活动水平
- **Mean Speed**：移动速度水平
- **Center Time / Center Ratio**：焦虑样行为的粗略指标之一
- **Center Entries**：对中心区探索意愿

### Y 迷宫

常用解释方向：

- **SAP_percent**：空间工作记忆或自发交替表现
- **Arm entries**：探索驱动力
- **Center vs Arm time**：中心徘徊与臂探索平衡
- **Arm sequence**：原始进臂顺序，便于人工复核

---

## 11. 当前版本的边界与注意事项

1. **轨迹单位默认是像素**  
   若没有提供 `--pixel-to-cm`，距离与速度单位默认是 px、px/s。

2. **依赖蓝色手工 ROI**  
   这是当前版本最重要的稳健性来源。

3. **遮挡严重时会用上一帧位置续接**  
   这是为了避免短时检测失败导致轨迹断裂。

4. **Y 迷宫自动分区需人工检查**  
   建议始终查看 `*_Zone_Debug.jpg`，确认 Center/Arm 1/2/3 分区合理。

5. **中心区定义是工程化定义**  
   旷场中心区当前按 ROI 外接矩形的中心比例定义，如需与你实验室既定规范完全一致，可继续修改 `center_ratio` 或改成固定厘米尺寸。

---

## 12. 后续可扩展方向

后面还可以继续加：

- 批量视频 QC 报告
- 更稳健的遮挡恢复
- 鼠体朝向/头尾识别
- 更正式的统计分析流程
- 自动生成论文级图版
- 将 ROI 标注改为交互式 GUI
- 输出行为事件表（freezing / mobility bouts）

---

## 13. 引用与署名建议

如果该仓库用于组内共享、论文补充材料、方法学说明或项目归档，建议保留如下署名：

> ISTBI, Fudan University, Xu Lab.  
> Contact: shumaoxu@fudan.edu.cn

