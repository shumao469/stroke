# NEFEL 生物标志物特征提取（Section 1–7）

本仓库将原始 `Biofeature_extraction_NEFEL.ipynb` notebook 整理为可复用的 Python 包，
并严格按你给定的 **7 个 Section** 拆分为独立模块，便于审稿/复现/批处理。

---

## 安装

### 开发模式（推荐）

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 目录结构（核心）

- `nefel/core.py`：通用工具（tissue mask、通道分离等）
- `nefel/sections.py`：Section 入口（1–7 一一对应）
- `nefel/markers/*`：每个 Section 的实现模块

---

## Section 入口（建议用法）

```python
import cv2
from nefel.sections import (
    section1_inos_arg,
    section2_iba1_day1,
    section3_claudin5,
    section4_cd31,
    section5_synapse,
    section6_gap43,
)

rgb = cv2.cvtColor(cv2.imread("sample.jpg"), cv2.COLOR_BGR2RGB)

m1 = section1_inos_arg(rgb)
m2 = section2_iba1_day1(rgb, prefix="day1_m1", out_dir="qc_masks")
m3 = section3_claudin5(rgb, prefix="day1_m1", out_dir="qc_masks")
m4 = section4_cd31(rgb, prefix="day1_m1", out_dir="qc_masks")
m5 = section5_synapse(rgb, pre_channel=1, post_channel=0)  # 举例：G=Syn, R=PSD95
m6 = section6_gap43(rgb, prefix="day3_m1", out_dir="qc_masks")
```

---

## QC 可视化输出（重要）

为避免“导入即写文件 / 路径不一致导致报错”，现在所有保存 overlay / mask 的行为都改为：

- **只有当你传入 `out_dir="..."` 时才会写文件**  
- 否则默认只返回 metrics dict，便于批处理跑完再统一整理

适用 Section：
- Iba1（Section 2）
- Claudin-5（Section 3）
- CD31（Section 4）
- GAP43（Section 6）
- TUNEL（Section 7）

---

## 每个 Section 的输出字段说明

请见英文更完整文档：`docs/sections.md`（字段、参数、注意事项更详细）。
