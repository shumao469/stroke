import re
from pathlib import Path
import pandas as pd
import pdfplumber

PDF = Path("/mnt/h/Data/Yuchun-yanshi/眼屎收集病历/_processed/clinical_merged_ocr.pdf") 
OUT = PDF.parent / "clinical_extract_out.xlsx"

# ---------- helpers ----------
def find_first(patterns, text):
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return m.group(0), m.group(1) if m.lastindex else m.group(0)
    return "", ""

def norm_space(s):
    return re.sub(r"[ \t]+", " ", s.replace("\u3000"," ")).strip()

# 时间类（给原文片段 + 尝试抽取日期时间）
PAT_ONSET = [
    r"(发病时间|起病时间)[:：]?\s*([0-9]{4}[./-][0-9]{1,2}[./-][0-9]{1,2}\s*[0-9]{0,2}:?[0-9]{0,2})",
    r"(发病时间|起病时间)[:：]?\s*([0-9]{1,2}月[0-9]{1,2}日\s*[0-9]{0,2}:?[0-9]{0,2})",
]
PAT_ARRIVE = [
    r"(到院时间|入院时间|首次到院时间)[:：]?\s*([0-9]{4}[./-][0-9]{1,2}[./-][0-9]{1,2}\s*[0-9]{0,2}:?[0-9]{0,2})",
    r"(到院时间|入院时间|首次到院时间)[:：]?\s*([0-9]{1,2}月[0-9]{1,2}日\s*[0-9]{0,2}:?[0-9]{0,2})",
]
PAT_SAMPLE = [
    r"(采样时间|取样时间|标本采集时间)[:：]?\s*([0-9]{4}[./-][0-9]{1,2}[./-][0-9]{1,2}\s*[0-9]{0,2}:?[0-9]{0,2})",
    r"(采样时间|取样时间|标本采集时间)[:：]?\s*([0-9]{1,2}月[0-9]{1,2}日\s*[0-9]{0,2}:?[0-9]{0,2})",
]

PAT_NIHSS = [
    r"NIHSS(?:评分|)\s*[:：]?\s*([0-9]{1,2})",
    r"NIHSS\s*([0-9]{1,2})",
]
PAT_AGE = [
    r"(年龄)[:：]?\s*([0-9]{1,3})\s*岁",
    r"(Age)[:：]?\s*([0-9]{1,3})",
]
PAT_SEX = [
    r"(性别)[:：]?\s*(男|女)",
    r"\b(Sex)[:：]?\s*(M|F)\b",
]

# HTN/DM：只做粗判定（出现关键词=是）
KW_HTN = ["高血压", "HTN"]
KW_DM  = ["糖尿病", "DM", "T2DM", "T1DM"]

# 溶栓/取栓：粗判定
KW_THROMBOLYSIS = ["溶栓", "rt-PA", "阿替普酶", "尿激酶"]
KW_THROMBECTOMY = ["取栓", "机械取栓", "支架取栓", "血管内治疗"]

# 用药（粗抓关键词）
KW_MEDS = [
    "阿司匹林","氯吡格雷","替格瑞洛","他汀","阿托伐他汀","瑞舒伐他汀",
    "华法林","利伐沙班","阿哌沙班","达比加群",
    "依达拉奉","丁苯酞","甘露醇","尼卡地平","硝苯地平","缬沙坦","氨氯地平"
]

def has_any(text, kws):
    t = text
    hit = [k for k in kws if k.lower() in t.lower()]
    return ("是" if len(hit) else "否"), ";".join(hit)

# ---------- main ----------
rows = []
with pdfplumber.open(str(PDF)) as pdf:
    for pi, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        text = norm_space(text)
        if not text:
            continue

        # 把每页当作一个“片段单元”，后续你也可以按“住院号/姓名”再聚合
        sample_id = f"UNK_p{pi:04d}"

        onset_snip, onset_val = find_first(PAT_ONSET, text)
        arrive_snip, arrive_val = find_first(PAT_ARRIVE, text)
        sample_snip, sample_val = find_first(PAT_SAMPLE, text)

        nihss_snip, nihss_val = find_first(PAT_NIHSS, text)
        age_snip, age_val = find_first(PAT_AGE, text)
        sex_snip, sex_val = find_first(PAT_SEX, text)

        htn, htn_hit = has_any(text, KW_HTN)
        dm, dm_hit   = has_any(text, KW_DM)
        throm, throm_hit = has_any(text, KW_THROMBOLYSIS)
        thromb, thromb_hit = has_any(text, KW_THROMBECTOMY)

        meds_hit = [k for k in KW_MEDS if k in text]
        meds_str = ";".join(sorted(set(meds_hit)))

        # 只要这一页出现 NIHSS/发病/入院/采样 任意一个，就记录
        if any([onset_val, arrive_val, sample_val, nihss_val]):
            rows.append({
                "SampleID": sample_id,
                "Group(HS/ZS/NC)": "",
                "采样时间_value": sample_val,
                "采样时间_snip": sample_snip,
                "发病时间_value": onset_val,
                "发病时间_snip": onset_snip,
                "首次到院时间_value": arrive_val,
                "首次到院时间_snip": arrive_snip,
                "NIHSS": nihss_val,
                "NIHSS_snip": nihss_snip,
                "年龄": age_val,
                "年龄_snip": age_snip,
                "性别": sex_val,
                "性别_snip": sex_snip,
                "HTN": htn,
                "HTN_hit": htn_hit,
                "DM": dm,
                "DM_hit": dm_hit,
                "溶栓(是/否)": throm,
                "溶栓_hit": throm_hit,
                "取栓(是/否)": thromb,
                "取栓_hit": thromb_hit,
                "用药粗抓": meds_str,
                "page": pi
            })

df = pd.DataFrame(rows).sort_values("page")
df.to_excel(OUT, index=False)
print("✅ Saved:", OUT, " rows=", len(df))
