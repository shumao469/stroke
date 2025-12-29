import re, argparse
from pathlib import Path
import pandas as pd
import numpy as np

import pdfplumber

# fallback OCR
try:
    import pytesseract
    from PIL import Image
    HAS_TESS = True
except Exception:
    HAS_TESS = False

def norm_space(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\u3000", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def pick_sample_id(text: str) -> str:
    # Try to find something like 姓名/住院号/病案号等；也允许你后续用文件名对照修正
    pats = [
        r"(住院号|病案号|就诊号|门诊号)[:：]?\s*([A-Za-z0-9\-]+)",
        r"(姓名)[:：]?\s*([^\s]{2,6})",
    ]
    for pat in pats:
        m = re.search(pat, text)
        if m:
            return f"{m.group(1)}_{m.group(2)}"
    return ""

def find_first(patterns, text):
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            # 返回原文片段 + 值
            if m.lastindex and m.lastindex >= 2:
                return m.group(0), m.group(2)
            if m.lastindex and m.lastindex >= 1:
                return m.group(0), m.group(1)
            return m.group(0), m.group(0)
    return "", ""

def find_bool(keywords, text):
    # 粗抓：出现任一关键词视为“是”
    for kw in keywords:
        if re.search(kw, text, flags=re.I):
            return 1
    return 0

def extract_text_page(pdfplumber_page):
    txt = pdfplumber_page.extract_text() or ""
    return norm_space(txt)

def ocr_page_to_text(pdfplumber_page, lang="chi_sim+eng", dpi=200):
    if not HAS_TESS:
        return ""
    im = pdfplumber_page.to_image(resolution=dpi).original
    if not isinstance(im, Image.Image):
        im = Image.fromarray(np.array(im))
    txt = pytesseract.image_to_string(im, lang=lang)
    return norm_space(txt)

# ---- regex patterns ----
PAT_ONSET = [
    r"(发病时间|起病时间)[:：]?\s*([0-9]{4}[./-][0-9]{1,2}[./-][0-9]{1,2}\s*[0-9]{0,2}:?[0-9]{0,2})",
    r"(发病时间|起病时间)[:：]?\s*([0-9]{1,2}月[0-9]{1,2}日\s*[0-9]{0,2}:?[0-9]{0,2})",
    r"(发病|起病)[:：]?\s*([0-9]{1,2}\s*[:：]\s*[0-9]{1,2})",
]
PAT_ARRIVE = [
    r"(到院时间|入院时间|首次到院时间)[:：]?\s*([0-9]{4}[./-][0-9]{1,2}[./-][0-9]{1,2}\s*[0-9]{0,2}:?[0-9]{0,2})",
    r"(到院时间|入院时间|首次到院时间)[:：]?\s*([0-9]{1,2}月[0-9]{1,2}日\s*[0-9]{0,2}:?[0-9]{0,2})",
]
PAT_SAMPLE = [
    r"(采样时间|采血时间|标本采集时间)[:：]?\s*([0-9]{4}[./-][0-9]{1,2}[./-][0-9]{1,2}\s*[0-9]{0,2}:?[0-9]{0,2})",
    r"(采样时间|采血时间|标本采集时间)[:：]?\s*([0-9]{1,2}月[0-9]{1,2}日\s*[0-9]{0,2}:?[0-9]{0,2})",
]
PAT_NIHSS = [
    r"(NIHSS)[:：]?\s*([0-9]{1,2})",
    r"(NIHSS评分)[:：]?\s*([0-9]{1,2})",
    r"(美国国立卫生研究院卒中量表)[:：]?\s*([0-9]{1,2})",
]
PAT_AGE = [
    r"(年龄)[:：]?\s*([0-9]{1,3})\s*岁",
    r"([0-9]{1,3})\s*岁",
]
PAT_SEX = [
    r"(性别)[:：]?\s*(男|女)",
    r"\b(男|女)\b",
]

KW_HTN = [r"高血压", r"HTN"]
KW_DM  = [r"糖尿病", r"\bDM\b", r"2型糖尿病", r"1型糖尿病"]
KW_TPA = [r"溶栓", r"rt-PA", r"阿替普酶", r"替奈普酶", r"TNK"]
KW_MT  = [r"取栓", r"血管内治疗", r"机械取栓", r"支架取栓", r"抽吸取栓"]

# 用药粗抓（不追求完整，只做“出现过的片段”）
PAT_MEDS = [
    r"(用药)[:：]?\s*([^\n]{0,80})",
    r"(口服药|静滴|静脉)[:：]?\s*([^\n]{0,80})",
    r"(阿司匹林|氯吡格雷|他汀|阿托伐他汀|瑞舒伐他汀|依达拉奉|丁苯酞|奥拉西坦|甘露醇|降压|胰岛素)[^\n]{0,80}",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ocr_fallback", action="store_true", help="If page text empty, run tesseract OCR per page")
    ap.add_argument("--ocr_lang", default="chi_sim+eng")
    ap.add_argument("--max_pages", type=int, default=0, help="0 means all pages")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")

    out_xlsx = Path(args.out) if args.out else pdf.parent / "clinical_extract_out.xlsx"
    out_csv  = out_xlsx.with_suffix(".csv")

    rows = []
    with pdfplumber.open(str(pdf)) as P:
        n_pages = len(P.pages)
        use_pages = n_pages if args.max_pages <= 0 else min(args.max_pages, n_pages)

        for pi in range(use_pages):
            page = P.pages[pi]
            txt = extract_text_page(page)

            if args.ocr_fallback and len(txt) < 30:
                txt2 = ocr_page_to_text(page, lang=args.ocr_lang)
                if len(txt2) > len(txt):
                    txt = txt2

            if len(txt) < 10:
                # 仍然没文本：跳过，但记录空页方便定位
                rows.append({
                    "page": pi + 1,
                    "SampleID": "",
                    "Group": "",
                    "OnsetTime_raw": "",
                    "ArriveTime_raw": "",
                    "SampleTime_raw": "",
                    "NIHSS": "",
                    "Age": "",
                    "Sex": "",
                    "HTN": "",
                    "DM": "",
                    "Thrombolysis": "",
                    "Thrombectomy": "",
                    "Meds_raw": "",
                    "Text_snip": ""
                })
                continue

            # fields
            onset_snip, onset_val   = find_first(PAT_ONSET, txt)
            arrive_snip, arrive_val = find_first(PAT_ARRIVE, txt)
            sample_snip, sample_val = find_first(PAT_SAMPLE, txt)

            nihss_snip, nihss_val = find_first(PAT_NIHSS, txt)
            age_snip, age_val     = find_first(PAT_AGE, txt)
            sex_snip, sex_val     = find_first(PAT_SEX, txt)

            # group (rough) - later you can merge by sample id mapping table
            grp = ""
            if re.search(r"出血|脑出血|ICH|HS", txt, flags=re.I):
                grp = "HS"
            if re.search(r"缺血|脑梗|梗死|IS|ZS", txt, flags=re.I):
                grp = "ZS"

            sid = pick_sample_id(txt)

            meds_snip, meds_val = find_first(PAT_MEDS, txt)
            meds_val = meds_val if meds_val else meds_snip

            row = {
                "page": pi + 1,
                "SampleID": sid,
                "Group": grp,
                "OnsetTime_raw": onset_snip or onset_val,
                "ArriveTime_raw": arrive_snip or arrive_val,
                "SampleTime_raw": sample_snip or sample_val,
                "NIHSS": nihss_val,
                "Age": age_val,
                "Sex": sex_val,
                "HTN": find_bool(KW_HTN, txt),
                "DM": find_bool(KW_DM, txt),
                "Thrombolysis": find_bool(KW_TPA, txt),
                "Thrombectomy": find_bool(KW_MT, txt),
                "Meds_raw": meds_val[:120] if meds_val else "",
                "Text_snip": txt[:400].replace("\n", " ")
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # ---- 防止 KeyError：即便 rows 为空也能输出空表 ----
    if "page" not in df.columns:
        df["page"] = []

    df = df.sort_values("page")

    # write
    df.to_excel(out_xlsx, index=False)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("✅ Saved:")
    print("  ", out_xlsx)
    print("  ", out_csv)
    print("Rows:", len(df))
    print("Tip: if most rows are empty, rerun with --ocr_fallback")

if __name__ == "__main__":
    main()
