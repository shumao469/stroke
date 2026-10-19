import os, subprocess, glob
from pathlib import Path

ROOT = Path("/mnt/h/Data/Yuchun-yanshi/眼屎收集病历") 
OUTDIR = ROOT / "_processed"
OUTDIR.mkdir(parents=True, exist_ok=True)

PDF_DIR = OUTDIR / "pdf"
PDF_DIR.mkdir(exist_ok=True)

MERGED_PDF = OUTDIR / "clinical_merged.pdf"
OCR_PDF    = OUTDIR / "clinical_merged_ocr.pdf"

def run(cmd):
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)

# 1) docx -> pdf (using libreoffice)
docx_files = sorted(glob.glob(str(ROOT / "**" / "*.docx"), recursive=True))
for f in docx_files:
    f = Path(f)
    # 输出到 PDF_DIR，文件名同 stem
    out_pdf = PDF_DIR / (f.stem + ".pdf")
    if out_pdf.exists() and out_pdf.stat().st_size > 0:
        continue
    run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(PDF_DIR), str(f)])

# 2) collect all pdf (original + converted)
pdf_files = sorted(glob.glob(str(ROOT / "**" / "*.pdf"), recursive=True)) + \
            sorted(glob.glob(str(PDF_DIR / "*.pdf")))
pdf_files = [p for p in pdf_files if Path(p).exists() and Path(p).stat().st_size > 0]
pdf_files = sorted(set(pdf_files))

if not pdf_files:
    raise SystemExit("No PDF found.")

# 3) merge into one pdf using ocrmypdf's underlying qpdf via python? simplest: use qpdf or pymupdf
# We'll use pymupdf for robust merge
import fitz  # pymupdf
doc_out = fitz.open()
for p in pdf_files:
    try:
        doc_in = fitz.open(p)
        doc_out.insert_pdf(doc_in)
        doc_in.close()
    except Exception as e:
        print("[SKIP]", p, e)
doc_out.save(str(MERGED_PDF))
doc_out.close()
print("✅ Merged:", MERGED_PDF)

# 4) OCR merged pdf -> searchable pdf
# --force-ocr: 即使已有文本也重做
# --deskew/--rotate-pages: 自动纠偏旋转
run([
    "ocrmypdf",
    "--force-ocr",
    "--rotate-pages",
    "--deskew",
    "--clean",
    "-l", "chi_sim+eng",
    str(MERGED_PDF),
    str(OCR_PDF)
])

print("✅ OCR PDF:", OCR_PDF)
