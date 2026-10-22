"""Deterministic native SVG, optional vector PDF and 300-dpi PNG.

All plotted values come from timing.py. No measurements or trained model are
loaded. The standard-library SVG writer preserves editable text and geometry.
"""
from html import escape
import math
from pathlib import Path
from .timing import legacy_heatmap_grid, retention_contour

WIDTH, HEIGHT = 1200, 800
PRINT_WIDTH_MM, PRINT_HEIGHT_MM = 180, 120


def _colour(value):
    light = (242, 247, 250)
    dark = (37, 89, 124)
    return "#" + "".join(f"{round(a + (b - a) * value):02X}" for a, b in zip(light, dark))


def scene():
    """Return shared vector objects so SVG and PDF have the same evidence."""
    items = []
    def text(x, y, value, size=23, anchor="start", rotation=0, colour="#243B53"):
        items.append(dict(kind="text", x=x, y=y, text=value, size=size,
                          anchor=anchor, rotation=rotation, colour=colour))
    def rect(x, y, w, h, colour):
        items.append(dict(kind="rect", x=x, y=y, w=w, h=h, colour=colour))
    def line(points, colour="#243B53", width=1.5, dash=False):
        items.append(dict(kind="line", points=points, colour=colour, width=width, dash=dash))
    x, y, w, h = 160, 120, 820, 520
    cx = lambda f: x + (f - 0.5) * w / 40
    cy = lambda sigma: y + h - (sigma + 0.5) * h / 21
    text(x, 52, "Coherent amplitude retained", 29)
    text(x, 86, "Gaussian residual relative timing error", 22, colour="#6F7E8A")
    line([(765, 80), (807, 80)], "#D3A13B", 2.5, True)
    text(822, 87, "H = 0.9", 22, colour="#A57415")
    for row in legacy_heatmap_grid():
        f, sigma = row["frequency_hz"], row["relative_jitter_sd_ms"]
        rect(x + (f - 1) * w / 40, y + h - (sigma + 1) * h / 21,
             w / 40, h / 21, _colour(row["coherent_amplitude_retained"]))
    contour = retention_contour()
    line([(cx(r["frequency_hz"]), cy(r["relative_jitter_sd_ms"])) for r in contour], "#D3A13B", 3, True)
    line([(x, y), (x, y + h), (x + w, y + h)])
    for f in (1, 10, 20, 30, 40):
        px = cx(f)
        line([(px, y + h), (px, y + h + 7)])
        text(px, y + h + 35, str(f), anchor="middle")
    for sigma in (0, 5, 10, 15, 20):
        py = cy(sigma)
        line([(x - 7, py), (x, py)])
        text(x - 17, py + 8, str(sigma), anchor="end")
    text(x + w / 2, 723, "Frequency (Hz)", 25, "middle")
    text(48, y + h / 2, "Residual relative jitter SD (ms)", 25, "middle", -90)
    text(1028, 99, "H", 24, "middle")
    for j in range(100):
        rect(1015, y + j * h / 100, 26, h / 100, _colour(1 - (j + 0.5) / 100))
    for value in (1, 0.5, 0):
        text(1055, y + (1 - value) * h + 8, str(value), 23)
    text(160, 775, "Analytical illustration. H is a relative coherent amplitude.", 20, colour="#6F7E8A")
    return items


def write_svg(destination):
    output = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{PRINT_WIDTH_MM}mm" height="{PRINT_HEIGHT_MM}mm" viewBox="0 0 {WIDTH} {HEIGHT}">',
              '<title>Gaussian relative timing-error budget</title>',
              '<desc>Equation-derived coherent-amplitude retention on the exact 840-cell frequency-jitter grid. Gold dashed curve: H=0.9. No empirical measurements.</desc>',
              '<rect width="1200" height="800" fill="white"/>']
    for a in scene():
        if a["kind"] == "rect":
            output.append(f'<rect x="{a["x"]:.8f}" y="{a["y"]:.8f}" width="{a["w"]:.8f}" height="{a["h"]:.8f}" fill="{a["colour"]}"/>')
        elif a["kind"] == "line":
            points = " ".join(f'{px:.8f},{py:.8f}' for px, py in a["points"])
            dash = ' stroke-dasharray="8 6"' if a["dash"] else ""
            output.append(f'<polyline points="{points}" fill="none" stroke="{a["colour"]}" stroke-width="{a["width"]}"{dash}/>')
        else:
            transform = f' transform="rotate({a["rotation"]} {a["x"]} {a["y"]})"' if a["rotation"] else ""
            output.append(f'<text x="{a["x"]}" y="{a["y"]}" font-family="Arial, Helvetica, sans-serif" font-size="{a["size"]}" text-anchor="{a["anchor"]}" fill="{a["colour"]}"{transform}>{escape(a["text"])}</text>')
    Path(destination).write_text("\n".join(output) + "\n</svg>\n", encoding="utf-8", newline="\n")


def write_pdf(destination):
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.colors import HexColor
    except ImportError as exc:
        raise RuntimeError("PDF export requires the optional plotting dependencies") from exc
    scale = PRINT_WIDTH_MM / 25.4 * 72 / WIDTH
    c = canvas.Canvas(str(destination), pagesize=(WIDTH * scale, HEIGHT * scale), invariant=1, pageCompression=1)
    c.setTitle("Gaussian relative timing-error budget")
    c.setAuthor("")
    c.setSubject("Analytical illustration; no empirical measurements")
    for a in scene():
        c.setFillColor(HexColor(a["colour"]))
        c.setStrokeColor(HexColor(a["colour"]))
        if a["kind"] == "rect":
            c.rect(a["x"] * scale, (HEIGHT - a["y"] - a["h"]) * scale,
                   a["w"] * scale, a["h"] * scale, stroke=0, fill=1)
        elif a["kind"] == "line":
            c.setLineWidth(a["width"] * scale)
            c.setDash([8 * scale, 6 * scale] if a["dash"] else [])
            p = c.beginPath()
            for i, (px, py) in enumerate(a["points"]):
                (p.moveTo if i == 0 else p.lineTo)(px * scale, (HEIGHT - py) * scale)
            c.drawPath(p, stroke=1, fill=0)
        else:
            c.saveState()
            c.setFont("Helvetica", a["size"] * scale)
            c.translate(a["x"] * scale, (HEIGHT - a["y"]) * scale)
            c.rotate(-a["rotation"])
            draw = {"start": c.drawString, "middle": c.drawCentredString, "end": c.drawRightString}[a["anchor"]]
            draw(0, 0, a["text"])
            c.restoreState()
    c.showPage()
    c.save()


def write_png(pdf_source, destination, dpi=300):
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 72:
        raise ValueError("dpi must be an integer of at least 72")
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("PNG export requires the optional plotting dependencies") from exc
    document = pdfium.PdfDocument(str(pdf_source))
    page = document[0]
    bitmap = page.render(scale=dpi / 72)
    image = bitmap.to_pil().convert("RGB")
    image.save(destination, format="PNG", dpi=(dpi, dpi), compress_level=9)
    image.close()
    bitmap.close()
    page.close()
    document.close()
