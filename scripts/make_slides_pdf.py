from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
SLIDES_MD = ROOT / "slides" / "entrega_final.md"
SLIDES_PDF = ROOT / "slides" / "entrega_final.pdf"


def parse_slides(markdown: str) -> list[tuple[str, list[str]]]:
    slides: list[tuple[str, list[str]]] = []
    title = ""
    lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            if title or lines:
                slides.append((title, lines))
            title = line[3:].strip()
            lines = []
        elif line:
            lines.append(line)
    if title or lines:
        slides.append((title, lines))
    return slides


def draw_slide(pdf: PdfPages, title: str, lines: list[str]):
    fig = plt.figure(figsize=(11, 6.2))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.08, 0.88, title, fontsize=26, fontweight="bold", va="top", wrap=True)

    y = 0.74
    for line in lines:
        text = line[2:] if line.startswith("- ") else line
        prefix = "\u2022 " if line.startswith("- ") else ""
        ax.text(0.1, y, prefix + text, fontsize=15, va="top", wrap=True)
        y -= 0.075
        if y < 0.12:
            break

    pdf.savefig(fig)
    plt.close(fig)


def main():
    markdown = SLIDES_MD.read_text(encoding="utf-8")
    slides = parse_slides(markdown)
    SLIDES_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(SLIDES_PDF) as pdf:
        for title, lines in slides:
            draw_slide(pdf, title, lines)
    print(f"wrote {SLIDES_PDF}")


if __name__ == "__main__":
    main()
