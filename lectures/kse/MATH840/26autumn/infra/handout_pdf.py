"""Print a lab handout to PDF.

Moodle holds the assignment as a link, and the AI-grading platform reads the assignment
*file* - it does not follow links. So each handout also ships as a PDF.

The PDF is printed from the rendered HTML with the headless Chrome that Quarto bundles,
rather than rendered through LaTeX or Typst: it is then exactly the page students saw,
with the emoji headings, callouts and MathJax formulas intact, and no second rendering
path to drift out of sync with the first.

Usage:
    python handout_pdf.py .lab02              # print the already-rendered .lab02.html
    python handout_pdf.py .lab03 --render     # render the .qmd first
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABS = HERE.parent / "labs"

CHROME = os.environ.get(
    "CHROME",
    str(Path.home() / "AppData/Local/quarto/chromium/win64-869685/chrome-win/chrome.exe"),
)


def words(text: str) -> str:
    """Letters and digits only, lower-cased - so headings compare regardless of emoji and spacing."""
    return re.sub(r"[^0-9a-zа-яіїєґ]+", " ", text.lower()).strip()


def verify(qmd: Path, pdf: Path) -> None:
    if not shutil.which("pdftotext"):
        print("pdftotext not found - skipping content check")
        return

    # Ask for UTF-8 explicitly: the mingw build otherwise writes in the Windows code page.
    raw = subprocess.run(["pdftotext", "-enc", "UTF-8", "-layout", str(pdf), "-"],
                         capture_output=True, check=True).stdout
    text = raw.decode("utf-8", errors="replace")
    pages = text.count("\f") or 1
    flat = words(text)

    headings = [h for h in re.findall(r"(?m)^#{2,3} (.+)$", qmd.read_text(encoding="utf-8"))]
    missing = [h for h in headings if words(h) and words(h) not in flat]

    problems = []
    if missing:
        problems.append(f"headings not found in the PDF: {missing}")
    if "$$" in text or r"\hat{" in text or r"\frac{" in text:
        problems.append("raw TeX in the PDF - MathJax did not finish before printing")
    if len(flat) < 500:
        problems.append("almost no text extracted - the PDF may be an image")

    print(f"{pdf.name}: {pages} pages, {len(flat.split())} words, "
          f"{len(headings) - len(missing)}/{len(headings)} headings present")
    if problems:
        sys.exit("PDF check failed:\n  - " + "\n  - ".join(problems))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="handout stem, e.g. .lab02")
    ap.add_argument("--render", action="store_true", help="render the .qmd to html first")
    args = ap.parse_args()

    qmd = LABS / f"{args.stem}.qmd"
    html = LABS / f"{args.stem}.html"
    pdf = LABS / f"{args.stem}.pdf"

    if args.render:
        subprocess.run(["quarto", "render", qmd.name, "--to", "html", "--no-execute-daemon"],
                       cwd=LABS, check=True, shell=(os.name == "nt"))
    if not html.exists():
        sys.exit(f"{html.name} not found - render it first (--render)")
    if not Path(CHROME).exists():
        sys.exit(f"Chrome not found at {CHROME} - set the CHROME environment variable")

    pdf.unlink(missing_ok=True)
    subprocess.run([
        CHROME, "--headless", "--disable-gpu", "--no-first-run",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=20000",          # let MathJax typeset before the page is printed
        "--print-to-pdf-no-header",              # flag name in the Chromium Quarto bundles
        "--no-pdf-header-footer",                # flag name in current Chrome
        f"--print-to-pdf={pdf}",
        html.resolve().as_uri(),
    ], check=True, capture_output=True)

    if not pdf.exists() or pdf.stat().st_size == 0:
        sys.exit("Chrome produced no PDF")
    verify(qmd, pdf)
    print(f"written: {pdf}")


if __name__ == "__main__":
    main()
