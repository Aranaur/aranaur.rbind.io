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

# TeX that survives into the printed text means MathJax had not typeset the page yet.
RAW_TEX = ("$$", "\\hat{", "\\frac{")


def words(text: str) -> str:
    """Letters and digits only, lower-cased - so headings compare regardless of emoji and spacing."""
    return re.sub(r"[^0-9a-zа-яіїєґ]+", " ", text.lower()).strip()


def pdf_text(pdf: Path) -> str:
    # Ask for UTF-8 explicitly: the mingw build otherwise writes in the Windows code page.
    raw = subprocess.run(["pdftotext", "-enc", "UTF-8", "-layout", str(pdf), "-"],
                         capture_output=True, check=True).stdout
    return raw.decode("utf-8", errors="replace")


def raw_tex(pdf: Path) -> int:
    text = pdf_text(pdf)
    return sum(text.count(token) for token in RAW_TEX)


def print_pdf(url: str, pdf: Path, budget_ms: int = 30000, attempts: int = 10) -> int:
    """Print `url` to `pdf`, retrying until no formula is left untypeset.

    MathJax loads from a CDN, and whether it has finished when Chrome decides to print is a
    race: the same deck prints clean on one run and with raw TeX on the next, and a longer
    virtual-time budget does not change the odds. So the check itself is the readiness
    signal - print, look for surviving TeX, print again if there is any.
    """
    for attempt in range(1, attempts + 1):
        pdf.unlink(missing_ok=True)
        subprocess.run([
            CHROME, "--headless", "--disable-gpu", "--no-first-run",
            "--run-all-compositor-stages-before-draw",
            f"--virtual-time-budget={budget_ms}",
            "--print-to-pdf-no-header",              # flag name in the Chromium Quarto bundles
            "--no-pdf-header-footer",                # flag name in current Chrome
            f"--print-to-pdf={pdf}", url,
        ], check=True, capture_output=True)
        if not pdf.exists() or pdf.stat().st_size == 0:
            continue
        if not shutil.which("pdftotext") or raw_tex(pdf) == 0:
            return attempt
    sys.exit(f"{pdf.name}: formulas still untypeset after {attempts} attempts")


def verify(qmd: Path, pdf: Path) -> None:
    if not shutil.which("pdftotext"):
        print("pdftotext not found - skipping content check")
        return

    text = pdf_text(pdf)
    pages = text.count("\f") or 1
    flat = words(text)

    # Drop Pandoc attributes ({.smaller}, {.tiny}, {#id}) - they are markup, not printed text -
    # and inline math: "$\lambda$" prints as "λ" once typeset, so only the words around it can match.
    headings = [re.sub(r"\$[^$]*\$", " ", re.sub(r"\s*\{[^}]*\}\s*$", "", h))
                for h in re.findall(r"(?m)^#{2,3} (.+)$", qmd.read_text(encoding="utf-8"))]
    missing = [h for h in headings if words(h) and words(h) not in flat]

    problems = []
    if missing:
        problems.append(f"headings not found in the PDF: {missing}")
    if any(token in text for token in RAW_TEX):
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

    attempts = print_pdf(html.resolve().as_uri(), pdf, budget_ms=20000)
    verify(qmd, pdf)
    print(f"written: {pdf}" + (f" (typeset on attempt {attempts})" if attempts > 1 else ""))


if __name__ == "__main__":
    main()
