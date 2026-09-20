"""Collect every student-facing MATH840 material into one folder of Gemini notebook sources.

Run it after any deck or handout changes; it rebuilds the folder from scratch:

    python notebook_sources.py

What goes in, all discovered automatically so a new week needs no edits here:
  * lecture decks (slides/.NN.html)  - printed with every folded code block opened, since a
    collapsed <details> prints as nothing and the notebook would never see the code;
  * lab handouts (labs/.labNN.html);
  * worked examples (labs/_labNN_worked_example.ipynb) - executed first, because the
    notebooks are stored without outputs and would otherwise print with no figures;
  * fpp_urls.txt - textbook chapters for the weeks covered so far, to add as website sources;
  * MANIFEST.md - what each file is, and what was left out on purpose.

Never included: grading/ (answers), SYLLABUS-CHANGES.md (instructor notes), and the syllabus
PDF while it still contradicts the rules deck.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from handout_pdf import CHROME, verify  # noqa: E402

COURSE = HERE.parent
SLIDES = COURSE / "slides"
LABS = COURSE / "labs"
OUT = COURSE / "ai-support" / "notebook-sources"

FPP = "https://otexts.com/fpppy/nbs"
# Course plan: which textbook chapters each lecture week draws on.
CHAPTERS_BY_WEEK = {
    1: ["01-intro", "02-graphics"],
    2: ["03-decomposition", "04-features"],
    3: ["05-toolbox"],
    4: ["08-exponential-smoothing"],
    5: ["07-regression"],
    6: ["09-arima"],
    7: ["09-arima", "10-dynamic-regression"],
    8: ["12-advanced"],
    9: ["14-neural-networks", "15-foundation-models"],
}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def title_of(qmd: Path) -> str:
    m = re.search(r'(?m)^title:\s*"?(.+?)"?\s*$', qmd.read_text(encoding="utf-8"))
    return m.group(1) if m else qmd.stem


def chrome_print(url: str, pdf: Path, budget_ms: int = 30000) -> None:
    # Retries until MathJax has typeset every formula - see handout_pdf.print_pdf for why.
    from handout_pdf import print_pdf
    attempts = print_pdf(url, pdf, budget_ms)
    if attempts > 1:
        print(f"{pdf.name}: formulas typeset on attempt {attempts}")


def stale(qmd: Path, html: Path) -> bool:
    return qmd.exists() and html.exists() and qmd.stat().st_mtime > html.stat().st_mtime


def pdf_text(pdf: Path) -> str:
    raw = subprocess.run(["pdftotext", "-enc", "UTF-8", str(pdf), "-"],
                         capture_output=True, check=True).stdout
    return raw.decode("utf-8", errors="replace")


def calls_in_visible_code(qmd: Path) -> list[str]:
    """Function calls a reader sees in the deck: chunks that are not hidden and are evaluated."""
    calls = []
    for block in re.findall(r"(?ms)^```\{python\}\n(.*?)^```", qmd.read_text(encoding="utf-8")):
        options = dict(re.findall(r"(?m)^#\|\s*([\w-]+):\s*(\S+)", block))
        if "false" in (options.get("include"), options.get("echo"), options.get("eval")):
            continue
        calls += re.findall(r"\b([A-Za-z_][A-Za-z_0-9]{5,})\(", block)
    # Most common first: those are the calls the deck is actually teaching.
    return [name for name, _ in Counter(calls).most_common()]


def code_not_printed(qmd: Path, pdf: Path) -> list[str]:
    """The deck's three most-used visible calls that are missing from the printed text.

    A deck whose code never reaches the PDF is useless as a notebook source - the reader sees
    results with no way to reproduce them - and that is exactly what a collapsed <details> gives.
    """
    wanted = calls_in_visible_code(qmd)[:3]
    if not wanted:
        return []
    text = pdf_text(pdf)
    missing = [f"{name}()" for name in wanted if f"{name}(" not in text]
    return missing if len(missing) == len(wanted) else []


def build_decks(manifest: list, warnings: list) -> int:
    weeks = []
    for qmd in sorted(SLIDES.glob(".[0-9][0-9].qmd")):
        stem = qmd.stem                                   # ".03"
        html = SLIDES / f"{stem}.html"
        if not html.exists():
            warnings.append(f"{qmd.name}: not rendered, skipped")
            continue
        if stale(qmd, html):
            warnings.append(f"{qmd.name} is newer than its HTML - re-render before uploading")

        n = int(stem[1:])
        week = max(1, n)
        weeks.append(week)
        name = "week01_course_rules" if n == 0 else f"week{week:02d}_lecture_{slug(title_of(qmd))}"
        pdf = OUT / f"{name}.pdf"

        # Open every folded code block, next to the original so relative assets resolve.
        printable = SLIDES / f"{stem}.__print__.html"
        printable.write_text(html.read_text(encoding="utf-8").replace("<details", "<details open"),
                             encoding="utf-8")
        try:
            chrome_print(printable.resolve().as_uri() + "?print-pdf", pdf)
        finally:
            printable.unlink(missing_ok=True)

        verify(qmd, pdf)
        missing = code_not_printed(qmd, pdf)
        if missing:
            sys.exit(f"{pdf.name}: the deck shows code the PDF does not contain, "
                     f"e.g. {', '.join(missing)} - were the folded blocks opened?")
        manifest.append((pdf.name, "Лекція" if n else "Правила курсу",
                         f"тиждень {week}", title_of(qmd)))
    return max(weeks, default=1)


def build_handouts(manifest: list, warnings: list) -> None:
    for qmd in sorted(LABS.glob(".lab[0-9][0-9].qmd")):
        stem = qmd.stem                                   # ".lab03"
        html = LABS / f"{stem}.html"
        if not html.exists():
            warnings.append(f"{qmd.name}: not rendered, skipped")
            continue
        if stale(qmd, html):
            warnings.append(f"{qmd.name} is newer than its HTML - re-render before uploading")
        n = int(stem[4:])
        title = title_of(qmd)
        pdf = OUT / f"lab{n:02d}_{slug(title.split(':', 1)[-1])}.pdf"
        chrome_print(html.resolve().as_uri(), pdf, budget_ms=20000)
        verify(qmd, pdf)
        manifest.append((pdf.name, "Хендаут лабораторної", f"практика тижня {n}", title))


def build_worked_examples(manifest: list) -> None:
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    for ipynb in sorted(LABS.glob("_lab[0-9][0-9]_worked_example.ipynb")):
        n = int(ipynb.stem[4:6])
        nb = nbformat.read(ipynb, as_version=4)
        for cell in nb.cells:                              # packages are already installed here
            if cell.cell_type == "code":
                cell.source = "\n".join(l for l in cell.source.split("\n")
                                        if not l.lstrip().startswith("!pip"))
        NotebookClient(nb, timeout=900, kernel_name="python3").execute()
        errors = [o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
        if errors:
            sys.exit(f"{ipynb.name}: {len(errors)} cell(s) raised - fix before publishing it as a source")

        body, _ = HTMLExporter().from_notebook_node(nb)
        pdf = OUT / f"lab{n:02d}_worked_example.pdf"
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "worked_example.html"
            page.write_text(body, encoding="utf-8")
            chrome_print(page.as_uri(), pdf)
        text = pdf_text(pdf)
        figures = sum(1 for c in nb.cells for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
        if len(text.split()) < 300 or figures == 0:
            sys.exit(f"{pdf.name}: too little content ({len(text.split())} words, {figures} figures)")
        print(f"{pdf.name}: executed, {figures} figures, {len(text.split())} words")
        manifest.append((pdf.name, "Розібраний приклад", f"до практики тижня {n}",
                         "Повністю виконане завдання на ряді поза студентським пулом"))


def write_urls(last_week: int) -> list[str]:
    chapters = []
    for week in range(1, last_week + 1):
        for ch in CHAPTERS_BY_WEEK.get(week, []):
            if ch not in chapters:
                chapters.append(ch)
    urls = [f"{FPP}/{ch}.html" for ch in chapters]
    (OUT / "fpp_urls.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
    return urls


def write_manifest(manifest: list, urls: list[str], warnings: list) -> None:
    rows = "\n".join(f"| `{f}` | {kind} | {when} | {what} |" for f, kind, when, what in manifest)
    links = "\n".join(f"- {u}" for u in urls)
    warn = "\n".join(f"- ⚠️ {w}" for w in warnings) or "- немає"
    text = f"""# MATH840 — джерела для Gemini notebook

Згенеровано `infra/notebook_sources.py`. Тека не комітиться: це відтворюваний набір, перезбирайте
його після кожної зміни дека чи хендауту.

## Файли для завантаження

| Файл | Що це | Коли | Назва |
|---|---|---|---|
{rows}

## Посилання на підручник

Додайте як джерела типу «вебсайт» (той самий список у `fpp_urls.txt`):

{links}

## Навмисно НЕ включено

- **`MATH840-TimeSeries26-27.pdf`** — суперечить актуальним правилам: у ньому вісім лабораторних по
  7 балів, 44 години практик, стеля асинхронної заміни 5/7 і немає правила лагу. Notebook цитуватиме
  його впевнено й неправильно. Додайте лише після того, як перенесете `SYLLABUS-CHANGES.md` у документ.
- **`grading/`** — еталонні відповіді до лабораторних.
- **`SYLLABUS-CHANGES.md`** — нотатки викладача, не для студентів.
- **Colab-шаблони** (`_labNN.ipynb`) — повторюють хендаути й містять лише заготовки.
- **Інструкції notebook** (`../notebook_instructions.txt`) — вставляються в *Configure chat*, а не
  завантажуються як джерело: інакше студенти бачитимуть їх у цитатах.

## Після оновлення

Джерело фіксується в момент додавання. Якщо дек змінився — видаліть старе джерело в notebook і
завантажте новий PDF.

## Попередження збірки

{warn}
"""
    (OUT / "MANIFEST.md").write_text(text, encoding="utf-8")


def main() -> None:
    if not Path(CHROME).exists():
        sys.exit(f"Chrome not found at {CHROME}")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    manifest, warnings = [], []
    last_week = build_decks(manifest, warnings)
    build_handouts(manifest, warnings)
    build_worked_examples(manifest)
    urls = write_urls(last_week)
    write_manifest(manifest, urls, warnings)

    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"\n{len(manifest)} PDFs + {len(urls)} textbook URLs -> {OUT}  ({total / 1048576:.1f} MB)")
    for w in warnings:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
