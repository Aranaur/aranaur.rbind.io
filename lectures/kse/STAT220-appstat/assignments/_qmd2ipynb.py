import re, json, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else ".07-chi.qmd"
OUT = sys.argv[2] if len(sys.argv) > 2 else ".07-chi.ipynb"

raw = open(SRC, encoding="utf-8").read()
lines = raw.split("\n")
LANG = "en" if re.search(r"^lang:\s*en", raw, re.M) else "ua"

# --- strip YAML front matter ---
i = 0
if lines and lines[0].strip() == "---":
    i = 1
    while lines[i].strip() != "---":
        i += 1
    i += 1

cells = []
md_buf = []
FENCE = "```"


def flush_md():
    global md_buf
    out = []
    for ln in md_buf:
        s = ln.rstrip("\n")
        if re.match(r"^:{3,}\s*(\{.*\})?\s*$", s):        # ::: callout/column fences
            continue
        s = re.sub(r"\s*\{[.#][^}]*\}\s*$", "", s)        # heading {.attr}/{#id}
        out.append(s)
    while out and out[0].strip() == "":
        out.pop(0)
    while out and out[-1].strip() == "":
        out.pop()
    md_buf = []
    if out:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(out)})


while i < len(lines):
    ln = lines[i]
    if re.match(r"^" + re.escape(FENCE) + r"\{python\}", ln):
        flush_md()
        i += 1
        code = []
        while i < len(lines) and not lines[i].startswith(FENCE):
            if not lines[i].lstrip().startswith("#|"):    # drop quarto chunk options
                code.append(lines[i])
            i += 1
        i += 1                                            # skip closing fence
        while code and code[0].strip() == "":
            code.pop(0)
        while code and code[-1].strip() == "":
            code.pop()
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": "\n".join(code)})
    else:
        md_buf.append(ln)
        i += 1
flush_md()

if LANG == "en":
    intro = (
        "# Practice 7 — chi-square tests for categorical data\n\n"
        "**Applied Statistics · KSE** · Ihor Miroshnychenko\n\n"
        "Open this notebook in Google Colab and run the cells top to bottom "
        "(Runtime → Run all, or Shift+Enter one by one). All libraries "
        "(numpy, scipy, matplotlib, statsmodels) are already installed in "
        "Colab — there is nothing to copy.\n\n"
        "The **Guess first** and **Your turn** blocks are for you: write down "
        "your answer before looking at the solution below."
    )
else:
    intro = (
        "# Практика 7 — хі-квадрат тести для категоріальних даних\n\n"
        "**Applied Statistics · KSE** · Ihor Miroshnychenko\n\n"
        "Відкрийте цей зошит у Google Colab і виконуйте комірки згори вниз "
        "(Runtime → Run all, або Shift+Enter по черзі). Усі бібліотеки "
        "(numpy, scipy, matplotlib, statsmodels) у Colab уже встановлені — "
        "копіювати код нікуди не потрібно.\n\n"
        "Блоки **Спершу вгадайте** і **Ваша черга** — для вас: запишіть свою "
        "відповідь, перш ніж дивитись розв'язання нижче."
    )
cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": intro})

# normalize source -> list of lines with trailing newlines (nbformat convention)
for c in cells:
    parts = c["source"].split("\n")
    c["source"] = [p + "\n" for p in parts[:-1]] + [parts[-1]]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{OUT}: {len(cells)} cells "
      f"({sum(c['cell_type']=='code' for c in cells)} code, "
      f"{sum(c['cell_type']=='markdown' for c in cells)} md)")
