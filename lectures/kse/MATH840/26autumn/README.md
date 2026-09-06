# MATH840 — Time Series, 2026/27

Autumn trimester, KSE. Tuesdays: lecture 10:00–11:20, practice 11:30–12:50 (GMT+3).

## Published materials

| What | Link |
|---|---|
| Week 1 — course syllabus deck | <https://01a05923-e02d-f78b-e75e-cec44127fc5d.share.connect.posit.cloud> |
| Week 1 — Introduction to Forecasting & Time Series Graphics | <https://01a05924-a704-8747-1dda-9917ad1a3d63.share.connect.posit.cloud> |
| Week 2 — Time Series Decomposition | <https://01a0784c-f128-848e-7cc4-a140ab0a2965.share.connect.posit.cloud> |
| Week 1 — practice notebook (Colab) | [Open in Colab](https://colab.research.google.com/github/Aranaur/aranaur.rbind.io/blob/main/lectures/kse/MATH840/26autumn/labs/_lab01.ipynb) |
| Week 2 — practice notebook (Colab) | [Open in Colab](https://colab.research.google.com/github/Aranaur/aranaur.rbind.io/blob/main/lectures/kse/MATH840/26autumn/labs/_lab02.ipynb) |
| Moodle | <https://teaching.kse.org.ua/course/view.php?id=4432> |

## Layout

```
slides/   .00 syllabus, .01 week 1, .02 week 2 (dot-prefixed: not rendered into the site)
labs/     .labNN.qmd handouts, _labNN.ipynb Colab notebooks
data/     datasets served to students over raw.githubusercontent.com
infra/    publishing, and later the Track B series pool and scorer
project/  final project (Week 10)
```

`SYLLABUS-CHANGES.md` lists the edits the live syllabus still needs — including the
practice-hours error (44 in the document, 22 in reality) and the move from eight
7-point labs to seven 8-point ones.

## Republishing a deck

Decks are rendered locally and shipped to Posit Connect Cloud as **static** content, so
the platform never has to install duckdb, statsforecast or plotly to display them.

```bash
export POSIT_CC_CLIENT_ID=...        # connect.posit.cloud → account → Client Credentials
export POSIT_CC_CLIENT_SECRET=...
export POSIT_CC_ACCOUNT=aranaur
export RSCONNECT_LIB=/path/to/r/library   # a library containing rsconnect >= 1.11

cd lectures/kse/MATH840/26autumn/infra
python publish_deck.py .00            # update the syllabus deck
python publish_deck.py .01 --render   # re-render the lecture, then update it
```

The URL never changes between deployments: `infra/rsconnect-records/` holds the
rsconnect deployment record for each deck, which is what tells it to update existing
content instead of creating new content with a new address. Do not delete that
directory — losing it means new URLs, and the old ones are already in students' hands.
`infra/published.json` is the human-readable index of what is deployed where.

Credentials are read from the environment and are never stored in this repository.

## Environment note

The decks need more than `pyproject.toml` declares: `statsforecast`, `utilsforecast`,
`statsmodels`, `scipy`, `scikit-learn`, `duckdb`, `plotly`, `seaborn`, `tsfeatures`,
`great_tables` and `fpppy`. They are installed in `.venv` but not recorded as project
dependencies, so **`uv sync` removes them** and the decks stop rendering. Reinstall with
`uv pip install --link-mode=copy <packages>` if that happens.

Pandas is 3.0.5, which satisfies the `>= 3.0.2` pin. Pandas 3 dropped the legacy
frequency aliases, so use `QE`/`QS`, `ME`/`MS`, `YE`/`YS` and lowercase `h` in any new
material — a bare `Q` or `M` now raises instead of warning.

## Slide numbering

Weeks are offset by one against the 2025/26 decks, because Week 1 merged last year's
`.01` and `.02` into a single deck:

| Week | 26autumn | 25autumn source |
|---|---|---|
| 1 | `.01` | `.01` + `.02` |
| 2 | `.02` | `.03` |
| 3 | `.03` | `.04` |
