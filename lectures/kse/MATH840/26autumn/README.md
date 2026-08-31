# MATH840 — Time Series, 2026/27

Autumn trimester, KSE. Tuesdays: lecture 10:00–11:20, practice 11:30–12:50 (GMT+3).

## Published materials

| What | Link |
|---|---|
| Week 1 — course syllabus deck | <https://01a05923-e02d-f78b-e75e-cec44127fc5d.share.connect.posit.cloud> |
| Week 1 — Introduction to Forecasting & Time Series Graphics | <https://01a05924-a704-8747-1dda-9917ad1a3d63.share.connect.posit.cloud> |
| Week 1 — practice notebook (Colab) | [Open in Colab](https://colab.research.google.com/github/Aranaur/aranaur.rbind.io/blob/main/lectures/kse/MATH840/26autumn/labs/_lab01.ipynb) |
| Moodle | <https://teaching.kse.org.ua/course/view.php?id=4432> |

## Layout

```
slides/   .00 syllabus, .01 lecture 1 (dot-prefixed: not rendered into the site)
labs/     .lab01.qmd handout, _lab01.ipynb Colab notebook
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

Rendering the decks needs `statsforecast`, which pins `pandas < 3`, while the project's
`pyproject.toml` asks for `pandas >= 3.0.2`. The working environment currently has
pandas 2.3.3 and renders correctly. **Running `uv sync` will restore pandas 3.0.2 and
remove statsforecast, duckdb, plotly and seaborn**, after which the lecture no longer
renders. Resolve the pin before running it.
