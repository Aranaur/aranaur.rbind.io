# MATH840 — Time Series, 2026/27

Autumn trimester, KSE. Tuesdays: lecture 10:00–11:20, practice 11:30–12:50 (GMT+3).

## Published materials

| What | Link |
|---|---|
| Week 1 — course syllabus deck | <https://01a05923-e02d-f78b-e75e-cec44127fc5d.share.connect.posit.cloud> |
| Week 1 — Introduction to Forecasting & Time Series Graphics | <https://01a05924-a704-8747-1dda-9917ad1a3d63.share.connect.posit.cloud> |
| Week 2 — Time Series Decomposition | <https://01a0784c-f128-848e-7cc4-a140ab0a2965.share.connect.posit.cloud> |
| Week 3 — The Forecaster's Toolbox | <https://01a09fdb-5a7c-a2ec-2f40-2979b1d48c0d.share.connect.posit.cloud> |
| Week 4 — Exponential Smoothing | <https://01a0beea-9ee5-a9ef-49a2-7329aa93e476.share.connect.posit.cloud> |
| Week 1 — practice notebook (Colab) | [Open in Colab](https://colab.research.google.com/github/Aranaur/aranaur.rbind.io/blob/main/lectures/kse/MATH840/26autumn/labs/_lab01.ipynb) |
| Week 2 — lab handout | <https://01a07874-b74d-873c-a106-e68a78304698.share.connect.posit.cloud> |
| Week 2 — practice notebook (Colab) | [Open in Colab](https://colab.research.google.com/github/Aranaur/aranaur.rbind.io/blob/main/lectures/kse/MATH840/26autumn/labs/_lab02.ipynb) |
| Week 3 — lab handout | <https://01a0a118-ef43-4671-3ca8-af3592c18aeb.share.connect.posit.cloud> |
| Week 3 — worked example (Colab) | [Open in Colab](https://colab.research.google.com/github/Aranaur/aranaur.rbind.io/blob/main/lectures/kse/MATH840/26autumn/labs/_lab03_worked_example.ipynb) |
| Week 3 — practice notebook (Colab) | [Open in Colab](https://colab.research.google.com/github/Aranaur/aranaur.rbind.io/blob/main/lectures/kse/MATH840/26autumn/labs/_lab03.ipynb) |
| Week 4 — practice demo, the toolbox end to end (Colab) | [Open in Colab](https://colab.research.google.com/github/Aranaur/aranaur.rbind.io/blob/main/lectures/kse/MATH840/26autumn/labs/_demo04.ipynb) |
| Week 4 — Lab 4 template (Colab) | [Open in Colab](https://colab.research.google.com/github/Aranaur/aranaur.rbind.io/blob/main/lectures/kse/MATH840/26autumn/labs/_lab04.ipynb) |
| Moodle | <https://teaching.kse.org.ua/course/view.php?id=4432> |

## Layout

```
slides/   .00 syllabus, .01–.04 weeks 1–4 (dot-prefixed: not rendered into the site)
labs/     .labNN.qmd handouts, _labNN.ipynb templates, _demoNN.ipynb practice demos,
          .demoNN-notes.md how to run the session
data/     datasets served to students over raw.githubusercontent.com
data/trackb/  the issued Track B series, one CSV per code, plus index.csv
infra/    publishing, deck PDFs, and the Track B pool and scorer
project/  final project (Week 10)
```

From Week 4 the practice session is a worked example on a shared series and the graded work goes
home, due 23:59 the same day. `labs/.demoNN-notes.md` is the running order for the session: what to
show, what to ask, and which answers to expect.

## Track B

```bash
cd lectures/kse/MATH840/26autumn/infra
python trackb_pool.py --jobs 16             # screen the pool and issue data/trackb/
python trackb_pool.py --refilter            # re-apply the issue rules to a pool already on disk
python trackb_score.py ../grading/submissions/lab04 --ungraded   # week 4: measured, not marked
python trackb_score.py ../grading/submissions/ch2 --out ../grading/ch2_scores.csv
```

`trackb_pool.py` cuts several variants of every source series, disguises each one (values rescaled,
calendar shifted by whole years, history trimmed), and issues only those that pass every screen: the
series is seasonal (F_S >= 0.6), an automatic ETS and an automatic ARIMA each beat its benchmark on
their own, and the better of the two lands between 0.70 and 0.90 of it. 140 of 2466 variants survive,
which is what makes the accuracy points from Week 6 measure the model rather than the draw - Week 4 is
scored with `--ungraded`, because with decomposition alone the benchmark falls about a third of the
time.

The served history goes to `data/trackb/`; the hidden holdout, the identities and the reference scores go to
`grading/trackb_pool.json`, which is gitignored and must stay that way — it is the answer key for
five challenges.

`.SYLLABUS-CHANGES.md` lists the edits the live syllabus still needs — including the
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
python publish_deck.py .lab02 --dir labs   # lab handouts live in labs/, rendered to html
```

### Assignment PDFs for AI grading

Moodle links to the handout, but the grading platform reads the assignment *file* and does not
follow links. Each lab handout therefore also ships as a PDF, printed from the rendered HTML with
Quarto's bundled headless Chrome — the same page students see, formulas included:

```bash
python handout_pdf.py .lab02            # print an already-rendered handout
python handout_pdf.py .lab03 --render   # render the .qmd first
```

The script checks the result before accepting it: every heading must be present in the extracted
text, and no raw TeX may remain (which would mean MathJax had not finished when the page was printed).

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
| 4 | `.04` | `.05` |
