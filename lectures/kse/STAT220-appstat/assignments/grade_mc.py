"""
grade_mc.py — INSTRUCTOR ONLY. Reveal the hidden criterion world for a roster.

Lab 5 (Monte Carlo) companion to ``grade.py``. Reads a list of student emails
(a CSV roster) and prints, for each one, the hidden criterion from
:mod:`greenbowl_mc`, the verdict the lab expects, and the decisive simulation
evidence. Optionally writes the table back to CSV.

Usage
-----
    # one email
    uv run --with numpy --with scipy --with pandas python grade_mc.py student5@kse.org.ua

    # a whole roster (any CSV; the email column is auto-detected)
    uv run --with numpy --with scipy --with pandas python grade_mc.py roster.csv
    uv run --with numpy --with scipy --with pandas python grade_mc.py roster.csv --out grades.csv

On Windows add ``PYTHONUTF8=1`` in front so Cyrillic prints cleanly.

The roster CSV may have a header. The email column is taken as the first
column whose name contains "email"/"mail"/"пошта"/"id", or — if there is no
such header — the first column. Blank lines are skipped.
"""

import argparse
import csv
import sys

import greenbowl_mc as g


def reference(email):
    """Return a dict: email -> (criterion, verdict, evidence)."""
    r = g._reveal(email)
    return {"email": email, **r}


def _read_emails(path):
    """Pull the email column out of a roster CSV (header optional)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    header = rows[0]
    keys = ("email", "mail", "пошта", "id")
    col = 0
    if any(any(k in cell.lower() for k in keys) for cell in header):
        col = next(i for i, cell in enumerate(header)
                   if any(k in cell.lower() for k in keys))
        rows = rows[1:]  # drop the header
    return [row[col].strip() for row in rows if row and row[col].strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Reveal the hidden criterion world per student (Lab 5).")
    ap.add_argument("source", help="a student email, or a path to a roster CSV")
    ap.add_argument("--out", help="write the full table to this CSV", default=None)
    args = ap.parse_args(argv)

    emails = (_read_emails(args.source) if args.source.lower().endswith(".csv")
              else [args.source])
    results = [reference(e) for e in emails]

    w = max((len(r["email"]) for r in results), default=5)
    for r in results:
        print(f"{r['email']:{w}s}  {r['criterion']:13s}  -> {r['verdict']}")
        print(f"{'':{w}s}  evidence: {r['evidence']}")

    if args.out:
        cols = ["email", "criterion", "verdict", "evidence"]
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            wr.writerows(results)
        print(f"\nSaved {len(results)} rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
