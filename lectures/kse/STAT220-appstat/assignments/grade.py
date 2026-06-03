"""
grade.py — INSTRUCTOR ONLY. Reveal the hidden A/B world for a roster.

Reads a list of student emails (a CSV roster) and prints, for each one, the
hidden scenario from :mod:`greenbowl_ab`, the true effect, and the *reference*
business decision the lab expects. Optionally writes the table back to CSV.

Usage
-----
    # one email
    uv run --with numpy python grade.py student5@kse.org.ua

    # a whole roster (any CSV; the email column is auto-detected)
    uv run --with numpy python grade.py roster.csv
    uv run --with numpy python grade.py roster.csv --out grades.csv

On Windows add ``PYTHONUTF8=1`` in front so Cyrillic prints cleanly.

The roster CSV may have a header. The email column is taken as the first
column whose name contains "email"/"mail"/"пошта"/"id", or — if there is no
such header — the first column. Blank lines are skipped.
"""

import argparse
import csv
import sys

import greenbowl_ab as g

# Per-scenario: the METHOD the student must demonstrate, and the reference
# DECISION. Where the decision depends on the true effect vs the +15
# break-even (it is jittered per id), DECISION is None and we resolve it
# from the revealed `above_breakeven` flag.
_EXPECTED = {
    "clear_win":      ("t-test (pooled or Welch)",   "SHIP"),
    "marginal_win":   ("t-test",                      "DO NOT SHIP — significant but lift < 15"),
    "no_effect":      ("t-test",                      "DO NOT SHIP — no real effect"),
    "underpowered":   ("t-test + power analysis",     "EXTEND TEST — too few users to conclude"),
    "unequal_var":    ("Welch t (NOT pooled)",        None),
    "skewed_small_n": ("bootstrap (t unreliable)",    None),
}


def reference(email):
    """Return a dict: email -> (scenario, method, decision, truth fields)."""
    r = g._reveal(email)
    method, decision = _EXPECTED[r["scenario"]]
    if decision is None:  # resolve from the true effect vs break-even
        decision = ("SHIP" if r["above_breakeven"]
                    else "DO NOT SHIP — lift < 15") + f"  (via {method})"
    return {
        "email": email,
        "scenario": r["scenario"],
        "method": method,
        "decision": decision,
        "true_effect": r["true_effect_UAH"],
        "above_breakeven": r["above_breakeven"],
        "above_design_mde": r["above_design_mde"],
        "n_control": r["n_control"],
        "n_test": r["n_test"],
    }


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
    emails = [row[col].strip() for row in rows if row and row[col].strip()]
    return emails


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reveal the hidden A/B world per student.")
    ap.add_argument("source", help="a student email, or a path to a roster CSV")
    ap.add_argument("--out", help="write the full table to this CSV", default=None)
    args = ap.parse_args(argv)

    emails = (_read_emails(args.source) if args.source.lower().endswith(".csv")
              else [args.source])
    results = [reference(e) for e in emails]

    # pretty console table
    w = max((len(r["email"]) for r in results), default=5)
    for r in results:
        print(f"{r['email']:{w}s}  {r['scenario']:14s}  "
              f"effect={r['true_effect']:+6.2f}  -> {r['decision']}")

    if args.out:
        cols = ["email", "scenario", "method", "decision", "true_effect",
                "above_breakeven", "above_design_mde", "n_control", "n_test"]
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            wr.writerows(results)
        print(f"\nSaved {len(results)} rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
