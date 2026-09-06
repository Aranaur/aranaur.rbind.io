"""Publish a rendered MATH840 deck to Posit Connect Cloud as static content.

Connect Cloud would otherwise try to render the deck itself, which means installing
duckdb, statsforecast, plotly and friends on its side. We render locally and ship the
HTML, so the deployed bundle has no runtime dependencies at all.

Two things the bundle needs fixing for:
  * rsconnect skips dot-prefixed files, and our decks are `.00.qmd` / `.01.qmd`;
  * the deck refers to its support directory by name (`.00_files/...`).
So the staging copy renames the deck to `index.html` and rewrites those references.

Usage:
    python publish_deck.py .00                     # publish/update the syllabus deck
    python publish_deck.py .01 --render            # re-render first, then publish
    python publish_deck.py .lab02 --dir labs       # publish a lab handout

Credentials come from the environment (never commit them):
    POSIT_CC_CLIENT_ID, POSIT_CC_CLIENT_SECRET, POSIT_CC_ACCOUNT
Generate them at connect.posit.cloud → account → Client Credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE / "published.json"

RSCRIPT = os.environ.get("RSCRIPT", r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe")
RLIB = os.environ.get("RSCONNECT_LIB", "")

# Runtime assets that live next to the deck rather than inside its support directory.
SIDECARS = ("img", "colored-particles")

DEPLOY_R = r"""
lib <- Sys.getenv("RSCONNECT_LIB")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressMessages(library(rsconnect))

a <- commandArgs(trailingOnly = TRUE)
dir <- a[1]; name <- a[2]; title <- a[3]; recordDir <- a[4]

connectCloudClientCredentials(
  clientId     = Sys.getenv("POSIT_CC_CLIENT_ID"),
  clientSecret = Sys.getenv("POSIT_CC_CLIENT_SECRET"),
  accountName  = Sys.getenv("POSIT_CC_ACCOUNT"),
  quiet        = TRUE
)

args <- list(
  appDir         = dir,
  recordDir      = recordDir,
  appName        = name,
  appTitle       = title,
  appPrimaryDoc  = "index.html",
  server         = "connect.posit.cloud",
  account        = Sys.getenv("POSIT_CC_ACCOUNT"),
  launch.browser = FALSE,
  forceUpdate    = TRUE,
  lint           = FALSE
)
do.call(deployApp, args)

d <- deployments(recordDir)
cat("DEPLOY_ID=", d$appId[nrow(d)], "\n", sep = "")
"""


def stage(src: Path, stem: str, out: Path) -> None:
    """Copy the rendered document into `out` under names rsconnect will actually bundle."""
    html_src = src / f"{stem}.html"
    files_src = src / f"{stem}_files"
    if not html_src.exists():
        sys.exit(f"{html_src} not found - render the deck first (--render)")

    html = html_src.read_text(encoding="utf-8").replace(f"{stem}_files/", "index_files/")
    (out / "index.html").write_text(html, encoding="utf-8")
    shutil.copytree(files_src, out / "index_files")
    for side in SIDECARS:
        if (src / side).is_dir():
            shutil.copytree(src / side, out / side)

    refs = {
        r.split("?")[0].replace("%20", " ")
        for r in re.findall(
            r'(?:src|href|data-background-iframe)="((?!https?:|data:|mailto:|#|//)[^"]+)"', html
        )
    }
    missing = sorted(r for r in refs if not (out / r).exists())
    if missing:
        sys.exit(f"bundle is incomplete, these are referenced but absent: {missing}")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"staged {len(refs)} local references, {size / 1048576:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="document stem, e.g. .00, .01 or .lab02")
    ap.add_argument("--dir", default="slides", help="source directory (default: slides)")
    ap.add_argument("--render", action="store_true", help="re-render the deck first")
    ap.add_argument("--name", help="Connect Cloud content name (defaults to the record)")
    ap.add_argument("--title", help="Connect Cloud content title")
    args = ap.parse_args()

    for var in ("POSIT_CC_CLIENT_ID", "POSIT_CC_CLIENT_SECRET", "POSIT_CC_ACCOUNT"):
        if not os.environ.get(var):
            sys.exit(f"{var} is not set")

    record = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else {}
    entry = record.get(args.stem, {})
    name = args.name or entry.get("name") or f"math840{args.stem.replace('.', '-')}"
    title = args.title or entry.get("title") or name

    src = HERE.parent / args.dir
    fmt = "revealjs" if args.dir == "slides" else "html"

    if args.render:
        subprocess.run(
            ["quarto", "render", f"{args.stem}.qmd", "--to", fmt, "--no-execute-daemon"],
            cwd=src, check=True, shell=(os.name == "nt"),
        )

    # rsconnect tracks "which content did this directory become" next to the sources it
    # deployed. Ours is a throwaway staging copy, so the record is kept here instead -
    # without it every deploy would create a new URL rather than updating the old one.
    records = HERE / "rsconnect-records" / name
    records.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / name
        out.mkdir()
        stage(src, args.stem, out)

        env = {**os.environ, "RSCONNECT_LIB": RLIB}
        script = Path(tmp) / "deploy.R"
        script.write_text(DEPLOY_R, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script), str(out), name, title, str(records)],
            env=env, text=True, capture_output=True,
        )
        print(proc.stdout[-2000:] or proc.stderr[-2000:])
        if proc.returncode != 0:
            sys.exit("deployment failed")

        found = re.search(r"DEPLOY_ID=(\S+)", proc.stdout)
        if not found:
            sys.exit("deployment reported no content id")
        content_id = found.group(1)

    record[args.stem] = {
        "name": name,
        "title": title,
        "id": content_id,
        "url": f"https://{content_id}.share.connect.posit.cloud",
    }
    RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{args.stem} -> {record[args.stem]['url']}")


if __name__ == "__main__":
    main()
