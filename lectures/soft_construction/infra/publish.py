"""Publish a rendered Software Construction document to Posit Connect Cloud as static content.

Adapted from lectures/kse/MATH840/26autumn/infra/publish_deck.py. We render locally and
ship the HTML, so the deployed bundle has no runtime dependencies at all.

Two things the bundle needs fixing for:
  * rsconnect skips dot-prefixed files, and our documents are `.01.qmd` / `.01-amazon.qmd`;
  * the document refers to its support directory by name (`.01_files/...`).
So the staging copy renames the document to `index.html` and rewrites those references.
Only the local files the page actually references are staged, not whole asset folders.

Render first, in a copy outside the repository: the repository's .Rprofile activates renv,
whose library lacks the packages the slides load.

Usage:
    python publish.py .01 --dir slides/2023 --name soft-construction-lecture01 --title "..."
    python publish.py .01 --dir labs/2023            # update: name and title come from the record

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
COURSE = HERE.parent
RECORD = HERE / "published.json"

RSCRIPT = os.environ.get("RSCRIPT", r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe")
RLIB = os.environ.get("RSCONNECT_LIB", "")

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


def local_refs(html: str) -> set[str]:
    refs = {
        r.split("?")[0].split("#")[0].replace("%20", " ")
        for r in re.findall(
            r'(?:src|href|data-background-iframe)="((?!https?:|data:|mailto:|#|//)[^"]+)"', html
        )
    }
    # Quarto's inline scripts build links from templates such as `${href}`; those are not files.
    return {r for r in refs if r and "${" not in r}


def stage(src: Path, stem: str, out: Path) -> None:
    """Copy the rendered document into `out` under names rsconnect will actually bundle."""
    html_src = src / f"{stem}.html"
    files_src = src / f"{stem}_files"
    if not html_src.exists():
        sys.exit(f"{html_src} not found - render the document first")

    html = html_src.read_text(encoding="utf-8").replace(f"{stem}_files/", "index_files/")
    (out / "index.html").write_text(html, encoding="utf-8")
    if files_src.is_dir():
        shutil.copytree(files_src, out / "index_files")

    refs = local_refs(html)
    for ref in sorted(refs):
        if ref.startswith("index_files/") or (out / ref).exists():
            continue
        source = src / ref
        if not source.is_file():
            continue
        # A page embedded in an iframe loads its own scripts and styles: take its folder.
        if source.name == "index.html" and source.parent != src:
            shutil.copytree(source.parent, out / source.parent.relative_to(src), dirs_exist_ok=True)
        else:
            (out / ref).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, out / ref)

    missing = sorted(r for r in refs if not (out / r).exists())
    if missing:
        sys.exit(f"bundle is incomplete, these are referenced but absent: {missing}")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"staged {len(refs)} local references, {size / 1048576:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="document stem, e.g. .01 or .01-amazon")
    ap.add_argument("--dir", required=True, help="source directory relative to the course, e.g. slides/2023")
    ap.add_argument("--name", help="Connect Cloud content name (defaults to the record)")
    ap.add_argument("--title", help="Connect Cloud content title (defaults to the record)")
    args = ap.parse_args()

    for var in ("POSIT_CC_CLIENT_ID", "POSIT_CC_CLIENT_SECRET", "POSIT_CC_ACCOUNT"):
        if not os.environ.get(var):
            sys.exit(f"{var} is not set")

    record = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else {}
    key = f"{args.dir}/{args.stem}"
    entry = record.get(key, {})
    name = args.name or entry.get("name")
    if not name:
        sys.exit(f"{key} has never been published - pass --name and --title")
    title = args.title or entry.get("title") or name

    # rsconnect tracks "which content did this directory become" next to the sources it
    # deployed. Ours is a throwaway staging copy, so the record is kept here instead -
    # without it every deploy would create a new URL rather than updating the old one.
    records = HERE / "rsconnect-records" / name
    records.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / name
        out.mkdir()
        stage(COURSE / args.dir, args.stem, out)

        env = {**os.environ, "RSCONNECT_LIB": RLIB}
        script = Path(tmp) / "deploy.R"
        script.write_text(DEPLOY_R, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script), str(out), name, title, str(records)],
            env=env, text=True, capture_output=True, encoding="utf-8", errors="replace",
        )
        print(proc.stdout[-2000:] or proc.stderr[-2000:])
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
            sys.exit("deployment failed")

        found = re.search(r"DEPLOY_ID=(\S+)", proc.stdout)
        if not found:
            sys.exit("deployment reported no content id")
        content_id = found.group(1)

    record[key] = {
        "name": name,
        "title": title,
        "id": content_id,
        "url": f"https://{content_id}.share.connect.posit.cloud",
    }
    RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{key} -> {record[key]['url']}")


if __name__ == "__main__":
    main()
