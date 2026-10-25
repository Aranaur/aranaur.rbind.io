"""Download daily English-Wikipedia page views for the products used in the lecture.

Writes data/pageviews.csv (unique_id, ds, y). Source: the Wikimedia pageviews REST API, user agents
only; the data are released under CC0. Rerun to extend the series:

    python infra/fetch_pageviews.py
"""

import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "pageviews.csv"
END = "20261024"
PAGES = {
    "ChatGPT": "ChatGPT",
    "Duolingo": "Duolingo",
    "Spotify": "Spotify",
    "Netflix": "Netflix",
    "Tinder": "Tinder_(app)",
    "Strava": "Strava",
    "TikTok": "TikTok",
    "Notion": "Notion_(productivity_software)",
    "Threads": "Threads_(social_network)",
    "Black Friday": "Black_Friday_(shopping)",
    "Valentine's Day": "Valentine's_Day",
    "Lionel Messi": "Lionel_Messi",
}
HEADERS = {"User-Agent": "aranaur.rbind.io lecture materials (https://aranaur.rbind.io)"}


def fetch(article: str) -> pd.Series:
    url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
           f"{requests.utils.quote(article, safe='')}/daily/2015070100/{END}00")
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    items = response.json()["items"]
    return pd.Series({pd.Timestamp(i["timestamp"][:8]): i["views"] for i in items})


def main() -> None:
    frames = []
    for name, article in PAGES.items():
        s = fetch(article).asfreq("D")
        frames.append(pd.DataFrame({"unique_id": name, "ds": s.index, "y": s.to_numpy()}))
        print(f"{name:16s} {s.index.min():%Y-%m-%d} .. {s.index.max():%Y-%m-%d}  missing days: {s.isna().sum()}")
        time.sleep(0.3)
    pd.concat(frames).to_csv(OUT, index=False)
    print("written", OUT)


if __name__ == "__main__":
    main()
