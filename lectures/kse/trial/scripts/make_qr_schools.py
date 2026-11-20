"""QR-коди для виступу «ШІ вже в школі» (25.11.2026).

Зовнішні джерела кодуються одразу. Коди на розділи супровідної сторінки
і на саму презентацію з'являються, коли відомі їхні адреси на Posit Connect:

    uv run --no-project --with segno python scripts/make_qr_schools.py \
        --materials https://.../ --deck https://.../

Без адрес на місці цих кодів лишаються заглушки.
Запускати з теки lectures/kse/trial.
"""

import argparse
import sys
from pathlib import Path

import segno

OUT = Path("img/schools")
DARK = "#05006B"

SOURCES = {
    "survey-2026": "https://life.pravda.com.ua/society/yak-chasto-vchiteli-v-ukrajini-vikoristovuyut-shi-opituvannya-316294/",
    "mriia-grade": "https://www.kmu.gov.ua/news/shtuchnyi-intelekt-u-mrii-vchyteli-teper-stvoriuvatymut-testy-za-kilka-khvylyn",
    "mriia-500": "https://www.kmu.gov.ua/news/shkilnyi-test-u-kilka-klikiv-rozshyriuiemo-shi-henerator-u-ekosystemi-mriia",
    "pnas-turkey": "https://doi.org/10.1073/pnas.2422633122",
    "wb-nigeria": "https://blogs.worldbank.org/en/education/From-chalkboards-to-chatbots-Transforming-learning-in-Nigeria",
    "eef-chatgpt": "https://educationendowmentfoundation.org.uk/projects-and-evaluation/projects/choices-in-edtech-using-generative-ai-chatgpt-for-ks3-science-lesson-preparation-2024-teacher-choices-trial",
    "eef-aila": "https://educationendowmentfoundation.org.uk/projects-and-evaluation/projects/aila-teacher-choices-trial",
    "illinois": "https://arxiv.org/abs/2605.19043",
    "notebooklm": "https://notebooklm.google.com/",
    "korea": "https://restofworld.org/2025/south-korea-ai-textbook",
    "estonia": "https://hm.ee/en/news/estonia-announces-groundbreaking-national-initiative-ai-leap-programme-bring-ai-tools-all",
    "dfe": "https://www.gov.uk/government/news/ai-revolution-to-give-teachers-more-time-with-pupils",
    "openai-data": "https://help.openai.com/en/articles/5722486-how-your-data-is-used-to-improve-model-performance",
    "law-pd": "https://zakon.rada.gov.ua/laws/show/2297-17",
    "mon-ai": "https://mon.gov.ua/news/mon-i-mintsyfra-rozrobyly-proiekt-rekomendatsii-iz-vykorystannia-shi-v-shkolakh",
    "mon-profilna": "https://mon.gov.ua/news/profilna-reforma-starshoi-shkoly-vidpovidi-na-naiposhyrenishi-zapytannia",
}

# Розділи супровідної сторінки (якорі мають збігатися з .ai_schools_materials.qmd)
SECTIONS = [
    "kse", "polityka", "krash-test", "chotyry", "ocinyuvannya", "mriia",
    "notatky", "dani", "vorkshopy", "anketa", "batky", "pravyla", "ne-dopomozhe",
]

PLACEHOLDER = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 29 29">
<rect x="0.5" y="0.5" width="28" height="28" rx="2" fill="#fff" stroke="#8a8fb3" stroke-width="0.6" stroke-dasharray="1.5 1"/>
<text x="14.5" y="13" font-family="Inter, Arial, sans-serif" font-size="3" fill="#6b7096" text-anchor="middle">QR після</text>
<text x="14.5" y="17.5" font-family="Inter, Arial, sans-serif" font-size="3" fill="#6b7096" text-anchor="middle">публікації</text>
</svg>
"""


def save(name: str, url: str | None) -> None:
    path = OUT / f"qr-{name}.svg"
    if url is None:
        path.write_text(PLACEHOLDER, encoding="utf-8")
        return
    segno.make(url, error="m").save(
        path, kind="svg", border=2, dark=DARK, light="#ffffff", xmldecl=False, omitsize=True
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", help="адреса супровідної сторінки на Posit Connect")
    ap.add_argument("--deck", help="адреса презентації на Posit Connect")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        save(name, url)

    base = args.materials.rstrip("/") if args.materials else None
    save("materials", base)
    for s in SECTIONS:
        save(f"m-{s}", f"{base}#{s}" if base else None)
    save("deck", args.deck)

    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Записано {len(list(OUT.glob('qr-*.svg')))} файлів у {OUT}")


if __name__ == "__main__":
    main()
