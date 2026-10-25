"""Ask a small instruction-tuned LLM for forecasts, the way people ask a chat assistant.

Writes data/llm_forecasts.csv: for every product and backtest window, the 28 numbers the model
returned (padded with its last value, or cut, to 28) and how many numbers it actually gave.

The model is Qwen2.5-1.5B-Instruct, run locally on CPU with greedy decoding, so the run is
reproducible. It stands in for "paste the numbers into a chat and ask": a large chat model writes
better numbers, but the procedure - and the need to backtest it - is the same.

    HF_HOME=D:/hf-cache python infra/llm_forecasts.py      # ~45 minutes on a laptop CPU
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PRODUCTS = ["ChatGPT", "Duolingo", "Spotify", "Netflix", "Tinder", "Strava", "TikTok", "Notion", "Threads",
            "Black Friday"]
H, WINDOWS, CONTEXT = 28, 6, 56


def prompt(name: str, context: pd.Series) -> list[dict]:
    lines = "\n".join(f"{day:%Y-%m-%d} ({day:%a}): {int(v)}" for day, v in context.items())
    return [{"role": "system", "content": "You are a forecasting assistant. Answer only with numbers."},
            {"role": "user", "content": f"Daily Wikipedia page views of the article '{name}':\n{lines}\n\n"
                                        f"Forecast the next {H} days. Reply with exactly {H} integers "
                                        "separated by commas, nothing else."}]


def main() -> None:
    views = pd.read_csv(DATA / "pageviews.csv", parse_dates=["ds"])
    dates = np.sort(views["ds"].unique())
    cutoffs = [dates[len(dates) - (WINDOWS - k) * H - 1] for k in range(WINDOWS)]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    rows = []
    for name in PRODUCTS:
        s = views[views["unique_id"] == name].set_index("ds")["y"]
        for cutoff in cutoffs:
            ids = tokenizer.apply_chat_template(prompt(name, s[s.index <= cutoff][-CONTEXT:]),
                                                add_generation_prompt=True, return_tensors="pt")
            ids = ids if torch.is_tensor(ids) else ids["input_ids"]
            out = model.generate(ids, max_new_tokens=12 * H, do_sample=False)
            answer = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", answer)]
            forecast = (numbers + [numbers[-1] if numbers else s[s.index <= cutoff].iloc[-1]] * H)[:H]
            future = s[s.index > cutoff][:H]
            rows.append(pd.DataFrame({"unique_id": name, "cutoff": cutoff, "ds": future.index,
                                      "LLM": forecast, "numbers returned": len(numbers)}))
            print(name, pd.Timestamp(cutoff).date(), len(numbers), flush=True)
    pd.concat(rows).to_csv(DATA / "llm_forecasts.csv", index=False)


if __name__ == "__main__":
    main()
