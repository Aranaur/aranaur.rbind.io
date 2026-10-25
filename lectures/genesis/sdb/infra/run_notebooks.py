"""Execute the worked-example notebooks and store them with their outputs.

The run uses the local data file and skips the pip cell; the stored notebook keeps the original Colab
cells (pip install, GitHub data URL) next to the outputs of this run.

    HF_HOME=D:/hf-cache python infra/run_notebooks.py        # several minutes per notebook
"""

import copy
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
LABS = HERE.parent / "labs"
LOCAL = (HERE.parent / "data" / "pageviews.csv").as_posix()
URL = "https://raw.githubusercontent.com/Aranaur/aranaur.rbind.io/main/lectures/genesis/sdb/data/pageviews.csv"


def run(name: str) -> None:
    original = nbformat.read(LABS / name, as_version=4)
    work = copy.deepcopy(original)
    for cell in work.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace("!pip install", "# !pip install").replace(URL, LOCAL)
    ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(work, {"metadata": {"path": str(LABS)}})
    for kept, done in zip(original.cells, work.cells):
        if kept.cell_type == "code":
            kept.outputs = [o for o in done.outputs if not (o.get("name") == "stderr")]
            kept.execution_count = done.execution_count
    nbformat.write(original, LABS / name)
    errors = [o for c in work.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    print(name, "errors:", len(errors))


if __name__ == "__main__":
    for name in sys.argv[1:] or ["worked_example.ipynb", "worked_example_uk.ipynb"]:
        run(name)
