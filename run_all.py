"""
run_all.py -- reproduce every result, table and figure in the manuscript (v5), from scratch.

    python run_all.py              full run (about 30 minutes on two cores)
    python run_all.py --quick      smoke test with 10% of the replicates (results not for publication)

Steps (each script can also be run on its own from the repository root with PYTHONPATH=src):
  1  src/score_pairs.py       fit the Splink model, score candidate pairs   -> data/pairs_raw.parquet, results/model.json
  1b src/prelinkage.py        examination of records and candidate pairs     -> results/prelink_*, bands_and_strata, headroom
  1c src/model_check.py       fitted-model parameters and calibration        -> results/model_parameters.csv, figures/figS2
  1d src/ambiguity_check.py   what ambiguity levels capture under this model  -> results/ambiguity_*.csv, ambiguity_check.json
  1e src/person_level.py      pairwise against person-level quality (threshold 0.5) -> results/person_level*.{json,csv}
  2  src/estimator_check.py   design-weighted vs pooled estimators           -> results/estimator_check*.csv
  3  src/evaluate.py compare  all designs at 1,000 / 10,000 pairs, 5%, 10%   -> results/compare_*.csv
  4  src/evaluate.py sens     number of strata; reviewer error               -> results/sensitivity*.csv
  5  src/summarise.py         tables, figures and every number in the text   -> results/tables.md, results/facts.json, figures/
"""
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUICK = "--quick" in sys.argv
SCALE = 0.1 if QUICK else 1.0
R = lambda n: str(max(5, int(round(n * SCALE))))
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def step(*args):
    t = time.time()
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, env=ENV, check=True)
    print(f"   done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    step("src/score_pairs.py")
    step("src/prelinkage.py")
    step("src/model_check.py")
    step("src/ambiguity_check.py")
    step("src/person_level.py")
    step("src/estimator_check.py", R(50))
    for b in ("1000", "10000", "5pct", "10pct"):
        step("src/evaluate.py", "compare", b, R(300))
    step("src/evaluate.py", "sens", R(200))
    step("src/summarise.py")
    if QUICK:
        print("Quick run: replicate counts reduced; do not use these results.")
