"""common.py -- paths and the shared loader for the scored candidate-pair table."""
from pathlib import Path
import pandas as pd
import clerical_design as cd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for p in (DATA, RESULTS, FIGURES):
    p.mkdir(exist_ok=True)

PAIRS = DATA / "pairs_raw.parquet"
RECORDS = DATA / "record_metrics.parquet"
BUDGET_LABELS = ("1000", "10000", "5pct", "10pct")
REPS_COMPARE, REPS_SENS, REPS_ESTCHECK = 300, 200, 50

_CACHE = {}


def load(levels="quantile", nb=10, **kw):
    """Return (raw pair table, record-level ambiguity metrics, PairTable).

    levels: "quantile" (default: pair-level tertiles of ambiguity plus a no-plausible-match level),
            "thresholds" (fixed cut-points on perplexity), or "none".
    """
    if "raw" not in _CACHE:
        if not PAIRS.exists():
            raise FileNotFoundError(f"{PAIRS} not found: run src/score_pairs.py first")
        _CACHE["raw"] = pd.read_parquet(PAIRS)
    raw = _CACHE["raw"]
    if RECORDS.exists() and RECORDS.stat().st_mtime >= PAIRS.stat().st_mtime:   # cache only if newer than the scored pairs
        rec = pd.read_parquet(RECORDS)
    else:
        rec = cd.record_metrics_symmetric(raw)
        rec.to_parquet(RECORDS)
    pair_levels = None
    if levels == "quantile":
        pair_levels = cd.pair_levels_quantile(raw, rec, n_groups=kw.get("n_groups", 3))
        lv = pd.Series(0, index=rec.index)
    elif levels == "thresholds":
        lv = cd.levels_thresholds(rec, m_min=kw.get("m_min", 0.05), cuts=kw.get("cuts", (1.5, 3.0)))
    elif levels == "none":
        lv = pd.Series(0, index=rec.index)
    else:
        raise ValueError(levels)
    pt = cd.build_pair_table(raw, lv, nb=nb, pair_levels=pair_levels)
    return raw, rec, pt


def budget_of(pt, label):
    return {"1000": 1000, "10000": 10000, "5pct": int(round(0.05 * pt.N)), "10pct": int(round(0.10 * pt.N))}[label]
