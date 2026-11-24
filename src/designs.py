"""
designs.py -- sampling designs for clerical review (v5).

Each design is a set of strata plus an allocation rule.

Strata (all nested within score bands except the ambiguity-first set):
  B     score band (deciles of match probability)                       = current practice when proportional
  BA    score band x ambiguity level
  AF    ambiguity level, then deciles of match probability within level
  BP    score band x agreement pattern
  BPA   score band x agreement pattern x ambiguity level
Strata are capped at B/CAP_DIV: the largest are kept and the rest merged into residual strata defined by band
(and ambiguity level, where used).

Allocations:
  PROP-*  proportional: the same fraction of pairs in every stratum (at least FLOOR pairs per stratum)
  AF-eq   equal share of the budget per ambiguity level, proportional within level (AF strata)
  TM-BPA  target margins of error by band and ambiguity level, sample sizes from model probabilities (Eq. 6)
  OR-*    Neyman allocation with the TRUE stratum match rates (the Neyman ceiling; not realisable)
"""
import numpy as np, pandas as pd
import clerical_design as cd

FLOOR = 2
CAP_DIV = 10

DESIGNS = ["PROP-B", "PROP-BA", "PROP-AF", "AF-eq", "PROP-BP", "PROP-BPA", "TM-BPA",
           "OR-B", "OR-BA", "OR-AF", "OR-BP", "OR-BPA"]
LABELS = {"PROP-B": "Score band (current practice)", "PROP-BA": "Band × ambiguity",
          "PROP-AF": "Ambiguity, then match score", "AF-eq": "Ambiguity first, equal budget per level",
          "PROP-BP": "Band × pattern", "PROP-BPA": "Band × pattern × ambiguity",
          "TM-BPA": "Band × pattern × ambiguity, target margins",
          "OR-B": "Neyman ceiling: score band", "OR-BA": "Neyman ceiling: band × ambiguity",
          "OR-AF": "Neyman ceiling: ambiguity, then match score", "OR-BP": "Neyman ceiling: band × pattern",
          "OR-BPA": "Neyman ceiling: band × pattern × ambiguity"}


def waterfill(w, lo, hi, B):
    """Integer allocation n_s = clip(lam * w_s, lo_s, hi_s) with sum n_s = B (or closest feasible)."""
    w = np.asarray(w, float); hi = np.asarray(hi, float); lo = np.minimum(np.asarray(lo, float), hi)
    if lo.sum() >= B:
        return lo.astype(int)
    if hi.sum() <= B:
        return hi.astype(int)
    pos = w > 0
    top = np.max(hi[pos] / w[pos]) if pos.any() else 1.0
    a, b = 0.0, top * 2
    for _ in range(100):
        m = 0.5 * (a + b)
        if np.clip(m * w, lo, hi).sum() > B:
            b = m
        else:
            a = m
    n = np.clip(a * w, lo, hi)
    base = np.floor(n)
    rem = int(round(B - base.sum()))
    if rem > 0:
        frac = n - base
        cand = np.where(base < hi)[0]
        base[cand[np.argsort(-frac[cand], kind="stable")][:rem]] += 1
    return base.astype(int)


def within_level_deciles(pt, ngroups=10, min_size=1000):
    lev = pt.ambig_level; out = np.zeros(pt.N, dtype=np.int64); base = 0
    for a in np.unique(lev):
        idx = np.where(lev == a)[0]
        g = ngroups if len(idx) >= min_size else 1
        if g == 1:
            out[idx] = base
        else:
            r = pd.Series(pt.prob[idx]).rank(method="first")
            out[idx] = base + pd.qcut(r, g, labels=False).to_numpy()
        base += g
    return pd.factorize(out)[0]


def build_strata(pt, kind, B, cap_div=None):
    cap = max(int(B // (cap_div or CAP_DIV)), pt.nb)
    if kind == "B":
        return pt.band.astype(np.int64)
    if kind == "BA":
        return cd.make_strata(pt, [pt.band, pt.ambig_level], [pt.band], cap)
    if kind == "AF":
        return within_level_deciles(pt)
    if kind == "BP":
        return cd.make_strata(pt, [pt.band, pt.pattern], [pt.band], cap)
    if kind == "BPA":
        return cd.make_strata(pt, [pt.band, pt.pattern, pt.ambig_level], [pt.band, pt.ambig_level], cap)
    raise ValueError(kind)


def make_design(pt, name, B, cap_div=None) -> cd.Design:
    if name == "TM-BPA":
        d = cd.build_design(pt, "D2A", B)
        d.name = name
        return d
    kind = "AF" if name == "AF-eq" else name.split("-")[1]
    s = build_strata(pt, kind, B, cap_div)
    S = int(s.max()) + 1
    N_s = np.bincount(s, minlength=S)
    lo = np.minimum(FLOOR, N_s)
    if name.startswith("PROP-"):
        n = waterfill(N_s, lo, N_s, B)
    elif name.startswith("OR-"):
        p = np.bincount(s, weights=pt.y.astype(float), minlength=S) / N_s
        n = waterfill(N_s * np.sqrt(p * (1 - p)), lo, N_s, B)
    elif name == "AF-eq":
        lev_of_s = np.zeros(S, int); lev_of_s[s] = pt.ambig_level
        L = lev_of_s.max() + 1
        N_a = np.bincount(lev_of_s, weights=N_s, minlength=L)
        n_a = waterfill(np.ones(L), np.zeros(L), N_a, B)
        n = np.zeros(S, int)
        for a in range(L):
            m = lev_of_s == a
            n[m] = waterfill(N_s[m], lo[m], N_s[m], int(n_a[a]))
    else:
        raise ValueError(name)
    return cd.Design(name, s, S, N_s, n)
