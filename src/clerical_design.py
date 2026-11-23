"""
clerical_design.py
==================
Corrected, data-agnostic implementation of the ambiguity-aware clerical review
sampling framework, plus the ladder of simpler designs used for comparison.

Replaces the estimation / allocation logic in 3_sample_size.py, 4_budget_sample.py
and 5_draw_sample.py. Fixes relative to the earlier code:

  1. Stratified (design-weighted) estimators: p_hat_d = sum_cells N_cell p_hat_cell / N_d.
     The earlier code pooled sampled pairs within a band (X_b / n_b), which is
     biased whenever sampling fractions differ between strata.
  2. Pairs with missing gender are kept (category "missing") rather than silently
     dropped by groupby(dropna=True).
  3. Record-level matchability / perplexity use BOTH directions of each pair, so a
     record that only ever appears as unique_id_r is no longer given default values.
     Pair ambiguity = max over its two records.
  4. Budget is a number of pairs B (not a percentage); strata are collapsed when
     there are too many to review at least a few pairs each.
  5. Signed error, empirical coverage, and many replicates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Z = 1.96
BASE_MOE_10 = np.array([0.07, 0.07, 0.06, 0.06, 0.05, 0.05, 0.04, 0.035, 0.03, 0.03])


# ----------------------------------------------------------------------
# Record-level ambiguity (symmetric)
# ----------------------------------------------------------------------
def record_metrics_symmetric(pairs: pd.DataFrame, weight_col="match_weight",
                             id_l="unique_id_l", id_r="unique_id_r") -> pd.DataFrame:
    """Matchability and conditional perplexity per record using both pair directions.

    For record r_i with candidate set C_i, odds O_ij = 2**w_ij and the NIL option
    has odds 1.  m_i = 1 - 1/(1+S_i), S_i = sum_j O_ij;  p*_ij = O_ij/S_i;
    H_i = -sum p* log p*;  PP_i = exp(H_i)  (range 1..|C_i|).
    """
    w = pairs[weight_col].to_numpy(float)
    ln_o = w * math.log(2.0)
    o = np.exp(np.clip(ln_o, -700, 700))
    a = np.concatenate([pairs[id_l].to_numpy(), pairs[id_r].to_numpy()])
    oo = np.concatenate([o, o])
    olo = np.concatenate([o * ln_o, o * ln_o])
    n_c = np.ones(len(a))
    g = pd.DataFrame({"rid": a, "o": oo, "olo": olo, "n": n_c}).groupby("rid").sum()
    S = g["o"].to_numpy()
    H = np.log(S) - g["olo"].to_numpy() / S
    H = np.maximum(H, 0.0)
    out = pd.DataFrame({
        "rid": g.index,
        "matchability": 1.0 - 1.0 / (1.0 + S),
        "perp_cond": np.exp(H),
        "n_candidates": g["n"].to_numpy(),
    }).set_index("rid")
    return out


def rescale_pp(pp):
    """PP in [1, inf) -> [0, 1): 1 - 1/PP (0 = one dominant candidate)."""
    return 1.0 - 1.0 / np.asarray(pp, float)


# ----------------------------------------------------------------------
# Ambiguity levels
# ----------------------------------------------------------------------
def levels_thresholds(rec: pd.DataFrame, m_min=0.05, cuts=(1.5, 3.0)) -> pd.Series:
    """Pre-specified ordinal levels.
    0: m < m_min (effectively no plausible match)
    1: PP < cuts[0]; 2: cuts[0] <= PP < cuts[1]; ... ; L-1: PP >= cuts[-1]
    """
    lev = 1 + np.searchsorted(np.asarray(cuts), rec["perp_cond"].to_numpy(), side="right")
    lev = np.where(rec["matchability"].to_numpy() < m_min, 0, lev)
    return pd.Series(lev.astype(int), index=rec.index, name="ambig_level")


def levels_kmeans(rec: pd.DataFrame, k_range=(2, 10), rescale=True, random_state=42,
                  sil_sample=5000) -> Tuple[pd.Series, int]:
    """k-means on (matchability, PP~), K by silhouette (sampled), clusters ordered by
    mean PP~ (ascending ambiguity). Used only in sensitivity analysis."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    pp = rescale_pp(rec["perp_cond"]) if rescale else np.log1p(np.minimum(rec["perp_cond"], 50))
    X = np.column_stack([rec["matchability"].to_numpy(), pp])
    best = (-np.inf, None, None)
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=2).fit(X)
        sc = silhouette_score(X, km.labels_, sample_size=min(sil_sample, len(X)),
                              random_state=random_state)
        if sc > best[0]:
            best = (sc, k, km.labels_)
    _, k, lab = best
    order = pd.Series(pp).groupby(lab).mean().sort_values().index.tolist()
    rank = {c: r for r, c in enumerate(order)}
    return pd.Series([rank[c] for c in lab], index=rec.index, name="ambig_level"), k


def levels_kmeans_fixed(rec, k, rescale=True, random_state=42):
    from sklearn.cluster import KMeans
    pp = rescale_pp(rec["perp_cond"]) if rescale else np.log1p(np.minimum(rec["perp_cond"], 50))
    X = np.column_stack([rec["matchability"].to_numpy(), pp])
    lab = KMeans(n_clusters=k, random_state=random_state, n_init=5).fit_predict(X)
    order = pd.Series(pp).groupby(lab).mean().sort_values().index.tolist()
    rank = {c: r for r, c in enumerate(order)}
    return pd.Series([rank[c] for c in lab], index=rec.index, name="ambig_level")


# ----------------------------------------------------------------------
# Pair table
# ----------------------------------------------------------------------
@dataclass
class PairTable:
    N: int
    prob: np.ndarray
    y: np.ndarray                 # true match 0/1
    pattern: np.ndarray           # int codes
    pattern_str: np.ndarray
    gender: np.ndarray            # int codes 0 male, 1 female, 2 other/missing
    ambig_level: np.ndarray       # int
    n_levels: int
    band: np.ndarray              # int 0..nb-1
    nb: int
    band_edges: np.ndarray = field(default=None)


def make_bands(prob: np.ndarray, nb: int = 10) -> np.ndarray:
    return pd.qcut(prob, q=nb, labels=False, duplicates="drop").astype(int)


def pair_levels_quantile(raw: pd.DataFrame, rec: pd.DataFrame, n_groups: int = 3, m_min: float = 0.05,
                         id_l="unique_id_l", id_r="unique_id_r") -> np.ndarray:
    """Data-adaptive pair-level ambiguity levels (no truth used).
    score_ij = max(PP~_i, PP~_j), PP~ = 1 - 1/PP.  Level 0: both records have matchability < m_min;
    remaining pairs split into n_groups equal-sized groups by score (levels 1..n_groups)."""
    ppt = pd.Series(rescale_pp(rec["perp_cond"].to_numpy()), index=rec.index)
    mm = rec["matchability"]
    s_l, s_r = ppt.loc[raw[id_l]].to_numpy(), ppt.loc[raw[id_r]].to_numpy()
    m_max = np.maximum(mm.loc[raw[id_l]].to_numpy(), mm.loc[raw[id_r]].to_numpy())
    score = np.maximum(s_l, s_r)
    ok = m_max >= m_min
    lev = np.zeros(len(raw), dtype=int)
    cuts = np.quantile(score[ok], np.linspace(0, 1, n_groups + 1)[1:-1])
    lev[ok] = 1 + np.searchsorted(cuts, score[ok], side="right")
    return lev


def build_pair_table(raw: pd.DataFrame, levels: Optional[pd.Series], nb: int = 10,
                     pair_levels: Optional[np.ndarray] = None) -> PairTable:
    gamma_cols = sorted(c for c in raw.columns if c.startswith("gamma_"))
    pat = raw[gamma_cols].astype(int).astype(str).agg("".join, axis=1).to_numpy()
    pat_codes, _ = pd.factorize(pat)
    g = raw["gender_l"].fillna("missing").map({"male": 0, "female": 1}).fillna(2).astype(int).to_numpy()
    if pair_levels is not None:
        lev = np.asarray(pair_levels).astype(int)
    else:
        lv_l = raw["unique_id_l"].map(levels).to_numpy()
        lv_r = raw["unique_id_r"].map(levels).to_numpy()
        lev = np.maximum(lv_l, lv_r).astype(int)
    prob = raw["match_probability"].to_numpy(float)
    band = make_bands(prob, nb)
    return PairTable(
        N=len(raw), prob=prob,
        y=(raw["cluster_l"].to_numpy() == raw["cluster_r"].to_numpy()).astype(np.int8),
        pattern=pat_codes, pattern_str=pat, gender=g, ambig_level=lev,
        n_levels=int(lev.max()) + 1, band=band, nb=int(band.max()) + 1)


# ----------------------------------------------------------------------
# Sample size and allocation
# ----------------------------------------------------------------------
def hypergeo_n(N: np.ndarray, p0: np.ndarray, moe: np.ndarray, z=Z) -> np.ndarray:
    pq = z * z * p0 * (1 - p0)
    n = pq * N / (pq + moe ** 2 * (N - 1))
    return np.minimum(np.ceil(n - 1e-9), N)


@dataclass
class Design:
    name: str
    s: np.ndarray            # stratum id per pair (0..S-1)
    S: int
    N_s: np.ndarray
    n_s: np.ndarray
    scale: float = float("nan")
    meta: dict = field(default_factory=dict)


def _factorize_keys(cols: Sequence[np.ndarray]) -> np.ndarray:
    key = np.zeros(len(cols[0]), dtype=np.int64)
    for c in cols:
        key = key * (int(c.max()) + 1) + c.astype(np.int64)
    return pd.factorize(key)[0]


def make_strata(pt: PairTable, key_cols: List[np.ndarray], resid_cols: List[np.ndarray],
                max_strata: Optional[int]) -> np.ndarray:
    """Cross-classify key_cols; if more than max_strata, keep the largest and merge the
    rest into residual strata keyed by resid_cols."""
    s = _factorize_keys(key_cols)
    S = s.max() + 1
    if max_strata is None or S <= max_strata:
        return s
    size = np.bincount(s, minlength=S)
    r = _factorize_keys(resid_cols)
    R = r.max() + 1
    keep_n = max(max_strata - R, 0)
    keep = np.argsort(-size, kind="stable")[:keep_n]
    is_keep = np.zeros(S, bool)
    is_keep[keep] = True
    new = np.where(is_keep[s], s, S + r)
    return pd.factorize(new)[0]


def stratum_p0(pt: PairTable, s: np.ndarray, S: int, p0_mode="model", clip=(0.05, 0.95)):
    cnt = np.bincount(s, minlength=S).astype(float)
    if p0_mode == "model":
        p = np.bincount(s, weights=pt.prob, minlength=S) / cnt
    elif p0_mode == "oracle":
        p = np.bincount(s, weights=pt.y.astype(float), minlength=S) / cnt
    elif p0_mode == "half":
        p = np.full(S, 0.5)
    else:
        raise ValueError(p0_mode)
    return np.clip(p, *clip)


def allocate_budget(N_s, p0, moe0, B, n_floor=2):
    """Find the global scale c so that sum_s n_s(c) <= B is as close to B as possible.
    moe_s(c) = c * moe0_s.  Returns (n_s, c)."""
    N_s = N_s.astype(float)
    floor = np.minimum(n_floor, N_s)
    if B >= N_s.sum():
        return N_s.astype(int), 0.0

    def tot(c):
        n = np.maximum(hypergeo_n(N_s, p0, c * moe0), floor)
        return n.sum(), n

    lo, hi = 1e-4, 1e4
    if tot(hi)[0] > B:
        raise ValueError(f"Budget {B} below minimum feasible {tot(hi)[0]:.0f} for {len(N_s)} strata")
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if tot(mid)[0] > B:
            lo = mid
        else:
            hi = mid
    tt, n = tot(hi)
    return n.astype(int), hi


def band_moe(nb: int, profile=None) -> np.ndarray:
    prof = BASE_MOE_10 if profile is None else np.asarray(profile, float)
    if nb == len(prof):
        return prof.copy()
    return np.interp(np.linspace(0, 1, nb), np.linspace(0, 1, len(prof)), prof)


def ambig_factors(L: int, fmax=1.6, fmin=0.7) -> np.ndarray:
    return np.linspace(fmax, fmin, L) if L > 1 else np.array([1.0])


def build_design(pt: PairTable, kind: str, B: Optional[int], *, profile=None,
                 fmax=1.6, fmin=0.7, p0_mode="model", p0_clip=(0.05, 0.95),
                 n_floor=2, min_per_stratum=10, natural=False) -> Design:
    """kind in {D0,D1,D2,D3,D4}; B = number of pairs (None with natural=True gives the
    uncollapsed design at c=1)."""
    N = pt.N
    if kind == "D0":
        s = np.zeros(N, dtype=np.int64)
        return Design("D0", s, 1, np.array([N]), np.array([min(B, N)]))
    if kind == "D1":
        s = pt.band.astype(np.int64)
        N_s = np.bincount(s, minlength=pt.nb)
        n = np.zeros(pt.nb)
        rem, active = float(B), np.ones(pt.nb, bool)
        while rem > 1e-9 and active.any():          # water-filling with cap at N_b
            share = rem / active.sum()
            add = np.minimum(share, N_s - n) * active
            n += add
            rem -= add.sum()
            active = active & (n < N_s - 1e-9)
            if add.sum() < 1e-9:
                break
        n = np.floor(n).astype(int)
        return Design("D1", s, pt.nb, N_s, n)

    # D2-D4
    cols = {"D2": [pt.band, pt.pattern],
            "D3": [pt.band, pt.pattern, pt.gender],
            "D4": [pt.band, pt.pattern, pt.ambig_level, pt.gender],
            "D2A": [pt.band, pt.pattern, pt.ambig_level]}[kind]
    resid = {"D2": [pt.band], "D3": [pt.band], "D4": [pt.band, pt.ambig_level], "D2A": [pt.band, pt.ambig_level]}[kind]
    max_strata = None if natural else max(int(B // min_per_stratum), pt.nb)
    s = make_strata(pt, cols, resid, max_strata)
    S = s.max() + 1
    N_s = np.bincount(s, minlength=S)
    p0 = stratum_p0(pt, s, S, p0_mode, p0_clip)
    bm = band_moe(pt.nb, profile)
    # band of each stratum (strata never straddle bands)
    sb = np.zeros(S, dtype=int); sb[s] = pt.band
    moe0 = bm[sb]
    if kind in ("D4", "D2A"):
        sa = np.zeros(S, dtype=int); sa[s] = pt.ambig_level
        moe0 = moe0 * ambig_factors(pt.n_levels, fmax, fmin)[sa]
    if natural:
        n = np.maximum(hypergeo_n(N_s.astype(float), p0, moe0), np.minimum(n_floor, N_s)).astype(int)
        c = 1.0
    else:
        n, c = allocate_budget(N_s, p0, moe0, B, n_floor)
    return Design(kind, s, S, N_s, n, c, {"S": S})


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------
def draw_sample(d: Design, rng: np.random.Generator) -> np.ndarray:
    """Simple random sampling without replacement within each stratum; returns indices."""
    N = len(d.s)
    key = d.s + rng.random(N) * 0.999999
    order = np.argsort(key, kind="stable")
    s_sorted = d.s[order]
    start = np.concatenate([[0], np.cumsum(d.N_s)[:-1]])
    pos = np.arange(N) - start[s_sorted]
    return order[pos < d.n_s[s_sorted]]


# ----------------------------------------------------------------------
# Estimation (stratified, with post-stratification for domains the design does not nest)
# ----------------------------------------------------------------------
class Estimator:
    """Pre-computes population cell counts for each domain variable."""

    def __init__(self, d: Design, pt: PairTable, domains: Dict[str, np.ndarray]):
        self.d, self.pt = d, pt
        self.domains = domains
        self.D = {k: int(v.max()) + 1 for k, v in domains.items()}
        self.cell = {k: d.s * self.D[k] + v for k, v in domains.items()}
        self.Ncell = {k: np.bincount(self.cell[k], minlength=d.S * self.D[k]).astype(float)
                      for k in domains}
        self.Nd = {k: np.bincount(v, minlength=self.D[k]).astype(float) for k, v in domains.items()}
        self.truth = {k: np.bincount(v, weights=pt.y.astype(float), minlength=self.D[k]) / self.Nd[k]
                      for k, v in domains.items()}

    def estimate(self, idx: np.ndarray, y_obs: np.ndarray):
        """y_obs aligned with idx. Returns {domain: (est, var)}."""
        d = self.d
        n_s = np.bincount(d.s[idx], minlength=d.S).astype(float)
        x_s = np.bincount(d.s[idx], weights=y_obs, minlength=d.S)
        p_s = np.where(n_s > 0, x_s / np.maximum(n_s, 1), 0.0)
        out = {}
        for k, dom in self.domains.items():
            D = self.D[k]
            cell = self.cell[k][idx]
            n_c = np.bincount(cell, minlength=d.S * D).astype(float)
            x_c = np.bincount(cell, weights=y_obs, minlength=d.S * D)
            Nc = self.Ncell[k]
            fb = np.repeat(p_s, D)                      # stratum-level fallback
            p_c = np.where(n_c > 0, x_c / np.maximum(n_c, 1), fb)
            # variance contribution
            n_eff = np.where(n_c > 0, n_c, np.repeat(n_s, D))
            N_eff = np.where(n_c > 0, Nc, np.repeat(d.N_s.astype(float), D))
            # unbiased within-cell variance s^2 = p(1-p) n/(n-1); cells with n=1 or none borrow the stratum's s^2
            x_e = np.where(n_c > 0, x_c, np.repeat(x_s, D))
            p_e = x_e / np.maximum(n_eff, 1)
            s2_s = np.where(n_s > 1, (x_s / np.maximum(n_s, 1)) * (1 - x_s / np.maximum(n_s, 1)) * n_s / np.maximum(n_s - 1, 1), 0.0)
            s2 = np.where(n_eff > 1, p_e * (1 - p_e) * n_eff / np.maximum(n_eff - 1, 1), np.repeat(s2_s, D))
            fpc = np.clip(1 - n_eff / np.maximum(N_eff, 1), 0, 1)
            v_c = (Nc ** 2) * fpc * s2 / np.maximum(n_eff, 1)
            dom_of_cell = np.arange(d.S * D) % D
            est = np.bincount(dom_of_cell, weights=Nc * p_c, minlength=D) / self.Nd[k]
            var = np.bincount(dom_of_cell, weights=v_c, minlength=D) / self.Nd[k] ** 2
            out[k] = (est, var)
        return out

    def pooled_band(self, idx, y_obs):
        """The earlier (biased under unequal sampling fractions) pooled band estimator."""
        band = self.domains["band"]
        D = self.D["band"]
        n_b = np.bincount(band[idx], minlength=D).astype(float)
        x_b = np.bincount(band[idx], weights=y_obs, minlength=D)
        p_b = np.where(n_b > 0, x_b / np.maximum(n_b, 1), 0.0)
        glob = (self.Nd["band"] * p_b).sum() / self.Nd["band"].sum()
        return glob, p_b


# ----------------------------------------------------------------------
# Representativeness
# ----------------------------------------------------------------------
class Representativeness:
    """L1 distance (as % mass to move, 0-100) between population and sample distributions of a
    categorical variable within each band, averaged across bands (unweighted)."""

    def __init__(self, pt: PairTable, cats: Dict[str, np.ndarray]):
        self.pt, self.cats = pt, cats
        self.K = {k: int(v.max()) + 1 for k, v in cats.items()}
        self.code = {k: pt.band * self.K[k] + v for k, v in cats.items()}
        self.pop = {}
        for k in cats:
            c = np.bincount(self.code[k], minlength=pt.nb * self.K[k]).reshape(pt.nb, self.K[k]).astype(float)
            self.pop[k] = c / c.sum(1, keepdims=True)

    def l1(self, idx):
        out = {}
        for k in self.cats:
            c = np.bincount(self.code[k][idx], minlength=self.pt.nb * self.K[k]).reshape(self.pt.nb, self.K[k]).astype(float)
            tot = c.sum(1, keepdims=True)
            samp = np.where(tot > 0, c / np.maximum(tot, 1), np.nan)
            out[k] = 50.0 * np.nansum(np.abs(samp - self.pop[k]), axis=1)
        return out
