"""
evaluate.py -- Steps 4 and 5. Simulate clerical review under each design and score the estimates.

The population of candidate pairs (scores, strata, true status) is fixed. Each replicate draws a new stratified
random sample, takes reviewers' decisions as the truth (or the truth with random errors, in the sensitivity
analysis) and estimates, with design weights:
  - the match rate among all candidate pairs, by score band, and among the most ambiguous pairs (level 3)
  - precision and recall of the decision "accept if match probability >= t", t = 0.5 and 0.9
    (recall among candidate pairs: true pairs that blocking never generated are outside the frame).
Within each stratum, domains with known counts (band, ambiguity level, accepted/rejected) are post-stratified.
Recall = A / (A + C), A and C the estimated true matches among accepted and rejected pairs (delta-method variance).

Each (design, budget) cell uses its own seed derived from its label, so cells are independent and reproducible.

Usage:
  python src/evaluate.py compare <1000|10000|5pct|10pct> [reps]   all designs at one budget
  python src/evaluate.py sens [reps]                             number of strata; reviewer error
"""
import sys, time, zlib
import numpy as np, pandas as pd
import clerical_design as cd
import designs as dz
from common import load, budget_of, RESULTS

Z = 1.96
THRESH = (0.5, 0.9)


def domains(pt):
    d = {"global": np.zeros(pt.N, dtype=int), "band": pt.band, "ambig": pt.ambig_level}
    for t in THRESH:
        d[f"acc{t}"] = (pt.prob >= t).astype(int)
    return d


def evaluate(pt, d, B, R, seed, flip=0.0, flip_mult=1.0):
    est = cd.Estimator(d, pt, domains(pt))
    tr = est.truth
    top = pt.n_levels - 1
    yt = pt.y.astype(float)
    acc = {t: pt.prob >= t for t in THRESH}
    N_acc = {t: int(acc[t].sum()) for t in THRESH}
    true_P = {t: yt[acc[t]].mean() for t in THRESH}
    true_R = {t: yt[acc[t]].sum() / yt.sum() for t in THRESH}
    q = flip * np.where(pt.ambig_level == top, flip_mult, 1.0)
    rng = np.random.default_rng(seed)
    rows, brows, alloc = [], [], np.zeros(pt.nb)
    for r in range(R):
        y_obs = np.where(rng.random(pt.N) < q, 1 - pt.y, pt.y).astype(np.int8) if flip > 0 else pt.y
        idx = cd.draw_sample(d, rng)
        y = y_obs[idx].astype(float)
        e = est.estimate(idx, y)
        g, gv = e["global"][0][0], e["global"][1][0]
        b, bv = e["band"]
        a, av = e["ambig"]
        row = dict(rep=r, n=len(idx),
                   mr_err=100 * (g - tr["global"][0]), mr_cov=abs(g - tr["global"][0]) <= Z * np.sqrt(gv),
                   mr_ciw=200 * Z * np.sqrt(gv),
                   hi_err=100 * (a[top] - tr["ambig"][top]), hi_cov=abs(a[top] - tr["ambig"][top]) <= Z * np.sqrt(av[top]))
        for t in THRESH:
            r_est, r_var = e[f"acc{t}"]
            P, VP = r_est[1], r_var[1]
            A, VA = N_acc[t] * P, N_acc[t] ** 2 * VP
            Nrej = pt.N - N_acc[t]
            C, VC = Nrej * r_est[0], Nrej ** 2 * r_var[0]
            Rc = A / (A + C)
            VR = (C ** 2 * VA + A ** 2 * VC) / (A + C) ** 4
            row[f"P{t}_err"] = 100 * (P - true_P[t]); row[f"P{t}_cov"] = abs(P - true_P[t]) <= Z * np.sqrt(VP)
            row[f"R{t}_err"] = 100 * (Rc - true_R[t]); row[f"R{t}_cov"] = abs(Rc - true_R[t]) <= Z * np.sqrt(VR)
            row[f"P{t}_ciw"] = 200 * Z * np.sqrt(VP); row[f"R{t}_ciw"] = 200 * Z * np.sqrt(VR)
        rows.append(row)
        alloc += np.bincount(pt.band[idx], minlength=pt.nb)
        for k in range(pt.nb):
            brows.append(dict(band=k + 1, err=100 * (b[k] - tr["band"][k]),
                              cov=abs(b[k] - tr["band"][k]) <= Z * np.sqrt(bv[k])))
    df, bdf = pd.DataFrame(rows), pd.DataFrame(brows)
    rm = lambda c: float(np.sqrt((df[c] ** 2).mean()))
    s = dict(design=d.name, budget=B, strata=d.S, reps=R, n_mean=df.n.mean(),
             mr_true=100 * tr["global"][0], mr_bias=df.mr_err.mean(), mr_rmse=rm("mr_err"),
             mr_cov=df.mr_cov.mean(), mr_ciw=df.mr_ciw.mean(),
             hi_true=100 * tr["ambig"][top], hi_bias=df.hi_err.mean(), hi_rmse=rm("hi_err"), hi_cov=df.hi_cov.mean())
    for t in THRESH:
        s[f"N_acc{t}"] = N_acc[t]; s[f"P{t}_true"] = 100 * true_P[t]; s[f"R{t}_true"] = 100 * true_R[t]
        for m in ("P", "R"):
            s[f"{m}{t}_bias"] = df[f"{m}{t}_err"].mean(); s[f"{m}{t}_rmse"] = rm(f"{m}{t}_err")
            s[f"{m}{t}_cov"] = df[f"{m}{t}_cov"].mean(); s[f"{m}{t}_ciw"] = df[f"{m}{t}_ciw"].mean()
    s.update({f"alloc_b{i + 1}": v for i, v in enumerate(alloc / R)})
    bdf["abs"] = bdf.err.abs()
    bb = bdf.groupby("band").agg(mae=("abs", "mean"), bias=("err", "mean"), cov=("cov", "mean")).reset_index()
    bb["design"] = d.name; bb["budget"] = B
    return s, bb


def run(configs, R, fname):
    raw, rec, pt = load("quantile")
    rows, brows = [], []
    for (group, label, design, blab, knobs) in configs:
        B = budget_of(pt, blab); t0 = time.time()
        d = dz.make_design(pt, design, B, knobs.get("cap_div"))
        seed = zlib.crc32(f"{group}|{label}|{design}|{blab}".encode()) % (2 ** 31)
        s, bb = evaluate(pt, d, B, R, seed, knobs.get("flip", 0.0), knobs.get("mult", 1.0))
        s.update(group=group, config=label, budget_label=blab, label=dz.LABELS[design])
        bb["budget_label"] = blab; bb["group"] = group; bb["config"] = label
        rows.append(s); brows.append(bb)
        pd.DataFrame(rows).to_csv(RESULTS / f"{fname}.csv", index=False)
        pd.concat(brows).to_csv(RESULTS / f"{fname}_by_band.csv", index=False)
        print(group, label, design, blab, f"S={s['strata']} mr={s['mr_rmse']:.3f} P.5={s['P0.5_rmse']:.3f} "
              f"R.5={s['R0.5_rmse']:.3f} hi={s['hi_rmse']:.3f} {time.time() - t0:.0f}s", flush=True)


def sens_configs():
    cfg = []
    for b in ("1000", "5pct"):
        for d in ("PROP-BP", "PROP-BPA"):
            for c in (5, 30):
                cfg.append(("strata cap", f"max strata B/{c}", d, b, {"cap_div": c}))
    for b in ("1000", "10000", "5pct"):
        for d in ("PROP-B", "PROP-BP", "PROP-BPA"):
            cfg.append(("reviewer error", "none", d, b, {}))
            cfg.append(("reviewer error", "5% uniform", d, b, {"flip": 0.05}))
            cfg.append(("reviewer error", "5%, x3 in most ambiguous", d, b, {"flip": 0.05, "mult": 3.0}))
    return cfg


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "compare":
        lab = sys.argv[2]; R = int(sys.argv[3]) if len(sys.argv) > 3 else 300
        run([("compare", "default", d, lab, {}) for d in dz.DESIGNS], R, f"compare_{lab}")
    elif stage == "sens":
        R = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        run(sens_configs(), R, "sensitivity")
    else:
        raise SystemExit(__doc__)
