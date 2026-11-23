"""
estimator_check.py -- Step 3. Show why estimates must be design-weighted.

Draws samples from uncollapsed stratified designs (every stratum kept, margin-based sample sizes) and compares
  - the design-weighted (stratified) estimator: sum_s N_s p_s / N
  - the pooled estimator: sampled pairs pooled within each score band, p_b = X_b / n_b, weighted by band size
The pooled estimator is what the earlier version of this work used. It is biased when strata within a band are
sampled at different rates.
Output: results/estimator_check.csv, results/estimator_check_by_band.csv
"""
import sys, zlib
import numpy as np, pandas as pd
import clerical_design as cd
from common import load, RESULTS


def main(R):
    raw, rec, pt = load("quantile")
    dom = {"global": np.zeros(pt.N, dtype=int), "band": pt.band}
    rows, brows = [], []
    for kind in ("D2", "D2A"):
        d = cd.build_design(pt, kind, None, natural=True)
        est = cd.Estimator(d, pt, dom)
        rng = np.random.default_rng(zlib.crc32(f"estcheck|{kind}".encode()))
        g, gp, b, bp = [], [], [], []
        for r in range(R):
            idx = cd.draw_sample(d, rng)
            y = pt.y[idx].astype(float)
            e = est.estimate(idx, y)
            pg, pb = est.pooled_band(idx, y)
            g.append(e["global"][0][0]); gp.append(pg); b.append(e["band"][0]); bp.append(pb)
        tg, tb = est.truth["global"][0], est.truth["band"]
        g, gp, b, bp = map(np.asarray, (g, gp, b, bp))
        rows.append(dict(design=kind, strata=d.S, sample=int(d.n_s.sum()), sample_share=d.n_s.sum() / pt.N, reps=R,
                         weighted_bias_pp=100 * (g.mean() - tg), weighted_sd_pp=100 * g.std(ddof=1),
                         pooled_bias_pp=100 * (gp.mean() - tg), pooled_sd_pp=100 * gp.std(ddof=1)))
        for k in range(pt.nb):
            brows.append(dict(design=kind, band=k + 1, weighted_bias_pp=100 * (b[:, k].mean() - tb[k]),
                              pooled_bias_pp=100 * (bp[:, k].mean() - tb[k])))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "estimator_check.csv", index=False)
    pd.DataFrame(brows).to_csv(RESULTS / "estimator_check_by_band.csv", index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
