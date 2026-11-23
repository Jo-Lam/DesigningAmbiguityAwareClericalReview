"""
ambiguity_check.py -- Step 1d. Does record-level ambiguity carry information about match status, or about the
linkage model's errors, once the model is correctly fitted?

For each ambiguity level (overall and within score bands) it reports:
  - pairs, true match rate and mean model probability (calibration gap = true rate - mean probability)
  - among accepted links (p >= 0.5): share of all accepted links, false-match rate (1 - precision)
  - among rejected pairs: share of all true matches that are missed
and, for each band, the within-band variance in match status explained by ambiguity levels.
It also re-computes ambiguity excluding each record's true duplicates (uses the truth; diagnostic only), to
separate "many records of one person" from "confusable with other people".

Outputs: results/ambiguity_by_level.csv, results/ambiguity_by_band_level.csv, results/ambiguity_check.json
"""
import json, math
import numpy as np, pandas as pd
import clerical_design as cd
from common import load, RESULTS


def main():
    raw, rec, pt = load("quantile")
    acc = pt.prob >= 0.5
    d = pd.DataFrame({"band": pt.band + 1, "a": pt.ambig_level, "y": pt.y, "p": pt.prob, "acc": acc})
    tot_acc, tot_match = acc.sum(), pt.y.sum()

    def summ(g):
        a_ = g[g.acc]; r_ = g[~g.acc]
        return pd.Series({"pairs": len(g), "true_match_rate": g.y.mean(), "mean_probability": g.p.mean(),
                          "calibration_gap": g.y.mean() - g.p.mean(),
                          "share_of_accepted": len(a_) / tot_acc,
                          "false_match_rate_accepted": 1 - a_.y.mean() if len(a_) else np.nan,
                          "share_of_false_matches": ((a_.y == 0).sum() / max((d.acc & (d.y == 0)).sum(), 1)),
                          "share_of_missed_matches": r_.y.sum() / max((~d.acc & (d.y == 1)).sum(), 1)})
    lv = d.groupby("a").apply(summ, include_groups=False)
    lv.to_csv(RESULTS / "ambiguity_by_level.csv")
    bl = d.groupby(["band", "a"]).apply(summ, include_groups=False)
    bl.to_csv(RESULTS / "ambiguity_by_band_level.csv")
    # within-band variance explained by ambiguity, band by band
    vb = d.groupby("band").y.agg(lambda s: len(s) * s.mean() * (1 - s.mean()))
    vba = d.groupby(["band", "a"]).y.agg(lambda s: len(s) * s.mean() * (1 - s.mean())).groupby(level=0).sum()
    expl = (1 - vba / vb).fillna(0)
    # ambiguity excluding true duplicates (truth-based diagnostic): perplexity over candidates from OTHER people
    other = raw[raw.cluster_l != raw.cluster_r]
    rec_o = cd.record_metrics_symmetric(other)
    ppo = pd.Series(cd.rescale_pp(rec_o.perp_cond.to_numpy()), index=rec_o.index)
    score_o = np.maximum(ppo.reindex(raw.unique_id_l).fillna(0).to_numpy(), ppo.reindex(raw.unique_id_r).fillna(0).to_numpy())
    cuts = np.quantile(score_o, [1 / 3, 2 / 3])
    lev_o = 1 + np.searchsorted(cuts, score_o, side="right")
    d["a_other"] = lev_o
    vbo = d.groupby(["band", "a_other"]).y.agg(lambda s: len(s) * s.mean() * (1 - s.mean())).groupby(level=0).sum()
    out = dict(within_band_variance_explained_ambiguity=float(1 - vba.sum() / vb.sum()),
               within_band_variance_explained_ambiguity_other_people=float(1 - vbo.sum() / vb.sum()),
               explained_by_band={int(k): float(v) for k, v in expl.items()},
               spearman_ambiguity_score_vs_other_people_score=float(pd.Series(score_o).corr(
                   pd.Series(np.maximum(cd.rescale_pp(rec.perp_cond.reindex(raw.unique_id_l).to_numpy()),
                                        cd.rescale_pp(rec.perp_cond.reindex(raw.unique_id_r).to_numpy()))), method="spearman")))
    other_lv = d.groupby("a_other").apply(summ, include_groups=False)
    other_lv.to_csv(RESULTS / "ambiguity_other_people_by_level.csv")
    json.dump(out, open(RESULTS / "ambiguity_check.json", "w"), indent=1)
    print(lv.round(4).to_string()); print(other_lv.round(4).to_string()); print(json.dumps(out, indent=1))
    print(bl.loc[[5, 6, 7]].round(3).to_string())


if __name__ == "__main__":
    main()
