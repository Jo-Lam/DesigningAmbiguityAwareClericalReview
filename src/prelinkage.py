"""
prelinkage.py -- Step 2. Examine the data and candidate pairs before designing the review.

Outputs (results/):
  prelink_signature_ambiguity.csv  exact-signature ambiguity of identifier combinations (as in
                                   Jo-Lam/splink_prelinkage combination_ambiguity_data): share of complete
                                   records whose identifier values are shared with at least one other record
  prelink_cluster_sizes.csv        records per true person (cluster)
  prelink_missingness.csv          share missing by identifier
  prelink_summary.json             counts, blocking recall, share of records in clusters of 6+
  bands_and_strata.csv             per score band: pairs, mean model probability, true match rate, number of
                                   agreement patterns and strata
  ambiguity_levels.csv             pair-level ambiguity levels: size and true match rate
  truth_by_band_and_ambiguity.csv  true match rate by band x ambiguity level
  perplexity_by_true_duplicates.csv record perplexity by number of true duplicates among candidates
  headroom.csv                     theoretical standard error of the overall match rate under optimal
                                   allocation with TRUE stratum rates, for each stratification, plus the share
                                   of within-band variance each stratification explains
"""
import json
import sys
import numpy as np, pandas as pd
from splink import splink_datasets
import clerical_design as cd
from common import load, RESULTS, budget_of

ID_COLS = ["first_name", "surname", "dob", "postcode_fake", "birth_place", "occupation", "gender"]
COMBOS = [["first_name", "surname"], ["first_name", "surname", "dob"], ["surname", "dob"],
          ["first_name", "surname", "postcode_fake"], ["dob", "postcode_fake"],
          ["first_name", "surname", "dob", "postcode_fake"]]


def signature_ambiguity(df):
    rows = []
    for c in COMBOS:
        sub = df[c].replace("", np.nan)
        complete = sub.notna().all(axis=1)
        g = sub[complete].groupby(c).size()
        sizes = sub[complete].merge(g.rename("n").reset_index(), on=c)["n"]
        # how often a shared signature is shared by DIFFERENT people (uses truth; not available in practice)
        cl = df.loc[complete, c + ["cluster"]].groupby(c)["cluster"].nunique()
        shared = g[g > 1].index
        multi_person = (cl.loc[shared] > 1).mean() if len(shared) else np.nan
        rows.append(dict(combination=" + ".join(c), complete_share=complete.mean(),
                         unique_share=(sizes == 1).mean(), ambiguous_share=(sizes > 1).mean(),
                         largest_group=int(g.max()),
                         shared_signatures_with_more_than_one_person=multi_person))
    return pd.DataFrame(rows)


def headroom(pt):
    y = pt.y.astype(float); N = pt.N
    strat = {"score band": [pt.band], "band x agreement pattern": [pt.band, pt.pattern],
             "band x ambiguity": [pt.band, pt.ambig_level],
             "band x pattern x ambiguity": [pt.band, pt.pattern, pt.ambig_level]}

    def stats(s):
        k = s.max() + 1
        Ns = np.bincount(s, minlength=k).astype(float)
        p = np.bincount(s, weights=y, minlength=k) / Ns
        return Ns, p, p * (1 - p) * Ns / np.maximum(Ns - 1, 1)

    def var(Ns, S2, n):
        return ((Ns / N) ** 2 * np.where(n > 0, (1 - n / Ns) * S2 / np.maximum(n, 1e-12), 0)).sum()

    def neyman(Ns, S2, B):
        w = Ns * np.sqrt(S2); n = np.zeros(len(Ns)); rem = B; act = w > 0
        while rem > 1e-9 and act.any():
            add = np.minimum(w * act / (w * act).sum() * rem, Ns - n) * act
            n += add; rem -= add.sum(); act = act & (n < Ns - 1e-9)
            if add.sum() < 1e-9:
                break
        return n

    Nb, pb, S2b = stats(pt.band.astype(np.int64))
    base_within = (Nb * pb * (1 - pb)).sum()
    rows = []
    for lab in ("1000", "10000", "5pct", "10pct"):
        B = budget_of(pt, lab)
        for name, cols in strat.items():
            s = cd._factorize_keys(cols)
            Ns, p, S2 = stats(s)
            R = 1 - (Ns * p * (1 - p)).sum() / base_within
            for alloc, n in (("proportional", B * Ns / Ns.sum()), ("neyman_true_rates", neyman(Ns, S2, B))):
                rows.append(dict(budget=lab, stratification=name, allocation=alloc, strata=len(Ns),
                                 se_pp=100 * np.sqrt(var(Ns, S2, n)), within_band_variance_explained=R))
    return pd.DataFrame(rows)


def main():
    df = pd.DataFrame(splink_datasets.historical_50k)
    raw, rec, pt = load("quantile")
    # 1. record-level examination
    amb = signature_ambiguity(df); amb.to_csv(RESULTS / "prelink_signature_ambiguity.csv", index=False)
    cs = df.groupby("cluster").size()
    cs.value_counts().sort_index().rename("clusters").to_csv(RESULTS / "prelink_cluster_sizes.csv")
    df[ID_COLS].isna().mean().rename("missing_share").to_csv(RESULTS / "prelink_missingness.csv")
    true_pairs = int((cs * (cs - 1) // 2).sum())
    cand_true = int(pt.y.sum())
    summary = dict(records=len(df), people=int(cs.size), median_records_per_person=float(cs.median()),
                   share_records_in_clusters_6plus=float((df.cluster.map(cs) >= 6).mean()),
                   candidate_pairs=int(pt.N), true_pairs_total=true_pairs, true_pairs_in_candidates=cand_true,
                   blocking_recall=cand_true / true_pairs, candidate_match_rate=float(pt.y.mean()),
                   agreement_patterns=int(pt.pattern.max() + 1),
                   gender_missing_share_of_pairs=float((pt.gender == 2).mean()))
    # 2. candidate pairs: bands, patterns, strata
    d = pd.DataFrame({"band": pt.band + 1, "prob": pt.prob, "y": pt.y, "pat": pt.pattern, "a": pt.ambig_level})
    bt = d.groupby("band").agg(pairs=("y", "size"), mean_probability=("prob", "mean"),
                               true_match_rate=("y", "mean"), patterns=("pat", "nunique"))
    bt["band_x_pattern_strata"] = d.groupby(["band", "pat"]).size().groupby(level=0).size()
    bt["band_x_pattern_x_ambiguity_strata"] = d.groupby(["band", "pat", "a"]).size().groupby(level=0).size()
    bt["median_pairs_band_x_pattern"] = d.groupby(["band", "pat"]).size().groupby(level=0).median()
    bt["median_pairs_band_x_pattern_x_ambiguity"] = d.groupby(["band", "pat", "a"]).size().groupby(level=0).median()
    bt.to_csv(RESULTS / "bands_and_strata.csv")
    sz_bp = d.groupby(["band", "pat"]).size(); sz_bpa = d.groupby(["band", "pat", "a"]).size()
    summary.update(median_stratum_bp=float(sz_bp.median()), median_stratum_bpa=float(sz_bpa.median()),
                   share_strata_lt10_bp=float((sz_bp < 10).mean()), share_strata_lt10_bpa=float((sz_bpa < 10).mean()),
                   share_bp_strata_split_by_ambiguity=float((d.groupby(["band", "pat"]).a.nunique() > 1).mean()))
    ppb = d.groupby("pat").band.nunique()
    summary.update(patterns_in_more_than_one_band=int((ppb > 1).sum()),
                   share_pairs_in_multiband_patterns=float(d.pat.isin(ppb[ppb > 1].index).mean()),
                   band_x_pattern_strata=int(bt.band_x_pattern_strata.sum()),
                   band_x_pattern_x_ambiguity_strata=int(bt.band_x_pattern_x_ambiguity_strata.sum()))
    # 3. ambiguity
    d.groupby("a").agg(pairs=("y", "size"), true_match_rate=("y", "mean"), mean_probability=("prob", "mean")) \
        .to_csv(RESULTS / "ambiguity_levels.csv")
    d.groupby(["band", "a"]).agg(pairs=("y", "size"), true_match_rate=("y", "mean")).reset_index() \
        .to_csv(RESULTS / "truth_by_band_and_ambiguity.csv", index=False)
    y = (raw.cluster_l == raw.cluster_r).to_numpy()
    ids = np.concatenate([raw.unique_id_l, raw.unique_id_r]); yy = np.concatenate([y, y])
    r = rec.join(pd.Series(yy).groupby(ids).sum().rename("true_candidates"))
    r["true_candidates_capped"] = r.true_candidates.clip(upper=6)
    r.groupby("true_candidates_capped").agg(records=("perp_cond", "size"), median_perplexity=("perp_cond", "median"),
                                            mean_perplexity=("perp_cond", "mean")) \
        .to_csv(RESULTS / "perplexity_by_true_duplicates.csv")
    summary["spearman_perplexity_vs_true_duplicates"] = float(r[["perp_cond", "true_candidates"]].corr("spearman").iloc[0, 1])
    # 4. headroom
    hr = headroom(pt); hr.to_csv(RESULTS / "headroom.csv", index=False)
    json.dump(summary, open(RESULTS / "prelink_summary.json", "w"), indent=1)
    print(amb.round(3).to_string(index=False)); print(json.dumps(summary, indent=1)); print(hr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
