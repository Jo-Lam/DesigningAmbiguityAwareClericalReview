"""
person_level.py -- Step 1e. Pairwise against person-level linkage quality (Discussion).

The estimands in the paper are pairwise. Here the links accepted at a threshold are joined into
clusters (transitive closure, i.e. connected components), and the clusters are compared with the
true people. Also reports how true matching candidate pairs are spread over people of different
sizes, which determines what a pair-level clerical sample mainly describes.

    python src/person_level.py [threshold]        default threshold 0.5

Output: results/person_level.json, results/person_level_by_size.csv
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from splink import splink_datasets

from common import RESULTS, load


def pairs_within(sizes):
    s = np.asarray(sizes, dtype=np.int64)
    return int((s * (s - 1) // 2).sum())


def main(t=0.5):
    raw, _, _ = load(levels="none")
    rec = pd.DataFrame(splink_datasets.historical_50k)[["unique_id", "cluster"]].drop_duplicates("unique_id")
    rec = rec.sort_values("unique_id").reset_index(drop=True)
    idx = pd.Series(np.arange(len(rec)), index=rec["unique_id"])
    n = len(rec)

    y = (raw["cluster_l"] == raw["cluster_r"]).to_numpy()
    acc = (raw["match_probability"] >= t).to_numpy()
    tp, n_acc, n_true_cand = int((y & acc).sum()), int(acc.sum()), int(y.sum())
    true_people = rec["cluster"].to_numpy()
    true_sizes = pd.Series(true_people).value_counts()
    n_true_pairs = pairs_within(true_sizes)

    # predicted clusters: connected components of accepted links
    a = raw.loc[acc]
    i, j = idx[a["unique_id_l"]].to_numpy(), idx[a["unique_id_r"]].to_numpy()
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    pred_sizes = pd.Series(comp).value_counts()
    implied = pairs_within(pred_sizes)
    both = pd.DataFrame({"pred": comp, "true": true_people})
    implied_tp = pairs_within(both.groupby(["pred", "true"]).size())

    # people split across clusters, clusters merging people, exact reconstruction
    per_person = both.groupby("true")["pred"].nunique()
    per_cluster = both.groupby("pred")["true"].nunique()
    merged = per_cluster[per_cluster > 1].index
    people_in_merged = both.loc[both["pred"].isin(merged), "true"].nunique()
    exact = int(((per_person == 1) & both.groupby("true")["pred"].first().map(per_cluster).eq(1)).sum())

    # where the true matching candidate pairs lie, by size of person
    cand_true = raw.loc[y, "cluster_l"].value_counts().reindex(true_sizes.index, fill_value=0)
    acc_true = raw.loc[y & acc, "cluster_l"].value_counts().reindex(true_sizes.index, fill_value=0)
    people = pd.DataFrame({"size": true_sizes, "cand_true": cand_true, "acc_true": acc_true})
    has = people[people["cand_true"] > 0]
    recall_person = (has["acc_true"] / has["cand_true"]).mean()
    big = people["size"] >= 10
    top = people["size"] >= people["size"].quantile(0.9)
    bins = pd.cut(has["size"], [1, 5, 10, 20, np.inf], labels=["2-5", "6-10", "11-20", "21+"])
    by_size = has.groupby(bins, observed=True).agg(people=("size", "size"), cand_true=("cand_true", "sum"),
                                                   acc_true=("acc_true", "sum"))
    by_size["recall_among_candidates"] = by_size["acc_true"] / by_size["cand_true"]
    by_size.to_csv(RESULTS / "person_level_by_size.csv")

    out = {
        "threshold": t,
        "pair_precision": tp / n_acc,
        "pair_recall_candidates": tp / n_true_cand,
        "pair_recall_all": tp / n_true_pairs,
        "implied_pairs": implied,
        "cluster_precision": implied_tp / implied,
        "cluster_recall_all": implied_tp / n_true_pairs,
        "people": int(len(true_sizes)),
        "predicted_clusters": int(len(pred_sizes)),
        "people_split": int((per_person > 1).sum()),
        "share_people_split": float((per_person > 1).mean()),
        "clusters_merging": int(len(merged)),
        "people_in_merging_clusters": int(people_in_merged),
        "people_exact": exact,
        "share_people_exact": exact / len(true_sizes),
        "median_records_per_person": float(true_sizes.median()),
        "p90_records_per_person": float(true_sizes.quantile(0.9)),
        "max_records_per_person": int(true_sizes.max()),
        "share_people_10plus": float(big.mean()),
        "share_true_cand_pairs_10plus": float(people.loc[big, "cand_true"].sum() / people["cand_true"].sum()),
        "share_true_cand_pairs_top10pct": float(people.loc[top, "cand_true"].sum() / people["cand_true"].sum()),
        "recall_candidates_person_averaged": float(recall_person),
    }
    (RESULTS / "person_level.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(f"{k:36s} {v:.4f}" if isinstance(v, float) else f"{k:36s} {v}")
    print(by_size)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.5)
