"""
model_check.py -- Step 1b. Sense-check the fitted linkage model.

Outputs:
  results/model_parameters.csv           m, u and match weight for every comparison level
  results/calibration_by_band.csv        mean model probability and true match rate by score band
  results/calibration_band_ambiguity.csv the same by band x ambiguity level
  results/band6_top_patterns.csv         largest agreement patterns in the least-calibrated band
  figures/figS1_model_check.(pdf|png)    (a) match weights; (b) calibration by score band
"""
import json, math
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, RESULTS, FIGURES

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
NAMES = {"first_name_surname": "Forename + surname", "dob": "Date of birth", "postcode_fake": "Postcode",
         "birth_place": "Birthplace", "occupation": "Occupation"}


def parameters():
    m = json.load(open(RESULTS / "model.json"))
    rows = []
    for c in m["comparisons"]:
        for l in c["comparison_levels"]:
            if l.get("is_null_level"):
                continue
            mp, up = l.get("m_probability"), l.get("u_probability")
            rows.append(dict(comparison=NAMES[c["output_column_name"]], level=l["label_for_charts"], m=mp, u=up,
                             match_weight=math.log2(mp / up) if (mp and up) else np.nan,
                             tf_adjusted="tf_adjustment_column" in l))
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "model_parameters.csv", index=False)
    return t, m["probability_two_random_records_match"]


def main():
    t, lam = parameters()
    raw, rec, pt = load("quantile")
    d = pd.DataFrame({"band": pt.band + 1, "a": pt.ambig_level, "p": pt.prob, "y": pt.y})
    cb = d.groupby("band").agg(pairs=("y", "size"), mean_probability=("p", "mean"), true_match_rate=("y", "mean"))
    cb.to_csv(RESULTS / "calibration_by_band.csv")
    d.groupby(["band", "a"]).agg(pairs=("y", "size"), mean_probability=("p", "mean"), true_match_rate=("y", "mean")) \
        .to_csv(RESULTS / "calibration_band_ambiguity.csv")
    g = [c for c in raw.columns if c.startswith("gamma_")]
    b6 = raw.loc[pt.band == 5, g].assign(y=pt.y[pt.band == 5], p=pt.prob[pt.band == 5])
    b6.groupby(g).agg(pairs=("y", "size"), mean_probability=("p", "mean"), true_match_rate=("y", "mean")) \
        .sort_values("pairs", ascending=False).head(15).to_csv(RESULTS / "band6_top_patterns.csv")
    yv = pt.y.astype(float)
    n_rec = len(pd.unique(np.concatenate([raw.unique_id_l, raw.unique_id_r])))
    js = json.load(open(RESULTS / "prelink_summary.json")) if (RESULTS / "prelink_summary.json").exists() else {}
    n_all = js.get("records", 50578)
    summ = dict(lambda_estimated=lam, lambda_true=js.get("true_pairs_total", np.nan) / (n_all * (n_all - 1) / 2),
                brier=float(((pt.prob - yv) ** 2).mean()))
    json.dump(summ, open(RESULTS / "model_check.json", "w"), indent=1)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ypos, ylab, y = [], [], 0
    for comp, gg in t.groupby("comparison", sort=False):
        a.text(-16.5, y - 0.1, comp, fontweight="bold", color=INK, ha="left", va="center")
        y += 1
        for _, r in gg.iterrows():
            a.barh(y, r.match_weight, height=0.7, color=BLUE if r.match_weight >= 0 else ORANGE)
            a.text(r.match_weight + (0.3 if r.match_weight >= 0 else -0.3), y, f"{r.match_weight:+.1f}",
                   va="center", ha="left" if r.match_weight >= 0 else "right", fontsize=8, color=INK)
            ypos.append(y); ylab.append(r.level + (" (TF)" if r.tf_adjusted else "")); y += 1
        y += 0.6
    a.set_yticks(ypos, ylab, fontsize=7.5); a.invert_yaxis(); a.axvline(0, color=MUTED, lw=0.8)
    a.set_xlim(-8, 14.5); a.set_xlabel("Match weight, log2(m/u), before term-frequency (TF) adjustment")
    a.set_title("(a) Match weight by comparison level", loc="left", color=INK)
    a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
    b.plot(cb.mean_probability, cb.true_match_rate, "-o", color=BLUE, lw=2, ms=6, mec="#fcfcfb", mew=1.5)
    for k in cb.index:
        b.annotate(str(k), (cb.mean_probability[k], cb.true_match_rate[k]), xytext=(5, -9),
                   textcoords="offset points", fontsize=7, color=MUTED)
    b.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=0.8)
    b.set_xlim(-0.02, 1.02); b.set_ylim(-0.02, 1.02)
    b.set_xlabel("Mean model match probability in score band"); b.set_ylabel("True match rate in score band")
    b.set_title("(b) Calibration by score band (labels: band)", loc="left", color=INK)
    b.grid(color=GRID, lw=0.6); b.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURES / "figS1_model_check.pdf"); fig.savefig(FIGURES / "figS1_model_check.png", dpi=200)
    print(t.round(4).to_string()); print(cb.round(3).to_string())


if __name__ == "__main__":
    main()
