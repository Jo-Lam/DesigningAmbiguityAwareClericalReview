"""
summarise.py -- Step 7. Every table and figure in the manuscript, and every number quoted in the text.

Inputs: results/*.csv and *.json from steps 1-6.
Outputs:
  results/tables.md              Tables 2-6 and Supplementary Tables S1-S9 (numbered in order of first citation)
  results/facts.json             every number quoted in the text, already rounded as printed
  figures/fig1_examination       Figure 1: match rate by band x ambiguity; perplexity by true duplicates
  figures/fig2_design_comparison Figure 2: RMSE of each design relative to current practice
  figures/fig3_reviewer_error    Figure 3: RMSE of estimated precision by budget, with and without reviewer error
  figures/figS2_estimator_check  Supplementary Figure S2: design-weighted vs pooled estimates
"""
import json, re
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import designs as dz
from common import RESULTS, FIGURES

BL = ["1000", "10000", "5pct", "10pct"]
BT = {"1000": "1,000 pairs", "10000": "10,000 pairs", "5pct": "5%", "10pct": "10%"}
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False})
PROP = ["PROP-B", "PROP-BA", "PROP-AF", "PROP-BP", "PROP-BPA"]
NEY = {"PROP-B": "OR-B", "PROP-BA": "OR-BA", "PROP-AF": "OR-AF", "PROP-BP": "OR-BP", "PROP-BPA": "OR-BPA"}
OTHER = {"PROP-AF": "AF-eq", "PROP-BPA": "TM-BPA"}
ROWNAME = {"PROP-B": "Score band", "PROP-BA": "Band × ambiguity", "PROP-AF": "Ambiguity, then match score",
           "PROP-BP": "Band × pattern", "PROP-BPA": "Band × pattern × ambiguity"}


def fmt(x, d=None):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if d is None:
        ax = abs(x); d = 3 if ax < 0.1 else (2 if ax < 10 else 1)
    return f"{x:.{d}f}".replace("-", "−")


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf"); fig.savefig(FIGURES / f"{name}.png", dpi=200); plt.close(fig)


def md(df):
    return df.astype(str).to_markdown(index=False, disable_numparse=True)


def main():
    F, T = {}, []
    js = json.load(open(RESULTS / "prelink_summary.json"))
    mc = json.load(open(RESULTS / "model_check.json"))
    cmp_ = pd.concat([pd.read_csv(RESULTS / f"compare_{b}.csv", dtype={"budget_label": str}) for b in BL])
    cmp_.to_csv(RESULTS / "compare_all.csv", index=False)
    C = cmp_.set_index(["design", "budget_label"])
    v = lambda d, b, c: float(C.loc[(d, b), c])

    # --- 4.1 data and model -------------------------------------------------------------------------------
    F.update(records=f"{js['records']:,}", people=f"{js['people']:,}", median_per_person=f"{js['median_records_per_person']:.0f}",
             share_6plus=pct(js["share_records_in_clusters_6plus"]), pairs=f"{js['candidate_pairs']:,}",
             true_pairs_total=f"{js['true_pairs_total']:,}", true_pairs_cand=f"{js['true_pairs_in_candidates']:,}",
             blocking_recall=f"{js['blocking_recall']:.2f}", cand_match_rate=pct(js["candidate_match_rate"]),
             patterns=f"{js['agreement_patterns']:,}", patterns_multiband=f"{js['patterns_in_more_than_one_band']:,}",
             share_multiband=f"{100 * js['share_pairs_in_multiband_patterns']:.0f}",
             strata_bp=f"{js['band_x_pattern_strata']:,}", strata_bpa=f"{js['band_x_pattern_x_ambiguity_strata']:,}",
             median_stratum_bp=f"{js['median_stratum_bp']:.0f}", median_stratum_bpa=f"{js['median_stratum_bpa']:.0f}",
             share_lt10_bp=f"{100 * js['share_strata_lt10_bp']:.0f}", share_lt10_bpa=f"{100 * js['share_strata_lt10_bpa']:.0f}",
             share_bp_split=f"{100 * js['share_bp_strata_split_by_ambiguity']:.0f}",
             rho_pp_dup=f"{js['spearman_perplexity_vs_true_duplicates']:.2f}",
             gender_missing=pct(js["gender_missing_share_of_pairs"]),
             lambda_est=f"{mc['lambda_estimated']:.6f}", lambda_true=f"{mc['lambda_true']:.6f}", brier=f"{mc['brier']:.3f}")
    r0 = cmp_.iloc[0]
    for t in ("0.5", "0.9"):
        F[f"N_acc{t}"] = f"{int(r0[f'N_acc{t}']):,}"; F[f"P{t}_true"] = fmt(r0[f"P{t}_true"], 1); F[f"R{t}_true"] = fmt(r0[f"R{t}_true"], 1)
    al = pd.read_csv(RESULTS / "ambiguity_levels.csv")
    for _, r in al.iterrows():
        F[f"amb{int(r.a)}_pairs"] = f"{int(r.pairs):,}"; F[f"amb{int(r.a)}_rate"] = pct(r.true_match_rate)

    # --- Table 2 and calibration ---------------------------------------------------------------------------
    bt = pd.read_csv(RESULTS / "bands_and_strata.csv")
    t2 = pd.DataFrame({"Match score band": bt.band, "N pairs": bt.pairs.map("{:,}".format),
                       "Mean probability": bt.mean_probability.map("{:.3f}".format),
                       "True match rate (%)": (100 * bt.true_match_rate).map("{:.2f}".format),
                       "N band × pattern strata": bt.band_x_pattern_strata.map("{:,}".format),
                       "Median size, band × pattern": bt.median_pairs_band_x_pattern.map("{:.0f}".format),
                       "N band × pattern × ambiguity strata": bt.band_x_pattern_x_ambiguity_strata.map("{:,}".format),
                       "Median size, band × pattern × ambiguity": bt.median_pairs_band_x_pattern_x_ambiguity.map("{:.0f}".format)})
    T.append("## Table 2. Score bands: size, mean model probability, true match rate, and number and median size of strata\n\n" + md(t2))
    for _, r in bt.iterrows():
        F[f"band{int(r.band)}_rate"] = pct(r.true_match_rate); F[f"band{int(r.band)}_prob"] = f"{r.mean_probability:.2f}"
    tba = pd.read_csv(RESULTS / "truth_by_band_and_ambiguity.csv")
    for _, r in tba.iterrows():
        F[f"tba_b{int(r.band)}_a{int(r.a)}"] = f"{100 * r.true_match_rate:.0f}"

    # --- Table 3: within-band variance explained and theoretical SEs --------------------------------------
    hr = pd.read_csv(RESULTS / "headroom.csv", dtype={"budget": str})
    rows = []
    for st in ["score band", "band x ambiguity", "band x agreement pattern", "band x pattern x ambiguity"]:
        for al_ in ["proportional", "neyman_true_rates"]:
            h = hr[(hr.stratification == st) & (hr.allocation == al_)].set_index("budget")
            rows.append({"Stratification": st.replace(" x ", " × ").replace("agreement ", "").capitalize() if al_ == "proportional" else "",
                         "Allocation": "Proportional" if al_ == "proportional" else "Neyman (true rates)",
                         "Strata": f"{int(h.strata.iloc[0]):,}" if al_ == "proportional" else "",
                         "Within-band variance explained (%)": f"{100 * h.within_band_variance_explained.iloc[0]:.1f}" if al_ == "proportional" else "",
                         **{f"SE, {BT[b]}": fmt(h.loc[b, "se_pp"]) for b in BL}})
            key = {"score band": "b", "band x ambiguity": "ba", "band x agreement pattern": "bp", "band x pattern x ambiguity": "bpa"}[st]
            for b in BL:
                F[f"se_{'prop' if al_ == 'proportional' else 'ney'}_{key}_{b}"] = fmt(h.loc[b, "se_pp"])
            F[f"R_{key}"] = f"{100 * h.within_band_variance_explained.iloc[0]:.1f}"
    T.append("## Table 3. Within-band variance in match status explained by each set of strata, and standard error (pp) of the overall match rate under proportional and Neyman allocation\n\n" + md(pd.DataFrame(rows)))

    # --- Table 4: true values and RMSE, current practice vs band x pattern --------------------------------
    rows = []
    for est, lab, tcol in [("mr", "Overall match rate", "mr_true"), ("P0.5", "Precision (threshold 0.5)", "P0.5_true"),
                           ("R0.5", "Recall (threshold 0.5)", "R0.5_true"), ("hi", "Match rate, most ambiguous level", "hi_true")]:
        for dsg, dl in [("PROP-B", "Current practice"), ("PROP-BP", "Band × pattern")]:
            rows.append({"Estimand": lab if dsg == "PROP-B" else "", "True value (%)": fmt(v("PROP-B", "1000", tcol), 1) if dsg == "PROP-B" else "",
                         "Design": dl, **{BT[b]: fmt(v(dsg, b, f"{est}_rmse")) for b in BL}})
        rows.append({"Estimand": "", "True value (%)": "", "Design": "Reduction (%)",
                     **{BT[b]: f"{100 * (1 - v('PROP-BP', b, f'{est}_rmse') / v('PROP-B', b, f'{est}_rmse')):.0f}" for b in BL}})
        F[f"{est}_true"] = fmt(v("PROP-B", "1000", tcol), 1)
    T.append("## Table 4. True values and RMSE (pp): current practice and band × pattern strata, both proportional\n\n" + md(pd.DataFrame(rows)))

    # --- Tables 5 and 6: RMSE grids at 10,000 pairs and 5% -------------------------------------------------
    def grid(col):
        rows = []
        for d in PROP:
            r = {"Strata": ROWNAME[d]}
            for b in ("10000", "5pct"):
                r[f"Proportional, {BT[b]}"] = fmt(v(d, b, col)) + (" (ref.)" if d == "PROP-B" else "")
            for b in ("10000", "5pct"):
                r[f"Neyman, true rates, {BT[b]}"] = fmt(v(NEY[d], b, col))
            for b in ("10000", "5pct"):
                r[f"Other allocation, {BT[b]}"] = (fmt(v(OTHER[d], b, col)) + (" ᵃ" if d == "PROP-AF" else " ᵇ")) if d in OTHER else "–"
            rows.append(r)
        return pd.DataFrame(rows)
    T.append("## Table 5. RMSE (pp) of the estimated overall match rate, 300 replicates\n\n" + md(grid("mr_rmse")) +
             "\n\nᵃ Equal budget per ambiguity level. ᵇ Target margins of error (Equation 6).")
    T.append("## Table 6. RMSE (pp) of estimated precision and recall (threshold 0.5), 300 replicates\n\n**Precision**\n\n" +
             md(grid("P0.5_rmse")) + "\n\n**Recall**\n\n" + md(grid("R0.5_rmse")))

    # --- facts for 4.3-4.5 -----------------------------------------------------------------------------------
    for b in BL:
        for est in ("mr", "P0.5", "R0.5", "hi"):
            for d in dz.DESIGNS:
                F[f"{est}_rmse_{d}_{b}"] = fmt(v(d, b, f"{est}_rmse"))
                F[f"{est}_rel_{d}_{b}"] = f"{v(d, b, f'{est}_rmse') / v('PROP-B', b, f'{est}_rmse'):.2f}"
                F[f"{est}_red_{d}_{b}"] = f"{100 * (1 - v(d, b, f'{est}_rmse') / v('PROP-B', b, f'{est}_rmse')):.0f}"
        F[f"bp_share_of_neyman_gain_{b}"] = f"{100 * (v('PROP-B', b, 'mr_rmse') - v('PROP-BP', b, 'mr_rmse')) / (v('PROP-B', b, 'mr_rmse') - v('OR-BP', b, 'mr_rmse')):.0f}"
        F[f"P0.5_ciw_PROP-B_{b}"] = fmt(v("PROP-B", b, "P0.5_ciw")); F[f"P0.5_ciw_PROP-BP_{b}"] = fmt(v("PROP-BP", b, "P0.5_ciw"))
    prop_designs = [d for d in dz.DESIGNS if d.startswith("PROP-")]
    for m in ("mr", "P0.5", "R0.5"):
        cov = cmp_[cmp_.design.isin(prop_designs)][f"{m}_cov"]
        F[f"{m}_cov_prop_range"] = f"{cov.min():.2f} to {cov.max():.2f}"
    F["max_abs_bias_B_BP"] = fmt(cmp_[cmp_.design.isin(["PROP-B", "PROP-BP"])][["mr_bias", "P0.5_bias", "R0.5_bias"]].abs().max().max())

    # --- Supplementary tables --------------------------------------------------------------------------------
    sig = pd.read_csv(RESULTS / "prelink_signature_ambiguity.csv")
    T.append("## Supplementary Table S2. Match-key ambiguity among complete records\n\n" + md(pd.DataFrame({
        "Identifiers": sig.combination.str.replace("postcode_fake", "postcode").str.replace("first_name", "forename"),
        "Complete records (%)": (100 * sig.complete_share).map("{:.1f}".format),
        "Sharing their values (%)": (100 * sig.ambiguous_share).map("{:.1f}".format),
        "Largest group": sig.largest_group,
        "Shared values with more than one person (%)": (100 * sig.shared_signatures_with_more_than_one_person).map("{:.1f}".format)})))
    for col, name in [("mr_rmse", "S5a. RMSE (pp), overall match rate"), ("P0.5_rmse", "S5b. RMSE (pp), precision (0.5)"),
                      ("R0.5_rmse", "S5c. RMSE (pp), recall (0.5)"), ("hi_rmse", "S5d. RMSE (pp), most ambiguous level"),
                      ("P0.9_rmse", "S5e. RMSE (pp), precision (0.9)"), ("R0.9_rmse", "S5f. RMSE (pp), recall (0.9)"),
                      ("mr_bias", "S4a. Bias (pp), overall match rate"), ("P0.5_bias", "S4b. Bias (pp), precision (0.5)"),
                      ("R0.5_bias", "S4c. Bias (pp), recall (0.5)"), ("mr_cov", "S8a. Coverage, overall match rate"),
                      ("P0.5_cov", "S8b. Coverage, precision (0.5)"), ("R0.5_cov", "S8c. Coverage, recall (0.5)")]:
        t = cmp_.pivot_table(index="design", columns="budget_label", values=col)[BL].reindex(dz.DESIGNS)
        t = t.map(lambda x: f"{x:.2f}" if "cov" in col else fmt(x))
        t.insert(0, "Design", [dz.LABELS[d] for d in t.index]); t.columns = ["Design"] + [BT[b] for b in BL]
        T.append(f"## Supplementary Table {name}\n\n" + md(t))
    a = cmp_[cmp_.budget_label == "10000"].set_index("design").reindex(dz.DESIGNS)
    t = a[[f"alloc_b{i}" for i in range(1, 11)]].round(0).astype(int)
    t.insert(0, "Design", [dz.LABELS[d] for d in t.index]); t.columns = ["Design"] + [f"b{i}" for i in range(1, 11)]
    T.append("## Supplementary Table S6. Mean number of pairs reviewed in each score band, 10,000 pairs\n\n" + md(t))
    sens = pd.read_csv(RESULTS / "sensitivity.csv", dtype={"budget_label": str})
    sc = sens[sens.group == "strata cap"]
    base = cmp_[cmp_.design.isin(["PROP-BP", "PROP-BPA"]) & cmp_.budget_label.isin(["1000", "5pct"])].assign(config="max strata B/10 (default)")
    s5 = pd.concat([base, sc]).sort_values(["budget_label", "design", "config"])
    T.append("## Supplementary Table S7. Maximum number of strata: RMSE (pp)\n\n" + md(pd.DataFrame({
        "Budget": s5.budget_label.map(BT), "Design": s5.design.map(dz.LABELS), "Max. strata": s5.config, "Strata": s5.strata,
        "Match rate": s5.mr_rmse.map(fmt), "Precision": s5["P0.5_rmse"].map(fmt), "Recall": s5["R0.5_rmse"].map(fmt)})))
    for _, r in s5.iterrows():
        F[f"cap_{r.design}_{r.budget_label}_{r.config.split()[-2] if 'default' in r.config else r.config.split()[-1]}_mr"] = fmt(r.mr_rmse)
        F[f"cap_{r.design}_{r.budget_label}_{r.config.split()[-2] if 'default' in r.config else r.config.split()[-1]}_strata"] = f"{int(r.strata):,}"
    re_ = sens[sens.group == "reviewer error"]
    T.append("## Supplementary Table S9. Reviewer error: bias (pp) and coverage of 95% intervals\n\n" + md(pd.DataFrame({
        "Budget": re_.budget_label.map(BT), "Design": re_.design.map(dz.LABELS), "Reviewer error": re_.config,
        "MR bias": re_.mr_bias.map(lambda x: fmt(x, 2)), "P bias": re_["P0.5_bias"].map(lambda x: fmt(x, 2)),
        "R bias": re_["R0.5_bias"].map(lambda x: fmt(x, 2)), "P cov.": re_["P0.5_cov"].map("{:.2f}".format),
        "R cov.": re_["R0.5_cov"].map("{:.2f}".format)})))
    for cfg, k in [("5% uniform", "u"), ("5%, x3 in most ambiguous", "x3")]:
        s = re_[(re_.config == cfg) & re_.budget_label.isin(BL)]
        for m in ("mr", "P0.5", "R0.5"):
            F[f"re_{k}_{m}_bias_range"] = f"{fmt(s[f'{m}_bias'].min(), 2)} to {fmt(s[f'{m}_bias'].max(), 2)}"
        F[f"re_{k}_P_cov_max_10000_5pct"] = f"{s[s.budget_label.isin(['10000', '5pct'])]['P0.5_cov'].max():.2f}"
    mp = pd.read_csv(RESULTS / "model_parameters.csv")
    T.append("## Supplementary Table S1. Fitted linkage model: m- and u-probabilities and match weights\n\n" + md(pd.DataFrame({
        "Comparison": mp.comparison, "Level": mp.level, "m": mp.m.map("{:.4f}".format), "u": mp.u.map("{:.5f}".format),
        "Match weight": mp.match_weight.map("{:+.2f}".format), "TF adjusted": mp.tf_adjusted.map({True: "yes", False: ""})})))
    ac = json.load(open(RESULTS / "ambiguity_check.json"))
    F.update(R_ba_model=f"{100 * ac['within_band_variance_explained_ambiguity']:.1f}",
             R_ba_other_people=f"{100 * ac['within_band_variance_explained_ambiguity_other_people']:.0f}",
             rho_amb_vs_other=f"{ac['spearman_ambiguity_score_vs_other_people_score']:.2f}")
    abl = pd.read_csv(RESULTS / "ambiguity_by_level.csv").set_index("a")
    for k in (1, 2, 3):
        F[f"amb{k}_fmr_acc"] = pct(abl.loc[k, "false_match_rate_accepted"]); F[f"amb{k}_share_missed"] = f"{100 * abl.loc[k, 'share_of_missed_matches']:.0f}"
        F[f"amb{k}_share_false"] = f"{100 * abl.loc[k, 'share_of_false_matches']:.0f}"
    T.append("## Supplementary Table S3. Ambiguity levels: match status, calibration and linkage errors\n\n" + md(pd.DataFrame({
        "Level": abl.index, "Pairs": abl.pairs.map("{:,.0f}".format), "True match rate (%)": (100 * abl.true_match_rate).map("{:.1f}".format),
        "Mean probability (%)": (100 * abl.mean_probability).map("{:.1f}".format),
        "False-match rate among accepted (%)": (100 * abl.false_match_rate_accepted).map(lambda x: "–" if np.isnan(x) else f"{x:.1f}"),
        "Share of all false matches (%)": (100 * abl.share_of_false_matches).map("{:.0f}".format),
        "Share of all missed matches (%)": (100 * abl.share_of_missed_matches).map("{:.0f}".format)})))
    ec = pd.read_csv(RESULTS / "estimator_check.csv").set_index("design")
    F.update(ec_bpa_share=f"{100 * ec.loc['D2A', 'sample_share']:.0f}", ec_bpa_strata=f"{int(ec.loc['D2A', 'strata']):,}",
             ec_bpa_wbias=fmt(ec.loc["D2A", "weighted_bias_pp"], 3), ec_bpa_pbias=fmt(ec.loc["D2A", "pooled_bias_pp"], 1),
             ec_bp_pbias=fmt(ec.loc["D2", "pooled_bias_pp"], 1), ec_bpa_psd=fmt(ec.loc["D2A", "pooled_sd_pp"], 3))

    def _key(t):
        m = re.match(r"## Supplementary Table S(\d+)([a-z]?)", t)
        return (1, int(m.group(1)), m.group(2)) if m else (0, 0, "")
    T = sorted(T, key=_key)
    (RESULTS / "tables.md").write_text("# Tables (generated by src/summarise.py)\n\n" + "\n\n".join(T) + "\n")
    json.dump(F, open(RESULTS / "facts.json", "w"), indent=1, ensure_ascii=False)
    figures(cmp_, sens)
    print(f"{len(T)} tables, {len(F)} facts")


def figures(cmp_, sens):
    C = cmp_.set_index(["design", "budget_label"])
    # Figure 1 -------------------------------------------------------------------------------------------------
    tba = pd.read_csv(RESULTS / "truth_by_band_and_ambiguity.csv")
    pp = pd.read_csv(RESULTS / "perplexity_by_true_duplicates.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for lev, col, lab in [(1, "#86b6ef", "Level 1 (least ambiguous)"), (2, BLUE, "Level 2"), (3, "#104281", "Level 3 (most ambiguous)")]:
        s = tba[(tba.a == lev) & (tba.pairs >= 100)]
        a.plot(s.band, 100 * s.true_match_rate, "-o", color=col, lw=2, ms=5, label=lab, mec="#fcfcfb", mew=1)
    a.set_xticks(range(1, 11)); a.set_xlabel("Score band"); a.set_ylabel("True match rate (%)")
    a.set_title("(a) True match rate by score band and ambiguity level", loc="left"); a.legend(fontsize=7.5)
    a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    lab = [str(int(x)) if x < 6 else "6+" for x in pp.true_candidates_capped]
    b.bar(lab, pp.median_perplexity, color=BLUE, width=0.7)
    b.set_xlabel("True duplicates among a record's candidates"); b.set_ylabel("Median conditional perplexity")
    b.set_title("(b) Perplexity by number of true duplicates", loc="left"); b.grid(axis="y", color=GRID, lw=0.6); b.set_axisbelow(True)
    fig.tight_layout(); save(fig, "fig1_examination")
    # Figure 2 -------------------------------------------------------------------------------------------------
    rows = ["PROP-B", "PROP-BA", "PROP-AF", "AF-eq", "PROP-BP", "PROP-BPA", "TM-BPA"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.6), sharey=True)
    for ax, bl in zip(axes, BL):
        for i, d in enumerate(rows):
            for j, (m, col, lab) in enumerate([("mr", BLUE, "Match rate"), ("P0.5", ORANGE, "Precision"), ("R0.5", AQUA, "Recall")]):
                val = C.loc[(d, bl), f"{m}_rmse"] / C.loc[("PROP-B", bl), f"{m}_rmse"]
                ax.barh(i + (j - 1) * 0.26, val, height=0.24, color=col, label=lab if (i == 0 and bl == BL[0]) else None)
            if d in NEY:
                nv = C.loc[(NEY[d], bl), "mr_rmse"] / C.loc[("PROP-B", bl), "mr_rmse"]
                ax.plot([nv, nv], [i - 0.4, i - 0.12], color=INK, lw=2)
        ax.axvline(1, ls="--", color=MUTED, lw=0.8); ax.set_title(BT[bl]); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(rows)), [dz.LABELS[d] for d in rows]); axes[0].invert_yaxis()
    fig.supxlabel("RMSE relative to current practice (proportional within score bands; 1 = same). Black tick: Neyman ceiling, match rate", fontsize=8.5)
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.6, 1.0)); fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig2_design_comparison")
    # Figure 3: reviewer error does not shrink with the budget ---------------------------------------------------
    re_ = sens[sens.group == "reviewer error"]
    bls = ["1000", "10000", "5pct"]; x = [1000, 10000, 22537]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for cfg, col, lab in [("none", BLUE, "No reviewer error"), ("5% uniform", ORANGE, "5% of decisions wrong"),
                          ("5%, x3 in most ambiguous", "#b2182b", "5%, ×3 for most ambiguous pairs")]:
        y = [re_[(re_.budget_label == b) & (re_.design == "PROP-B") & (re_.config == cfg)]["P0.5_rmse"].iloc[0] for b in bls]
        ax.plot(x, y, marker="o", color=col, lw=2, ms=6, mec="#fcfcfb", label=lab)
        ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xscale("log"); ax.set_xticks(x, ["1,000", "10,000", "22,537\n(5%)"]); ax.minorticks_off()
    ax.set_xlabel("Number of reviewed pairs"); ax.set_ylabel("RMSE of estimated precision (pp)")
    ax.set_ylim(0, None); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(800, 32000); ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0.02, 0.45))
    fig.tight_layout(); save(fig, "fig3_reviewer_error")
    # Supplementary Figure S2 ------------------------------------------------------------------------------------
    eb = pd.read_csv(RESULTS / "estimator_check_by_band.csv"); eb = eb[eb.design == "D2A"]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(eb.band, eb.pooled_bias_pp, "-s", color=ORANGE, lw=2, ms=6, label="Pooled within band")
    ax.plot(eb.band, eb.weighted_bias_pp, "-o", color=BLUE, lw=2, ms=6, label="Design-weighted")
    ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(range(1, 11)); ax.set_xlabel("Score band"); ax.set_ylabel("Mean error (pp)")
    ax.legend(); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); save(fig, "figS2_estimator_check")


if __name__ == "__main__":
    main()
