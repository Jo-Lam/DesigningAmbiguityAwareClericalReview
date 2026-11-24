# Designing clerical review samples for estimating record linkage quality (v5)

Code for *Designing clerical review samples to estimate record linkage quality: stratification by agreement
pattern and record-level ambiguity* (Lam et al., in preparation). Every number, table and figure in the v5
manuscript is produced by this repository from Splink's public `historical_50k` dataset.

## Reproduce everything

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py                                     # about 30 minutes on two cores
python run_all.py --quick                             # smoke test, 10% of replicates
```

`run_all.py` runs the steps below in order. The dataset is downloaded by Splink on first use. All random steps are
seeded (u-probability sampling: seed 42; each simulation cell: a seed derived from its design and budget label),
so a re-run reproduces the results exactly on the same package versions. Scored pairs are sorted by record
identifiers before saving, because DuckDB returns rows in a nondeterministic order. Two independent full runs
(23 November 2026) gave identical simulation results, `tables.md` and `facts.json`; the only differences were
floating-point noise (≤ 1e-4) in diagnostic values that are not printed at that precision.

**Monte Carlo error.** With 300 replicates the Monte Carlo standard error of an RMSE is about 4% of its value, so
ratios of two RMSEs carry roughly ±6% (one SE). Small differences between designs should be read against the exact
theoretical standard errors in `headroom.csv` (Table 3).

| Step | Script | Produces |
|---|---|---|
| 1 | `src/score_pairs.py` | Fitted Splink model (`results/model.json`) and scored candidate pairs (`data/pairs_raw.parquet`, not tracked) |
| 1b | `src/prelinkage.py` | Examination of records and candidate pairs: match-key ambiguity, records per person, blocking recall, bands and strata, ambiguity levels, perplexity by true duplicates, within-band variance explained and theoretical standard errors (`headroom.csv`) |
| 1c | `src/model_check.py` | Model parameters, calibration by band and by band × ambiguity, largest patterns in band 6, Supplementary Figure S1 |
| 1d | `src/ambiguity_check.py` | Ambiguity levels: true-match proportion, calibration gap, false and missed matches by level; within-band variance explained; truth-based diagnostic separating confusability with other people from the number of a person's records |
| 1e | `src/person_level.py` | Pairwise against person-level quality: links joined into clusters, people split and merged, share of true matching pairs held by people with many records (Discussion) |
| 2 | `src/estimator_check.py` | Design-weighted against pooled-within-band estimates (Supplementary Note 1, Supplementary Figure S2) |
| 3 | `src/evaluate.py compare <budget>` | All designs at 1,000 pairs, 10,000 pairs, 5% and 10% of candidate pairs; 300 replicates |
| 4 | `src/evaluate.py sens` | Maximum number of strata; reviewer error; 200 replicates |
| 5 | `src/summarise.py` | `results/tables.md` (Tables 2–6, S1–S9), `results/facts.json` (every number quoted in the text), `figures/` |

## Linkage model (v5)

Fellegi–Sunter deduplication model in Splink 4.0.17 (DuckDB), ten blocking rules for prediction.

- **Comparisons.** Forename and surname jointly: exact full name (term-frequency adjusted), both names
  Jaro–Winkler ≥ 0.88, exact surname only (TF), exact forename only (TF), other. The "names reversed" and "both names
  Jaro–Winkler ≥ 0.92" levels of Splink's default are dropped. Date of birth and postcode: Splink's default graded
  comparisons. Birthplace and occupation: exact match with TF adjustment.
- **Training.** λ from three deterministic rules assuming recall 0.6; u-probabilities by random sampling
  (10 million pairs); m-probabilities by expectation maximisation on three blocking rules (date of birth; forename +
  surname; postcode area + year of birth), so that every comparison is trained in at least one session.

The v3 model (October 2026) trained neither u-probabilities nor date-of-birth parameters; all v4 simulation results
came from that model and are superseded.

The December 2025 proof-of-concept scripts that previously sat at the root of this repository are in
`archive/2025-12_proof_of_concept/`, with their original README.

## Designs

| Code | Strata | Allocation |
|---|---|---|
| `PROP-B` | score band (deciles of match probability) | proportional (**current practice**, reference) |
| `PROP-BA` | band × ambiguity level | proportional |
| `PROP-AF` | ambiguity level, then score deciles within level | proportional |
| `AF-eq` | as `PROP-AF` | equal budget per ambiguity level, proportional within level |
| `PROP-BP` | band × agreement pattern | proportional |
| `PROP-BPA` | band × pattern × ambiguity level | proportional |
| `TM-BPA` | band × pattern × ambiguity level | target margins of error from model probabilities (Eq. 6) |
| `OR-*` | as the matching `PROP-*` | Neyman allocation with the **true** stratum rates (ceiling; not realisable) |

Strata are capped at B/10 (largest kept, the rest merged by band, and by ambiguity level where used), with at least
two pairs per stratum. Ambiguity levels: level 0 if both records have matchability < 0.05; otherwise tertiles of
max(1 − 1/PP) over the pair's two records.

## Layout

```
run_all.py            driver
src/                  pipeline (clerical_design.py holds the reusable sampling and estimation library)
results/              all outputs (CSV, JSON, tables.md, facts.json, model.json)
figures/              Figures 1-3, S1, S2 (PDF and PNG)
archive/              earlier proof-of-concept scripts (December 2025), kept for reference; not used by the pipeline
data/                 scored pairs cache (regenerated by step 1; not tracked)
```
