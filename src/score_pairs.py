"""
score_pairs.py -- Step 1. Fit a Fellegi-Sunter deduplication model to Splink's historical_50k dataset and score
every candidate pair.

Model (v5):
  comparisons  forename + surname (joint): exact full name (term-frequency adjusted), both names Jaro-Winkler >= 0.88,
               exact surname only (TF), exact forename only (TF), other.  The v3 "names reversed" and
               "both names JW >= 0.92" levels are dropped.
               date of birth, postcode: Splink 4.0.17 default graded comparisons.
               birthplace, occupation: exact match with term-frequency adjustment.
  training     lambda (probability two random records match) from three deterministic rules, assuming recall 0.6;
               u-probabilities by random sampling (fixed seed);
               m-probabilities by expectation maximisation on three blocking rules, so that every comparison is
               trained in at least one session (a session cannot train the columns it blocks on).

Outputs: data/pairs_raw.parquet (scored candidate pairs), results/model.json (fitted Splink model).
"""
import time
import pandas as pd
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on, splink_datasets
from common import DATA, RESULTS

PAIRS = DATA / "pairs_raw.parquet"
SEED = 42

BLOCKING_RULES = [
    block_on("substr(first_name,1,3)", "substr(surname,1,4)"),
    block_on("surname", "dob"),
    block_on("first_name", "dob"),
    block_on("postcode_fake", "first_name"),
    block_on("postcode_fake", "surname"),
    block_on("dob", "birth_place"),
    block_on("substr(postcode_fake,1,3)", "dob"),
    block_on("substr(postcode_fake,1,3)", "first_name"),
    block_on("substr(postcode_fake,1,3)", "surname"),
    block_on("substr(first_name,1,2)", "substr(surname,1,2)", "substr(dob,1,4)"),
]
LAMBDA_RULES = [block_on("first_name", "surname", "dob"),
                block_on("substr(first_name,1,2)", "surname", "substr(postcode_fake,1,2)"),
                block_on("dob", "postcode_fake")]
LAMBDA_RECALL = 0.6
EM_RULES = [block_on("dob"),                                         # trains names, postcode, birthplace, occupation
            block_on("first_name", "surname"),                       # trains dob, postcode, birthplace, occupation
            block_on("substr(postcode_fake,1,3)", "substr(dob,1,4)")]  # trains names, birthplace, occupation
U_MAX_PAIRS = 1e7

NAME_COMPARISON = {
    "output_column_name": "first_name_surname",
    "comparison_description": "Forename and surname (joint)",
    "comparison_levels": [
        {"sql_condition": '("first_name_l" IS NULL OR "first_name_r" IS NULL) AND ("surname_l" IS NULL OR "surname_r" IS NULL)',
         "label_for_charts": "Both names missing", "is_null_level": True},
        {"sql_condition": '"first_name_surname_concat_l" = "first_name_surname_concat_r"',
         "label_for_charts": "Exact full name",
         "tf_adjustment_column": "first_name_surname_concat", "tf_adjustment_weight": 1.0},
        {"sql_condition": '(jaro_winkler_similarity("first_name_l", "first_name_r") >= 0.88) AND '
                          '(jaro_winkler_similarity("surname_l", "surname_r") >= 0.88)',
         "label_for_charts": "Both names Jaro-Winkler >= 0.88"},
        {"sql_condition": '"surname_l" = "surname_r"', "label_for_charts": "Exact surname only",
         "tf_adjustment_column": "surname", "tf_adjustment_weight": 1.0},
        {"sql_condition": '"first_name_l" = "first_name_r"', "label_for_charts": "Exact forename only",
         "tf_adjustment_column": "first_name", "tf_adjustment_weight": 1.0},
        {"sql_condition": "ELSE", "label_for_charts": "Other"},
    ],
}


def load_data() -> pd.DataFrame:
    df = pd.DataFrame(splink_datasets.historical_50k).copy()
    df["first_name_surname_concat"] = df["first_name"] + " " + df["surname"]
    return df


def settings() -> SettingsCreator:
    return SettingsCreator(
        link_type="dedupe_only",
        blocking_rules_to_generate_predictions=BLOCKING_RULES,
        comparisons=[
            NAME_COMPARISON,
            cl.DateOfBirthComparison("dob", input_is_string=True),
            cl.PostcodeComparison("postcode_fake"),
            cl.ExactMatch("birth_place").configure(term_frequency_adjustments=True),
            cl.ExactMatch("occupation").configure(term_frequency_adjustments=True),
        ],
        retain_intermediate_calculation_columns=True,
        additional_columns_to_retain=["gender", "cluster"],
    )


def fit(df: pd.DataFrame) -> Linker:
    linker = Linker(df, settings(), db_api=DuckDBAPI())
    linker.training.estimate_probability_two_random_records_match(LAMBDA_RULES, recall=LAMBDA_RECALL)
    linker.training.estimate_u_using_random_sampling(max_pairs=U_MAX_PAIRS, seed=SEED)
    for rule in EM_RULES:
        linker.training.estimate_parameters_using_expectation_maximisation(rule, estimate_without_term_frequencies=True)
    return linker


def main():
    t = time.time()
    df = load_data()
    linker = fit(df)
    linker.misc.save_model_to_json(str(RESULTS / "model.json"), overwrite=True)
    pairs = linker.inference.predict().as_pandas_dataframe()
    keep = ["unique_id_l", "unique_id_r", "match_weight", "match_probability", "cluster_l", "cluster_r",
            "gender_l", "gender_r"] + [c for c in pairs.columns if c.startswith(("gamma_", "bf_", "tf_"))]
    # DuckDB returns rows in a nondeterministic order; sort so that sampling is exactly reproducible
    pairs[keep].sort_values(["unique_id_l", "unique_id_r"], kind="stable").reset_index(drop=True).to_parquet(PAIRS)
    print(f"scored {len(pairs):,} candidate pairs in {time.time() - t:.0f}s -> {PAIRS}")


if __name__ == "__main__":
    main()
