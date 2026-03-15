from __future__ import annotations

import argparse
import difflib
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import challenge_data_prep as prep


CHALLENGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = CHALLENGE_DIR / "data-Kaggle"
DEFAULT_BARTTORVIK_DATA_DIR = CHALLENGE_DIR / "data-BartTorvik"
RANDOM_STATE = 27

# Locked from iter_2.ipynb after backward elimination and validation.
FINAL_FEATURES = [
    "seed_diff",
    "diff_kp_kadj_em",
    "diff_kp_barthag",
    "diff_tr_sos_rating",
    "diff_kp_badj_em_rank",
    "diff_res_b_power",
    "diff_kp_kadj_em_rank",
    "diff_tr_lo",
    "diff_tr_sos_last",
    "diff_res_r_score",
    "diff_res_wab_rank",
    "diff_kp_kadj_o",
]


def normalize_team_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def make_pipeline(feature_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)),
        ]
    )


def resolve_team_row(team_features: pd.DataFrame, year: int, team_name: str) -> pd.Series:
    season_rows = team_features.loc[team_features["year"] == year].copy()
    if season_rows.empty:
        raise ValueError(f"No team features found for year {year}.")

    lookup = season_rows.assign(_normalized_team=season_rows["team"].map(normalize_team_name))
    normalized_name = normalize_team_name(team_name)
    matches = lookup.loc[lookup["_normalized_team"] == normalized_name]
    if len(matches) == 1:
        return matches.iloc[0]

    available_names = sorted(season_rows["team"].dropna().astype(str).unique().tolist())
    suggestions = difflib.get_close_matches(team_name, available_names, n=5, cutoff=0.5)
    suggestion_text = f" Close matches: {suggestions}" if suggestions else ""
    raise ValueError(f"Could not uniquely match team '{team_name}' for year {year}.{suggestion_text}")


def build_prediction_row(team_features: pd.DataFrame, year: int, team_name: str, opponent_name: str) -> pd.DataFrame:
    team_row = resolve_team_row(team_features=team_features, year=year, team_name=team_name)
    opp_row = resolve_team_row(team_features=team_features, year=year, team_name=opponent_name)

    if int(team_row["team_no"]) == int(opp_row["team_no"]):
        raise ValueError("Team and opponent resolved to the same school.")

    base = {
        "year": year,
        "team_no": int(team_row["team_no"]),
        "team": team_row["team"],
        "seed": team_row["seed"],
        "seed_num": team_row["seed_num"],
        "opp_team_no": int(opp_row["team_no"]),
        "opp_team": opp_row["team"],
        "opp_seed": opp_row["seed"],
        "opp_seed_num": opp_row["seed_num"],
    }

    team_feature_values = {
        column: team_row[column]
        for column in team_features.columns
        if column not in prep.TEAM_KEYS + ["team", "seed", "seed_num"]
    }
    opp_feature_values = {
        f"opp_{column}": opp_row[column]
        for column in team_features.columns
        if column not in prep.TEAM_KEYS + ["team", "seed", "seed_num"]
    }

    prediction_row = pd.DataFrame([{**base, **team_feature_values, **opp_feature_values}])
    prediction_row = prep.add_starter_calculated_columns(prediction_row)
    prediction_row = prep.add_matchup_interaction_columns(prediction_row)
    prediction_row = prep.add_relative_strength_features(prediction_row)
    prediction_row = prep.add_consensus_rating_features(prediction_row)
    prediction_row = prep.add_rank_and_volatility_features(prediction_row)
    prediction_row = prep.add_auto_difference_columns(prediction_row)
    prediction_row = prep.drop_leakage_columns(prediction_row)
    return prediction_row


def train_model(kaggle_data_dir: Path, barttorvik_data_dir: Path, prediction_year: int) -> tuple[Pipeline, pd.DataFrame]:
    bundle = prep.build_modeling_table(
        kaggle_data_dir=kaggle_data_dir,
        barttorvik_data_dir=barttorvik_data_dir,
        include_auto_diffs=True,
    )
    modeling_table = bundle.modeling_table.copy()
    training_rows = modeling_table.loc[modeling_table["year"] < prediction_year].copy()
    if training_rows.empty:
        raise ValueError(f"No historical rows available before year {prediction_year}.")

    missing_features = [column for column in FINAL_FEATURES if column not in training_rows.columns]
    if missing_features:
        raise KeyError(f"Missing required final features in training table: {missing_features}")

    model = make_pipeline(FINAL_FEATURES)
    model.fit(training_rows, training_rows["win"])
    return model, bundle.team_features.copy()


def list_teams(kaggle_data_dir: Path, barttorvik_data_dir: Path, year: int) -> None:
    team_features, _ = prep.build_team_feature_table(
        kaggle_data_dir=kaggle_data_dir,
        barttorvik_data_dir=barttorvik_data_dir,
    )
    season_teams = (
        team_features.loc[team_features["year"] == year, "team"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
    if not season_teams:
        raise ValueError(f"No teams found for year {year}.")
    for name in season_teams:
        print(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a March Madness matchup from team names.")
    parser.add_argument("--year", type=int, help="Tournament year to predict, such as 2025.")
    parser.add_argument("--team", type=str, help="Team name as it appears in the data.")
    parser.add_argument("--opponent", type=str, help="Opponent name as it appears in the data.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing Kaggle challenge CSV files.")
    parser.add_argument(
        "--barttorvik-data-dir",
        type=Path,
        default=DEFAULT_BARTTORVIK_DATA_DIR,
        help="Directory containing BartTorvik season files.",
    )
    parser.add_argument("--list-teams", action="store_true", help="List available teams for the selected year.")
    args = parser.parse_args()

    if args.year is None:
        parser.error("--year is required.")

    if args.list_teams:
        list_teams(
            kaggle_data_dir=args.data_dir,
            barttorvik_data_dir=args.barttorvik_data_dir,
            year=args.year,
        )
        return

    if not args.team or not args.opponent:
        parser.error("--team and --opponent are required unless --list-teams is used.")

    model, team_features = train_model(
        kaggle_data_dir=args.data_dir,
        barttorvik_data_dir=args.barttorvik_data_dir,
        prediction_year=args.year,
    )
    matchup_row = build_prediction_row(
        team_features=team_features,
        year=args.year,
        team_name=args.team,
        opponent_name=args.opponent,
    )

    missing_features = [column for column in FINAL_FEATURES if column not in matchup_row.columns]
    if missing_features:
        raise KeyError(f"Missing required final features in prediction row: {missing_features}")

    team_win_probability = float(model.predict_proba(matchup_row)[:, 1][0])
    predicted_class = int(team_win_probability >= 0.5)
    predicted_winner = args.team if predicted_class == 1 else args.opponent

    print(f"Year: {args.year}")
    print(f"Matchup: {args.team} vs {args.opponent}")
    print(f"Predicted winner: {predicted_winner}")
    print(f"{args.team} win probability: {team_win_probability:.3f}")
    print(f"{args.opponent} win probability: {1.0 - team_win_probability:.3f}")
    print("Features used:")
    for feature in FINAL_FEATURES:
        print(f"  {feature}: {matchup_row.iloc[0][feature]}")


if __name__ == "__main__":
    main()
