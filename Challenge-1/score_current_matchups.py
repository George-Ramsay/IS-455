from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import matchup_cli_2026 as team_cli
import predict_matchup as predictor


CHALLENGE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = CHALLENGE_DIR / "data" / "current_matchups.csv"
DEFAULT_YEAR = 2026


def validate_input_columns(df: pd.DataFrame) -> None:
    required_columns = ["away_team", "home_team"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Input CSV is missing required columns: {missing_columns}")


def resolve_team_value(raw_name: object, catalog: team_cli.TeamCatalog) -> dict[str, object]:
    cleaned_name = "" if pd.isna(raw_name) else str(raw_name).strip()
    resolved, suggestions, normalized_name = catalog.resolve(cleaned_name)

    if resolved is None:
        return {
            "raw_name": cleaned_name,
            "normalized_name": normalized_name,
            "canonical_name": None,
            "sources": None,
            "matched": False,
            "suggestions": " | ".join(suggestions),
        }

    return {
        "raw_name": cleaned_name,
        "normalized_name": normalized_name,
        "canonical_name": resolved.canonical_name,
        "sources": team_cli.source_summary(resolved),
        "matched": True,
        "suggestions": "",
    }


def score_matchup_row(
    model,
    team_features: pd.DataFrame,
    year: int,
    away_team_name: str,
    home_team_name: str,
) -> dict[str, object]:
    matchup_row = predictor.build_prediction_row(
        team_features=team_features,
        year=year,
        team_name=away_team_name,
        opponent_name=home_team_name,
    )
    away_win_probability = float(model.predict_proba(matchup_row)[:, 1][0])
    home_win_probability = 1.0 - away_win_probability
    predicted_winner = away_team_name if away_win_probability >= 0.5 else home_team_name

    return {
        "predicted_winner": predicted_winner,
        "away_win_probability": away_win_probability,
        "home_win_probability": home_win_probability,
        "score_status": "scored",
        "score_message": "",
    }


def enrich_matchups(input_df: pd.DataFrame, year: int) -> pd.DataFrame:
    catalog = team_cli.build_team_catalog(year)
    missing_inputs = team_cli.collect_missing_2026_prediction_inputs()
    scoring_ready = len(missing_inputs) == 0

    model = None
    team_features = None
    if scoring_ready:
        model, team_features = predictor.train_model(
            kaggle_data_dir=predictor.DEFAULT_DATA_DIR,
            barttorvik_data_dir=predictor.DEFAULT_BARTTORVIK_DATA_DIR,
            teamrankings_data_dir=predictor.DEFAULT_TEAMRANKINGS_DATA_DIR,
            prediction_year=year,
        )

    rows: list[dict[str, object]] = []
    readiness_message = "ready" if scoring_ready else " | ".join(missing_inputs)

    for _, row in input_df.iterrows():
        away_resolution = resolve_team_value(row.get("away_team"), catalog)
        home_resolution = resolve_team_value(row.get("home_team"), catalog)

        result_row = row.to_dict()
        result_row.update(
            {
                "season_year": year,
                "away_team_local": away_resolution["canonical_name"],
                "home_team_local": home_resolution["canonical_name"],
                "away_team_matched": away_resolution["matched"],
                "home_team_matched": home_resolution["matched"],
                "away_team_sources": away_resolution["sources"],
                "home_team_sources": home_resolution["sources"],
                "away_team_suggestions": away_resolution["suggestions"],
                "home_team_suggestions": home_resolution["suggestions"],
                "prediction_ready": scoring_ready,
                "prediction_ready_message": readiness_message,
                "predicted_winner": None,
                "away_win_probability": None,
                "home_win_probability": None,
                "score_status": None,
                "score_message": None,
            }
        )

        if not away_resolution["matched"] or not home_resolution["matched"]:
            result_row["score_status"] = "unresolved_team_name"
            result_row["score_message"] = "Resolve both ESPN team names to local canonical names before scoring."
            rows.append(result_row)
            continue

        if not scoring_ready:
            result_row["score_status"] = "prediction_inputs_missing"
            result_row["score_message"] = readiness_message
            rows.append(result_row)
            continue

        try:
            score_result = score_matchup_row(
                model=model,
                team_features=team_features,
                year=year,
                away_team_name=str(away_resolution["canonical_name"]),
                home_team_name=str(home_resolution["canonical_name"]),
            )
            result_row.update(score_result)
        except Exception as exc:
            result_row["score_status"] = "scoring_error"
            result_row["score_message"] = str(exc)

        rows.append(result_row)

    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve ESPN matchup rows to local team names and score them when model inputs exist."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV created by espn_matchups.py containing at least away_team and home_team columns.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to <input_stem>_scored.csv beside the input file.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"Season year to score. Only {DEFAULT_YEAR} is currently supported cleanly.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.year != DEFAULT_YEAR:
        parser.error(f"--year must be {DEFAULT_YEAR} for this iteration.")

    input_df = pd.read_csv(args.input_csv)
    validate_input_columns(input_df)

    scored_df = enrich_matchups(input_df=input_df, year=args.year)
    output_csv = args.output_csv or args.input_csv.with_name(f"{args.input_csv.stem}_scored.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_csv, index=False)

    total_rows = len(scored_df)
    matched_rows = int((scored_df["away_team_matched"] & scored_df["home_team_matched"]).sum())
    scored_rows = int((scored_df["score_status"] == "scored").sum())

    print(f"Input rows: {total_rows}")
    print(f"Rows with both teams resolved: {matched_rows}")
    print(f"Rows fully scored: {scored_rows}")
    print(f"Wrote scored matchup file: {output_csv}")

    unresolved_rows = scored_df.loc[scored_df["score_status"] == "unresolved_team_name"]
    if not unresolved_rows.empty:
        print()
        print("Unresolved rows:")
        preview_columns = [
            column
            for column in [
                "away_team",
                "home_team",
                "away_team_suggestions",
                "home_team_suggestions",
            ]
            if column in unresolved_rows.columns
        ]
        print(unresolved_rows[preview_columns].head(10).to_string(index=False))

    blocked_rows = scored_df.loc[scored_df["score_status"] == "prediction_inputs_missing"]
    if not blocked_rows.empty:
        print()
        print("Scoring is still blocked by missing current-year model inputs.")
        print(blocked_rows["prediction_ready_message"].iloc[0])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
