from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


CHALLENGE_DIR = Path(__file__).resolve().parent
DEFAULT_KAGGLE_DATA_DIR = CHALLENGE_DIR / "data-Kaggle"
DEFAULT_BARTTORVIK_DATA_DIR = CHALLENGE_DIR / "data-BartTorvik"
DEFAULT_OUTPUT_DIR = CHALLENGE_DIR / "prepared-data"

TEAM_KEYS = ["year", "team_no"]
CONFERENCE_KEYS = ["year", "conf_id"]
MATCHUP_ROW_KEYS = ["year", "by_year_no", "team_no"]
GAME_KEYS = ["year", "current_round", "game_index", "game_id", "slot_in_game"]

TEAM_BASE_COLUMNS = ["year", "team_no", "team", "seed", "seed_num"]
AUTO_DIFF_EXCLUDE = {
    "by_year_no",
    "current_round",
    "game_index",
    "game_id",
    "round",
    "score",
    "seed",
    "seed_num",
    "slot_in_game",
    "team_no",
    "win",
    "year",
}

LEAKAGE_COLUMNS = [
    "round",
    "current_round",
    "score",
    "opp_round",
    "opp_current_round",
    "opp_score",
]


@dataclass(frozen=True)
class SourceSpec:
    name: str
    filename: str
    prefix: str
    join_keys: tuple[str, ...] = ("year", "team_no")
    default: bool = False
    extra_drop_columns: tuple[str, ...] = ()
    data_group: str = "kaggle"


SOURCE_SPECS = {
    "kenpom_barttorvik": SourceSpec(
        name="kenpom_barttorvik",
        filename="KenPom Barttorvik.csv",
        prefix="kp",
        default=True,
        extra_drop_columns=("round", "seed", "team", "team_id", "quad_no", "quad_id"),
    ),
    "teamrankings": SourceSpec(
        name="teamrankings",
        filename="TeamRankings.csv",
        prefix="tr",
        default=True,
        extra_drop_columns=("round", "seed", "team"),
    ),
    "resumes": SourceSpec(
        name="resumes",
        filename="Resumes.csv",
        prefix="res",
        default=True,
        extra_drop_columns=("round", "seed", "team"),
    ),
    "evanmiya": SourceSpec(
        name="evanmiya",
        filename="EvanMiya.csv",
        prefix="em",
        default=False,
        extra_drop_columns=("round", "seed", "team"),
    ),
    "shooting_splits": SourceSpec(
        name="shooting_splits",
        filename="Shooting Splits.csv",
        prefix="shot",
        default=False,
        extra_drop_columns=("team", "team_id", "conf"),
    ),
    "kenpom_preseason": SourceSpec(
        name="kenpom_preseason",
        filename="KenPom Preseason.csv",
        prefix="pre",
        default=False,
        extra_drop_columns=("round", "seed", "team"),
    ),
    "rppf_ratings": SourceSpec(
        name="rppf_ratings",
        filename="RPPF Ratings.csv",
        prefix="rppf",
        default=False,
        extra_drop_columns=("round", "seed", "team"),
    ),
    "barttorvik_neutral": SourceSpec(
        name="barttorvik_neutral",
        filename="Barttorvik Neutral.csv",
        prefix="kpn",
        default=False,
        extra_drop_columns=("round", "seed", "team", "team_id"),
    ),
    "teamrankings_neutral": SourceSpec(
        name="teamrankings_neutral",
        filename="TeamRankings Neutral.csv",
        prefix="trn",
        default=False,
        extra_drop_columns=("round", "seed", "team"),
    ),
    "barttorvik_team_results": SourceSpec(
        name="barttorvik_team_results",
        filename="*_team_results.csv",
        prefix="bt",
        default=False,
        extra_drop_columns=("team", "conf", "record", "con_rec"),
        data_group="barttorvik",
    ),
    "barttorvik_player_aggregates": SourceSpec(
        name="barttorvik_player_aggregates",
        filename="*_trank_data.csv",
        prefix="btp",
        default=False,
        extra_drop_columns=(),
        data_group="barttorvik",
    ),
}

CONFERENCE_SOURCE_SPECS = {
    "conference_stats": SourceSpec(
        name="conference_stats",
        filename="Conference Stats.csv",
        prefix="conf",
        join_keys=("year", "conf_id"),
        default=False,
        extra_drop_columns=("conf",),
    ),
    "conference_stats_neutral": SourceSpec(
        name="conference_stats_neutral",
        filename="Conference Stats Neutral.csv",
        prefix="confn",
        join_keys=("year", "conf_id"),
        default=False,
        extra_drop_columns=("conf",),
    ),
}


@dataclass
class PreparedDataBundle:
    team_features: pd.DataFrame
    matchup_rows: pd.DataFrame
    modeling_table: pd.DataFrame
    merge_report: dict


def normalize_column_name(name: str) -> str:
    text = str(name).strip().lower()
    replacements = {
        "%": " pct ",
        "#": " num ",
        "&": " and ",
        "/": " ",
        "(": " ",
        ")": " ",
        "+": " plus ",
        "-": " ",
        ".": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def make_unique(names: Iterable[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique_names: list[str] = []
    for name in names:
        base_name = name
        current_count = counts.get(base_name, 0)
        if current_count == 0:
            unique_names.append(base_name)
        else:
            unique_names.append(f"{base_name}_{current_count + 1}")
        counts[base_name] = current_count + 1
    return unique_names


def read_csv_normalized(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = make_unique(normalize_column_name(column) for column in df.columns)
    object_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        df[column] = df[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    return df


def normalize_team_key(name: str) -> str:
    text = str(name).strip().lower()
    alias_map = {
        "college of charleston": "charleston",
        "north carolina st.": "ncstate",
        "north carolina state": "ncstate",
        "n.c. state": "ncstate",
        "louisiana lafayette": "louisiana",
        "louisiana-lafayette": "louisiana",
        "saint marys": "stmarys",
        "saint mary's": "stmarys",
        "st mary's": "stmarys",
    }
    text = alias_map.get(text, text)
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def weighted_average(series: pd.Series, weights: pd.Series) -> float:
    valid_mask = series.notna() & weights.notna() & (weights > 0)
    if not valid_mask.any():
        return float(series.mean()) if series.notna().any() else float("nan")
    return float((series.loc[valid_mask] * weights.loc[valid_mask]).sum() / weights.loc[valid_mask].sum())


def ensure_columns(df: pd.DataFrame, required_columns: Iterable[str], source_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"{source_name} is missing required columns: {missing_columns}")


def validate_unique_keys(df: pd.DataFrame, keys: list[str], source_name: str) -> None:
    duplicate_mask = df.duplicated(keys, keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, keys].head(10).to_dict("records")
        raise ValueError(f"{source_name} has duplicate rows on keys {keys}. Examples: {duplicates}")


def coerce_seed_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def prefix_columns(df: pd.DataFrame, prefix: str, join_keys: list[str]) -> pd.DataFrame:
    rename_map = {
        column: f"{prefix}_{column}"
        for column in df.columns
        if column not in join_keys
    }
    return df.rename(columns=rename_map)


def resolve_source_names(selected_sources: list[str] | None, include_optional_sources: bool) -> list[str]:
    if selected_sources:
        unknown_sources = [name for name in selected_sources if name not in SOURCE_SPECS]
        if unknown_sources:
            raise KeyError(f"Unknown source names: {unknown_sources}")
        return selected_sources

    resolved_sources = [name for name, spec in SOURCE_SPECS.items() if spec.default]
    if include_optional_sources:
        resolved_sources.extend(name for name, spec in SOURCE_SPECS.items() if not spec.default)
    return resolved_sources


def resolve_conference_source_names(include_optional_sources: bool) -> list[str]:
    if not include_optional_sources:
        return []
    return list(CONFERENCE_SOURCE_SPECS)


def load_matchups(kaggle_data_dir: Path) -> pd.DataFrame:
    matchups = read_csv_normalized(kaggle_data_dir / "Tournament Matchups.csv")
    ensure_columns(
        matchups,
        ["year", "by_year_no", "team_no", "team", "seed", "round", "current_round", "score"],
        "Tournament Matchups.csv",
    )
    return matchups


def build_base_team_frame(matchups: pd.DataFrame) -> pd.DataFrame:
    consistency = (
        matchups.groupby(TEAM_KEYS)
        .agg(team_nunique=("team", "nunique"), seed_nunique=("seed", "nunique"))
        .reset_index()
    )
    inconsistent = consistency.loc[
        (consistency["team_nunique"] > 1) | (consistency["seed_nunique"] > 1)
    ]
    if not inconsistent.empty:
        examples = inconsistent.head(10).to_dict("records")
        raise ValueError(f"Team metadata is inconsistent within YEAR + TEAM NO. Examples: {examples}")

    base = (
        matchups.sort_values(["year", "team_no", "current_round", "by_year_no"], ascending=[True, True, False, False])
        .groupby(TEAM_KEYS, as_index=False)
        .agg({"team": "first", "seed": "first"})
        .sort_values(TEAM_KEYS)
        .reset_index(drop=True)
    )
    validate_unique_keys(base, TEAM_KEYS, "matchup team base")
    base["seed_num"] = coerce_seed_numeric(base["seed"])
    base["team_key"] = base["team"].map(normalize_team_key)
    return base


def load_barttorvik_team_results(barttorvik_data_dir: Path, team_base: pd.DataFrame) -> pd.DataFrame:
    raw_columns = [
        "rank",
        "team",
        "conf",
        "record",
        "adjoe",
        "oe Rank",
        "adjde",
        "de Rank",
        "barthag",
        "rank_1",
        "proj. W",
        "Proj. L",
        "Pro Con W",
        "Pro Con L",
        "Con Rec.",
        "sos",
        "ncsos",
        "consos",
        "Proj. SOS",
        "Proj. Noncon SOS",
        "Proj. Con SOS",
        "elite SOS",
        "elite noncon SOS",
        "Opp OE",
        "Opp DE",
        "Opp Proj. OE",
        "Opp Proj DE",
        "Con Adj OE",
        "Con Adj DE",
        "Qual O",
        "Qual D",
        "Qual Barthag",
        "Qual Games",
        "FUN",
        "ConPF",
        "ConPA",
        "ConPoss",
        "ConOE",
        "ConDE",
        "ConSOSRemain",
        "Conf Win%",
        "WAB",
        "WAB Rk",
        "Fun Rk",
        "adjt",
    ]
    normalized_columns = make_unique(normalize_column_name(column) for column in raw_columns)
    season_frames: list[pd.DataFrame] = []
    for path in sorted(barttorvik_data_dir.glob("*_team_results.csv")):
        year_text = path.stem.split("_", 1)[0]
        if not year_text.isdigit():
            continue
        season_frame = pd.read_csv(path, header=0, names=raw_columns)
        season_frame.columns = normalized_columns
        object_columns = season_frame.select_dtypes(include=["object", "string"]).columns
        for column in object_columns:
            season_frame[column] = season_frame[column].map(lambda value: value.strip() if isinstance(value, str) else value)
        season_frame["year"] = int(year_text)
        season_frames.append(season_frame)

    if not season_frames:
        raise FileNotFoundError(f"No BartTorvik team result files found in {barttorvik_data_dir}")

    source = pd.concat(season_frames, ignore_index=True)
    source["team_key"] = source["team"].map(normalize_team_key)
    source = source.merge(
        team_base[["year", "team_no", "team_key"]],
        on=["year", "team_key"],
        how="inner",
        validate="m:1",
    )
    validate_unique_keys(source, TEAM_KEYS, "BartTorvik team results")
    if "adjoe" in source.columns and "adjde" in source.columns:
        source["bt_em"] = source["adjoe"] - source["adjde"]
    if "record" in source.columns:
        record_parts = source["record"].astype(str).str.extract(r"(?P<wins>\d+)\s*-\s*(?P<losses>\d+)")
        source["record_wins"] = pd.to_numeric(record_parts["wins"], errors="coerce")
        source["record_losses"] = pd.to_numeric(record_parts["losses"], errors="coerce")
        source["record_win_pct"] = source["record_wins"] / (source["record_wins"] + source["record_losses"])
    return source


def load_barttorvik_player_aggregates(barttorvik_data_dir: Path, team_base: pd.DataFrame) -> pd.DataFrame:
    raw_columns = [
        "player_name",
        "team",
        "conf",
        "GP",
        "Min_per",
        "ORtg",
        "usg",
        "eFG",
        "TS_per",
        "ORB_per",
        "DRB_per",
        "AST_per",
        "TO_per",
        "FTM",
        "FTA",
        "FT_per",
        "twoPM",
        "twoPA",
        "twoP_per",
        "TPM",
        "TPA",
        "TP_per",
        "blk_per",
        "stl_per",
        "ftr",
        "yr",
        "ht",
        "num",
        "porpag",
        "adjoe",
        "pfr",
        "year",
        "pid",
        "type",
        "Rec Rank",
        " ast/tov",
        " rimmade",
        " rimmade+ri",
        " midmade",
        " midmade+m",
        " rimmade/(ri",
        " midmade/(m",
        " dunksmade",
        " dunksmiss+",
        " dunksmade/",
        " pick",
        " drtg",
        "adrtg",
        " dporpag",
        " stops",
        " bpm",
        " obpm",
        " dbpm",
        " gbpm",
        "mp",
        "ogbpm",
        "dgbpm",
        "oreb",
        "dreb",
        "treb",
        "ast",
        "stl",
        "blk",
        "pts",
        "role",
        "3p/100?",
        "dob",
    ]
    normalized_columns = make_unique(normalize_column_name(column) for column in raw_columns)
    season_frames: list[pd.DataFrame] = []
    for path in sorted(barttorvik_data_dir.glob("*_trank_data.csv")):
        year_text = path.stem.split("_", 1)[0]
        if not year_text.isdigit():
            continue
        season_frame = pd.read_csv(path, header=None, names=normalized_columns)
        season_frame["year"] = int(year_text)
        season_frames.append(season_frame)

    if not season_frames:
        raise FileNotFoundError(f"No BartTorvik player files found in {barttorvik_data_dir}")

    players = pd.concat(season_frames, ignore_index=True)
    players["team_key"] = players["team"].map(normalize_team_key)
    players = players.merge(
        team_base[["year", "team_no", "team_key"]],
        on=["year", "team_key"],
        how="inner",
        validate="m:1",
    )

    numeric_columns = [
        "gp",
        "min_per",
        "ortg",
        "usg",
        "efg",
        "ts_per",
        "orb_per",
        "drb_per",
        "ast_per",
        "to_per",
        "ftm",
        "fta",
        "ft_per",
        "twopm",
        "twopa",
        "twop_per",
        "tpm",
        "tpa",
        "tp_per",
        "blk_per",
        "stl_per",
        "ftr",
        "porpag",
        "adjoe",
        "pfr",
        "rec_rank",
        "ast_tov",
        "rimmade",
        "rimmade_plus_ri",
        "midmade",
        "midmade_plus_m",
        "rimmade_ri",
        "midmade_m",
        "dunksmade",
        "dunksmiss_plus",
        "dunksmade_2",
        "pick",
        "drtg",
        "adrtg",
        "dporpag",
        "stops",
        "bpm",
        "obpm",
        "dbpm",
        "gbpm",
        "mp",
        "ogbpm",
        "dgbpm",
        "oreb",
        "dreb",
        "treb",
        "ast",
        "stl",
        "blk",
        "pts",
        "3p_100",
    ]
    for column in numeric_columns:
        if column in players.columns:
            players[column] = pd.to_numeric(players[column], errors="coerce")

    players["mp_weight"] = players["mp"].fillna(0.0) if "mp" in players.columns else 0.0
    players["rotation_flag"] = (players["min_per"].fillna(0.0) >= 10.0).astype(int)
    players["upperclass_flag"] = players["yr"].astype(str).str.lower().str.startswith(("jr", "sr", "gr")).astype(int)

    grouped_rows: list[dict] = []
    for (year, team_no), group in players.groupby(TEAM_KEYS, dropna=False):
        group = group.copy()
        group = group.sort_values("porpag", ascending=False, na_position="last")
        total_mp = float(group["mp_weight"].sum())
        grouped_rows.append(
            {
                "year": int(year),
                "team_no": int(team_no),
                "player_count": int(len(group)),
                "rotation_player_count": int(group["rotation_flag"].sum()),
                "upperclass_share": float(group["upperclass_flag"].mean()),
                "minutes_total": total_mp,
                "weighted_porpag": weighted_average(group["porpag"], group["mp_weight"]),
                "weighted_adjoe": weighted_average(group["adjoe"], group["mp_weight"]),
                "weighted_ortg": weighted_average(group["ortg"], group["mp_weight"]),
                "weighted_usg": weighted_average(group["usg"], group["mp_weight"]),
                "weighted_efg": weighted_average(group["efg"], group["mp_weight"]),
                "weighted_ts_per": weighted_average(group["ts_per"], group["mp_weight"]),
                "weighted_bpm": weighted_average(group["bpm"], group["mp_weight"]),
                "weighted_obpm": weighted_average(group["obpm"], group["mp_weight"]),
                "weighted_dbpm": weighted_average(group["dbpm"], group["mp_weight"]),
                "weighted_gbpm": weighted_average(group["gbpm"], group["mp_weight"]),
                "weighted_dporpag": weighted_average(group["dporpag"], group["mp_weight"]),
                "top_1_porpag": float(group["porpag"].head(1).sum()),
                "top_3_porpag_sum": float(group["porpag"].head(3).sum()),
                "top_1_bpm": float(group["bpm"].head(1).sum()),
                "top_3_bpm_sum": float(group["bpm"].head(3).sum()),
                "top_3_minutes_share": float(group["mp_weight"].head(3).sum() / total_mp) if total_mp > 0 else float("nan"),
            }
        )

    aggregated = pd.DataFrame(grouped_rows)
    validate_unique_keys(aggregated, TEAM_KEYS, "BartTorvik player aggregates")
    return aggregated


def prepare_team_source(
    kaggle_data_dir: Path,
    barttorvik_data_dir: Path,
    spec: SourceSpec,
    team_base: pd.DataFrame,
) -> pd.DataFrame:
    if spec.data_group == "barttorvik" and spec.name == "barttorvik_team_results":
        source = load_barttorvik_team_results(barttorvik_data_dir, team_base)
        source_name = "BartTorvik yearly team results"
    elif spec.data_group == "barttorvik" and spec.name == "barttorvik_player_aggregates":
        source = load_barttorvik_player_aggregates(barttorvik_data_dir, team_base)
        source_name = "BartTorvik yearly player aggregates"
    else:
        source_dir = kaggle_data_dir if spec.data_group == "kaggle" else barttorvik_data_dir
        source = read_csv_normalized(source_dir / spec.filename)
        source_name = spec.filename

    ensure_columns(source, spec.join_keys, source_name)
    validate_unique_keys(source, list(spec.join_keys), source_name)
    drop_columns = set(spec.extra_drop_columns)
    prepared = source.drop(columns=[column for column in drop_columns if column in source.columns])
    prepared = prefix_columns(prepared, spec.prefix, list(spec.join_keys))
    return prepared


def build_team_feature_table(
    kaggle_data_dir: Path,
    barttorvik_data_dir: Path,
    selected_sources: list[str] | None = None,
    include_optional_sources: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    matchups = load_matchups(kaggle_data_dir)
    team_features = build_base_team_frame(matchups)
    base_keys = team_features[TEAM_KEYS].copy()
    merge_report: list[dict] = []

    for source_name in resolve_source_names(selected_sources, include_optional_sources):
        spec = SOURCE_SPECS[source_name]
        prepared = prepare_team_source(kaggle_data_dir, barttorvik_data_dir, spec, team_features)
        matched = base_keys.merge(prepared[TEAM_KEYS], on=TEAM_KEYS, how="left", indicator=True)
        coverage = float((matched["_merge"] == "both").mean())

        merge_report.append(
            {
                "source_name": source_name,
                "filename": spec.filename,
                "prefix": spec.prefix,
                "rows": int(len(prepared)),
                "columns_added": int(len(prepared.columns) - len(spec.join_keys)),
                "coverage_pct": round(coverage * 100, 2),
            }
        )
        team_features = team_features.merge(prepared, on=TEAM_KEYS, how="left", validate="1:1")

    if "kp_conf_id" in team_features.columns:
        conference_keys = team_features[TEAM_KEYS + ["kp_conf_id"]].rename(columns={"kp_conf_id": "conf_id"})
        for source_name in resolve_conference_source_names(include_optional_sources):
            spec = CONFERENCE_SOURCE_SPECS[source_name]
            prepared = prepare_team_source(kaggle_data_dir, barttorvik_data_dir, spec, team_features)
            matched = conference_keys.merge(prepared[list(spec.join_keys)], on=list(spec.join_keys), how="left", indicator=True)
            coverage = float((matched["_merge"] == "both").mean())

            merge_report.append(
                {
                    "source_name": source_name,
                    "filename": spec.filename,
                    "prefix": spec.prefix,
                    "rows": int(len(prepared)),
                    "columns_added": int(len(prepared.columns) - len(spec.join_keys)),
                    "coverage_pct": round(coverage * 100, 2),
                }
            )
            team_features = team_features.merge(
                conference_keys.merge(prepared, on=list(spec.join_keys), how="left").drop(columns=["conf_id"]),
                on=TEAM_KEYS,
                how="left",
                validate="1:1",
            )

    if "team_key" in team_features.columns:
        team_features = team_features.drop(columns=["team_key"])
    return team_features, merge_report


def build_matchup_rows(kaggle_data_dir: Path) -> tuple[pd.DataFrame, dict]:
    matchups = load_matchups(kaggle_data_dir)
    matchups = matchups.sort_values(["year", "current_round", "by_year_no"], ascending=[True, True, False]).copy()
    matchup_order = matchups.groupby(["year", "current_round"]).cumcount()
    matchups["game_index"] = matchup_order // 2
    matchups["slot_in_game"] = matchup_order % 2
    matchups["game_id"] = (
        matchups["year"].astype(int).astype(str)
        + "_"
        + matchups["current_round"].astype(int).astype(str)
        + "_"
        + matchups["game_index"].astype(int).astype(str)
    )
    matchups["seed_num"] = coerce_seed_numeric(matchups["seed"])

    pair_sizes = (
        matchups.groupby(["year", "current_round", "game_index"])
        .size()
        .rename("rows_per_game")
        .reset_index()
    )
    if not (pair_sizes["rows_per_game"] == 2).all():
        bad_groups = pair_sizes.loc[pair_sizes["rows_per_game"] != 2].head(10).to_dict("records")
        raise ValueError(f"Unexpected matchup grouping sizes. Examples: {bad_groups}")

    locations = read_csv_normalized(kaggle_data_dir / "Tournament Locations.csv")
    ensure_columns(locations, MATCHUP_ROW_KEYS, "Tournament Locations.csv")
    validate_unique_keys(locations, MATCHUP_ROW_KEYS, "Tournament Locations.csv")

    location_keep = locations.drop(
        columns=[
            column
            for column in ["team", "seed", "round", "current_round"]
            if column in locations.columns
        ]
    )
    location_keep = prefix_columns(location_keep, "loc", MATCHUP_ROW_KEYS)

    matchup_rows = matchups.merge(location_keep, on=MATCHUP_ROW_KEYS, how="left", validate="1:1")
    location_coverage = float(matchup_rows["loc_game_location"].notna().mean()) if "loc_game_location" in matchup_rows else 0.0

    location_report = {
        "matchup_rows": int(len(matchup_rows)),
        "location_rows_matched_pct": round(location_coverage * 100, 2),
        "location_rows_missing": int(len(matchup_rows) - matchup_rows["loc_game_location"].notna().sum())
        if "loc_game_location" in matchup_rows
        else int(len(matchup_rows)),
    }
    return matchup_rows, location_report


def add_starter_calculated_columns(modeling_table: pd.DataFrame) -> pd.DataFrame:
    calculations = {
        "seed_diff": ("seed_num", "opp_seed_num"),
        "distance_mi_diff": ("loc_distance_mi", "opp_loc_distance_mi"),
        "distance_km_diff": ("loc_distance_km", "opp_loc_distance_km"),
        "time_zones_crossed_diff": ("loc_time_zones_crossed_value", "opp_loc_time_zones_crossed_value"),
    }

    calculated_columns: dict[str, pd.Series] = {}
    for new_column, (left_column, right_column) in calculations.items():
        if left_column in modeling_table.columns and right_column in modeling_table.columns:
            calculated_columns[new_column] = modeling_table[left_column] - modeling_table[right_column]

    if not calculated_columns:
        return modeling_table
    return pd.concat([modeling_table, pd.DataFrame(calculated_columns, index=modeling_table.index)], axis=1)


def add_matchup_interaction_columns(modeling_table: pd.DataFrame) -> pd.DataFrame:
    calculated_columns: dict[str, pd.Series] = {}

    def add_difference_feature(new_column: str, left_column: str, right_column: str) -> None:
        if left_column in modeling_table.columns and right_column in modeling_table.columns:
            calculated_columns[new_column] = modeling_table[left_column] - modeling_table[right_column]

    def add_average_feature(new_column: str, left_column: str, right_column: str) -> None:
        if left_column in modeling_table.columns and right_column in modeling_table.columns:
            calculated_columns[new_column] = (modeling_table[left_column] + modeling_table[right_column]) / 2.0

    def add_abs_difference_feature(new_column: str, left_column: str, right_column: str) -> None:
        if left_column in modeling_table.columns and right_column in modeling_table.columns:
            calculated_columns[new_column] = (modeling_table[left_column] - modeling_table[right_column]).abs()

    add_average_feature("expected_possessions", "kp_kadj_t", "opp_kp_kadj_t")
    add_abs_difference_feature("tempo_mismatch", "kp_kadj_t", "opp_kp_kadj_t")

    add_difference_feature("offense_vs_opp_def_edge", "kp_kadj_o", "opp_kp_kadj_d")
    add_difference_feature("three_point_matchup_edge", "kp_3pt_pct", "opp_kp_3pt_pct_d")
    add_difference_feature("two_point_matchup_edge", "kp_2pt_pct", "opp_kp_2pt_pct_d")
    add_difference_feature("free_throw_matchup_edge", "kp_ftr", "opp_kp_ftrd")
    add_difference_feature("rebound_matchup_edge", "kp_oreb_pct", "opp_kp_dreb_pct")
    add_difference_feature("turnover_matchup_edge", "kp_tov_pct_d", "opp_kp_tov_pct")
    add_difference_feature("assist_matchup_edge", "kp_ast_pct", "opp_kp_op_ast_pct")

    add_difference_feature("close_two_matchup_edge", "shot_close_twos_fg_pct", "opp_shot_close_twos_fg_pct_d")
    add_difference_feature("far_two_matchup_edge", "shot_farther_twos_fg_pct", "opp_shot_farther_twos_fg_pct_d")
    add_difference_feature("three_share_matchup_edge", "shot_threes_share", "opp_shot_threes_d_share")
    add_difference_feature("three_accuracy_matchup_edge", "shot_threes_fg_pct", "opp_shot_threes_fg_pct_d")
    if all(
        column in modeling_table.columns
        for column in ["shot_dunks_share", "shot_close_twos_share", "opp_shot_dunks_d_share", "opp_shot_close_twos_d_share"]
    ):
        calculated_columns["rim_pressure_matchup_edge"] = (
            modeling_table["shot_dunks_share"]
            + modeling_table["shot_close_twos_share"]
            - modeling_table["opp_shot_dunks_d_share"]
            - modeling_table["opp_shot_close_twos_d_share"]
        )

    add_difference_feature("neutral_vs_overall_em_delta", "kpn_badj_em", "kp_badj_em")
    add_difference_feature("neutral_vs_overall_off_delta", "kpn_badj_o", "kp_badj_o")
    add_difference_feature("neutral_vs_overall_def_delta", "kpn_badj_d", "kp_badj_d")
    add_difference_feature("neutral_vs_overall_tr_delta", "trn_tr_rating", "tr_tr_rating")
    add_difference_feature("neutral_team_strength_edge", "kpn_badj_em", "opp_kpn_badj_em")
    add_difference_feature("neutral_offense_vs_opp_def_edge", "kpn_badj_o", "opp_kpn_badj_d")
    add_difference_feature("neutral_defense_vs_opp_off_edge", "opp_kpn_badj_o", "kpn_badj_d")
    add_difference_feature("neutral_tr_edge", "trn_tr_rating", "opp_trn_tr_rating")

    add_difference_feature("three_rate_matchup_edge", "kp_3ptr", "opp_kp_3ptrd")
    add_difference_feature("two_rate_matchup_edge", "kp_2ptr", "opp_kp_2ptrd")
    add_difference_feature("block_matchup_edge", "kp_blk_pct", "opp_kp_blked_pct")
    add_difference_feature("defensive_glass_matchup_edge", "kp_op_oreb_pct", "opp_kp_oreb_pct")
    add_difference_feature("efficiency_margin_matchup_edge", "kp_pppo", "opp_kp_pppd")
    add_difference_feature("barttorvik_team_strength_edge", "bt_bt_em", "opp_bt_bt_em")
    add_difference_feature("barttorvik_offense_vs_opp_def_edge", "bt_adjoe", "opp_bt_adjde")
    add_difference_feature("barttorvik_tempo_edge", "bt_adjt", "opp_bt_adjt")
    add_difference_feature("barttorvik_player_value_edge", "btp_weighted_porpag", "opp_btp_weighted_porpag")
    add_difference_feature("barttorvik_star_power_edge", "btp_top_1_porpag", "opp_btp_top_1_porpag")
    add_difference_feature("barttorvik_depth_edge", "btp_rotation_player_count", "opp_btp_rotation_player_count")
    add_difference_feature("barttorvik_upperclass_edge", "btp_upperclass_share", "opp_btp_upperclass_share")

    add_difference_feature("experience_edge", "kp_exp", "opp_kp_exp")
    add_difference_feature("talent_edge", "kp_talent", "opp_kp_talent")
    if "kp_exp" in modeling_table.columns and "kp_talent" in modeling_table.columns:
        calculated_columns["experience_talent_balance"] = modeling_table["kp_exp"] - modeling_table["kp_talent"]
        calculated_columns["experience_talent_combo"] = modeling_table["kp_exp"] * modeling_table["kp_talent"]

    add_difference_feature("trajectory_edge", "pre_kadj_em_change", "opp_pre_kadj_em_change")
    add_difference_feature("tempo_trajectory_edge", "pre_kadj_t_change", "opp_pre_kadj_t_change")
    add_difference_feature("preseason_to_current_em_gap", "kp_kadj_em", "pre_preseason_kadj_em")
    add_difference_feature("preseason_to_current_tempo_gap", "kp_kadj_t", "pre_preseason_kadj_t")

    if "loc_distance_mi" in modeling_table.columns and "opp_loc_distance_mi" in modeling_table.columns:
        calculated_columns["closer_to_site"] = (
            modeling_table["loc_distance_mi"] < modeling_table["opp_loc_distance_mi"]
        ).astype(int)
        calculated_columns["travel_distance_gap_abs"] = (
            modeling_table["loc_distance_mi"] - modeling_table["opp_loc_distance_mi"]
        ).abs()
        calculated_columns["long_trip_flag"] = (modeling_table["loc_distance_mi"] >= 1000).astype(int)
    if "loc_college_state" in modeling_table.columns and "loc_game_state" in modeling_table.columns:
        calculated_columns["same_state_site"] = (
            modeling_table["loc_college_state"] == modeling_table["loc_game_state"]
        ).astype(int)
    if "loc_college_time_zone_value" in modeling_table.columns and "loc_game_time_zone_value" in modeling_table.columns:
        calculated_columns["same_time_zone_site"] = (
            modeling_table["loc_college_time_zone_value"] == modeling_table["loc_game_time_zone_value"]
        ).astype(int)
    if "time_zones_crossed_diff" in modeling_table.columns:
        calculated_columns["travel_time_zone_gap_abs"] = modeling_table["time_zones_crossed_diff"].abs()
    if "loc_time_zones_crossed_value" in modeling_table.columns:
        calculated_columns["major_time_zone_shift"] = (
            modeling_table["loc_time_zones_crossed_value"] >= 2
        ).astype(int)

    if not calculated_columns:
        return modeling_table
    return pd.concat([modeling_table, pd.DataFrame(calculated_columns, index=modeling_table.index)], axis=1)


def add_relative_strength_features(modeling_table: pd.DataFrame) -> pd.DataFrame:
    calculated_columns: dict[str, pd.Series] = {}

    relative_pairs = {
        "team_vs_conference_em": ("kp_kadj_em", "conf_badj_em"),
        "team_vs_conference_off": ("kp_kadj_o", "conf_badj_o"),
        "team_vs_conference_def": ("kp_kadj_d", "conf_badj_d"),
        "team_vs_conference_resume": ("res_resume", "conf_wab"),
        "neutral_team_vs_conference_em": ("kpn_badj_em", "confn_badj_em"),
        "neutral_team_vs_conference_off": ("kpn_badj_o", "confn_badj_o"),
        "neutral_team_vs_conference_def": ("kpn_badj_d", "confn_badj_d"),
        "barttorvik_team_vs_conference_em": ("bt_bt_em", "conf_badj_em"),
    }
    for new_column, (left_column, right_column) in relative_pairs.items():
        if left_column in modeling_table.columns and right_column in modeling_table.columns:
            calculated_columns[new_column] = modeling_table[left_column] - modeling_table[right_column]

    matchup_relative_pairs = {
        "conference_strength_edge": ("conf_badj_em", "opp_conf_badj_em"),
        "conference_neutral_strength_edge": ("confn_badj_em", "opp_confn_badj_em"),
        "conference_off_edge": ("conf_badj_o", "opp_conf_badj_o"),
        "conference_def_edge": ("opp_conf_badj_d", "conf_badj_d"),
    }
    for new_column, (left_column, right_column) in matchup_relative_pairs.items():
        if left_column in modeling_table.columns and right_column in modeling_table.columns:
            calculated_columns[new_column] = modeling_table[left_column] - modeling_table[right_column]

    if not calculated_columns:
        return modeling_table
    return pd.concat([modeling_table, pd.DataFrame(calculated_columns, index=modeling_table.index)], axis=1)


def add_consensus_rating_features(modeling_table: pd.DataFrame) -> pd.DataFrame:
    rating_columns = [
        "kp_kadj_em",
        "tr_tr_rating",
        "res_elo",
        "res_b_power",
        "em_relative_rating",
        "rppf_rppf_rating",
        "kpn_badj_em",
        "bt_bt_em",
    ]
    available_rating_columns = [column for column in rating_columns if column in modeling_table.columns]
    if not available_rating_columns:
        return modeling_table

    team_ratings = modeling_table[available_rating_columns]
    team_consensus_mean = team_ratings.mean(axis=1)
    team_consensus_std = team_ratings.std(axis=1, ddof=0).fillna(0.0)

    calculated_columns: dict[str, pd.Series] = {
        "consensus_rating_mean": team_consensus_mean,
        "consensus_rating_std": team_consensus_std,
    }

    opponent_rating_columns = [f"opp_{column}" for column in available_rating_columns if f"opp_{column}" in modeling_table.columns]
    if opponent_rating_columns:
        opponent_ratings = modeling_table[opponent_rating_columns]
        opponent_consensus_mean = opponent_ratings.mean(axis=1)
        opponent_consensus_std = opponent_ratings.std(axis=1, ddof=0).fillna(0.0)
        calculated_columns["consensus_rating_diff"] = team_consensus_mean - opponent_consensus_mean
        calculated_columns["consensus_rating_uncertainty_diff"] = team_consensus_std - opponent_consensus_std

    if "seed_num" in modeling_table.columns:
        calculated_columns["seed_vs_consensus_gap"] = team_consensus_mean + modeling_table["seed_num"]
    if "seed_diff" in modeling_table.columns and "consensus_rating_diff" in calculated_columns:
        calculated_columns["seed_mispricing_index"] = (
            calculated_columns["consensus_rating_diff"] - modeling_table["seed_diff"]
        )

    return pd.concat([modeling_table, pd.DataFrame(calculated_columns, index=modeling_table.index)], axis=1)


def add_rank_and_volatility_features(modeling_table: pd.DataFrame) -> pd.DataFrame:
    calculated_columns: dict[str, pd.Series] = {}

    rank_disagreement_pairs = {
        "rank_disagreement_kp_vs_tr": ("kp_kadj_em_rank", "tr_tr_rank"),
        "rank_disagreement_resume_vs_kp": ("res_wab_rank", "kp_badj_em_rank"),
        "rank_disagreement_pre_vs_current": ("pre_preseason_kadj_em_rank", "kp_kadj_em_rank"),
        "rank_disagreement_tr_vs_consistency": ("tr_tr_rank", "tr_consistency_rank"),
        "rank_disagreement_kp_vs_bt": ("kp_kadj_em_rank", "bt_rank"),
    }
    for new_column, (left_column, right_column) in rank_disagreement_pairs.items():
        if left_column in modeling_table.columns and right_column in modeling_table.columns:
            calculated_columns[new_column] = (modeling_table[left_column] - modeling_table[right_column]).abs()

    volatility_pairs = {
        "tr_rating_range": ("tr_hi", "tr_lo"),
        "tr_form_gap": ("tr_tr_rating", "tr_last"),
        "tr_luck_gap": ("tr_luck_rating", "tr_tr_rating"),
        "tr_consistency_gap": ("tr_consistency_tr_rating", "tr_tr_rating"),
        "bt_projected_record_gap": ("bt_proj_w", "bt_proj_l"),
        "bt_quality_gap": ("bt_qual_o", "bt_qual_d"),
    }
    for new_column, (left_column, right_column) in volatility_pairs.items():
        if left_column in modeling_table.columns and right_column in modeling_table.columns:
            calculated_columns[new_column] = modeling_table[left_column] - modeling_table[right_column]

    resume_shape_pairs = {
        "quad_win_balance": ("res_q1_plus_q2_w", "res_q3_q4_l"),
        "elite_win_balance": ("tr_v_1_25_wins", "tr_v_1_25_loss"),
        "quality_win_balance": ("tr_v_26_50_wins", "tr_v_26_50_loss"),
    }
    for new_column, (left_column, right_column) in resume_shape_pairs.items():
        if left_column in modeling_table.columns and right_column in modeling_table.columns:
            calculated_columns[new_column] = modeling_table[left_column] - modeling_table[right_column]

    if "res_q1_w" in modeling_table.columns and "res_q2_w" in modeling_table.columns:
        calculated_columns["q1_share_of_quality_wins"] = (
            modeling_table["res_q1_w"] / (modeling_table["res_q1_w"] + modeling_table["res_q2_w"] + 1.0)
        )

    if not calculated_columns:
        return modeling_table
    return pd.concat([modeling_table, pd.DataFrame(calculated_columns, index=modeling_table.index)], axis=1)


def add_auto_difference_columns(modeling_table: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in modeling_table.columns
        if pd.api.types.is_numeric_dtype(modeling_table[column])
        and column not in AUTO_DIFF_EXCLUDE
        and not column.startswith("opp_")
    ]

    calculated_columns: dict[str, pd.Series] = {}
    for column in numeric_columns:
        opponent_column = f"opp_{column}"
        if opponent_column in modeling_table.columns:
            calculated_columns[f"diff_{column}"] = modeling_table[column] - modeling_table[opponent_column]

    if not calculated_columns:
        return modeling_table
    return pd.concat([modeling_table, pd.DataFrame(calculated_columns, index=modeling_table.index)], axis=1)


def drop_leakage_columns(modeling_table: pd.DataFrame) -> pd.DataFrame:
    leakage_columns_present = [column for column in LEAKAGE_COLUMNS if column in modeling_table.columns]
    if not leakage_columns_present:
        return modeling_table
    return modeling_table.drop(columns=leakage_columns_present)


def build_modeling_table(
    kaggle_data_dir: Path = DEFAULT_KAGGLE_DATA_DIR,
    barttorvik_data_dir: Path = DEFAULT_BARTTORVIK_DATA_DIR,
    selected_sources: list[str] | None = None,
    include_optional_sources: bool = False,
    include_auto_diffs: bool = False,
    data_dir: Path | None = None,
) -> PreparedDataBundle:
    if data_dir is not None:
        kaggle_data_dir = data_dir
    matchup_rows, location_report = build_matchup_rows(kaggle_data_dir)
    team_features, source_report = build_team_feature_table(
        kaggle_data_dir=kaggle_data_dir,
        barttorvik_data_dir=barttorvik_data_dir,
        selected_sources=selected_sources,
        include_optional_sources=include_optional_sources,
    )

    base_columns_to_drop = [
        column
        for column in TEAM_BASE_COLUMNS
        if column in team_features.columns and column not in TEAM_KEYS
    ]
    team_features_for_merge = team_features.drop(columns=base_columns_to_drop)
    matchup_rows = matchup_rows.merge(team_features_for_merge, on=TEAM_KEYS, how="left", validate="m:1")

    opponent_view = matchup_rows.copy()
    opponent_view["slot_in_game"] = 1 - opponent_view["slot_in_game"]
    opponent_view = opponent_view.rename(
        columns={
            column: f"opp_{column}"
            for column in opponent_view.columns
            if column not in GAME_KEYS
        }
    )

    modeling_table = matchup_rows.merge(opponent_view, on=GAME_KEYS, how="left", validate="1:1").copy()
    modeling_table["win"] = (modeling_table["score"] > modeling_table["opp_score"]).astype(int)
    modeling_table = add_starter_calculated_columns(modeling_table)
    modeling_table = add_matchup_interaction_columns(modeling_table)
    modeling_table = add_relative_strength_features(modeling_table)
    modeling_table = add_consensus_rating_features(modeling_table)
    modeling_table = add_rank_and_volatility_features(modeling_table)
    if include_auto_diffs:
        modeling_table = add_auto_difference_columns(modeling_table)
    modeling_table = drop_leakage_columns(modeling_table)

    merge_report = {
        "selected_sources": resolve_source_names(selected_sources, include_optional_sources),
        "team_feature_sources": source_report,
        "location_merge": location_report,
        "team_features_shape": [int(team_features.shape[0]), int(team_features.shape[1])],
        "matchup_rows_shape": [int(matchup_rows.shape[0]), int(matchup_rows.shape[1])],
        "modeling_table_shape": [int(modeling_table.shape[0]), int(modeling_table.shape[1])],
        "target_column": "win",
        "blocked_feature_columns": LEAKAGE_COLUMNS,
    }
    return PreparedDataBundle(
        team_features=team_features,
        matchup_rows=matchup_rows,
        modeling_table=modeling_table,
        merge_report=merge_report,
    )


def write_bundle(bundle: PreparedDataBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle.team_features.to_csv(output_dir / "team_features.csv", index=False)
    bundle.matchup_rows.to_csv(output_dir / "matchup_rows.csv", index=False)
    bundle.modeling_table.to_csv(output_dir / "modeling_table.csv", index=False)
    with (output_dir / "merge_report.json").open("w", encoding="utf-8") as file_handle:
        json.dump(bundle.merge_report, file_handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prepared March Madness modeling tables.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_KAGGLE_DATA_DIR,
        help="Directory containing the Kaggle challenge CSV files.",
    )
    parser.add_argument(
        "--barttorvik-data-dir",
        type=Path,
        default=DEFAULT_BARTTORVIK_DATA_DIR,
        help="Directory containing the supplemental BartTorvik CSV files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for prepared outputs.")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional explicit source names. Defaults to the stable fully covered sources.",
    )
    parser.add_argument(
        "--include-optional-sources",
        action="store_true",
        help="Include lower-coverage team-season sources such as EvanMiya and Shooting Splits.",
    )
    parser.add_argument(
        "--include-auto-diffs",
        action="store_true",
        help="Add difference columns for every numeric team/opponent feature pair.",
    )
    args = parser.parse_args()

    bundle = build_modeling_table(
        kaggle_data_dir=args.data_dir,
        barttorvik_data_dir=args.barttorvik_data_dir,
        selected_sources=args.sources,
        include_optional_sources=args.include_optional_sources,
        include_auto_diffs=args.include_auto_diffs,
    )
    write_bundle(bundle, args.output_dir)

    print(f"Wrote prepared data to: {args.output_dir}")
    print("Shapes:")
    print(f"  team_features: {bundle.team_features.shape}")
    print(f"  matchup_rows: {bundle.matchup_rows.shape}")
    print(f"  modeling_table: {bundle.modeling_table.shape}")
    print("Blocked feature columns:")
    print(f"  {bundle.merge_report['blocked_feature_columns']}")


if __name__ == "__main__":
    main()
