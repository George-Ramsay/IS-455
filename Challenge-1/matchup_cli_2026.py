from __future__ import annotations

import argparse
import difflib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import challenge_data_prep as prep


CHALLENGE_DIR = Path(__file__).resolve().parent
DATA_KAGGLE_DIR = CHALLENGE_DIR / "data-Kaggle"
DATA_BARTTORVIK_DIR = CHALLENGE_DIR / "data-BartTorvik"
DATA_TEAMRANKINGS_DIR = CHALLENGE_DIR / "data-TeamRankings"
DEFAULT_YEAR = 2026
EXAMPLE_LIMIT = 10
SUGGESTION_LIMIT = 8


@dataclass(frozen=True)
class ResolvedTeam:
    canonical_name: str
    normalized_name: str
    sources: tuple[str, ...]


class TeamCatalog:
    def __init__(self, year: int, teams_by_source: dict[str, list[str]]) -> None:
        self.year = year
        self.teams_by_source = teams_by_source
        self.name_sources: dict[str, set[str]] = {}
        for source_name, team_names in teams_by_source.items():
            for team_name in team_names:
                cleaned_name = str(team_name).strip()
                if not cleaned_name:
                    continue
                self.name_sources.setdefault(cleaned_name, set()).add(source_name)

        self.canonical_names = sorted(self.name_sources)
        self.normalized_to_canonical: dict[str, list[str]] = {}
        for canonical_name in self.canonical_names:
            normalized_name = prep.normalize_team_key(canonical_name)
            self.normalized_to_canonical.setdefault(normalized_name, []).append(canonical_name)

    def resolve(self, raw_name: str) -> tuple[ResolvedTeam | None, list[str], str]:
        cleaned_name = " ".join(str(raw_name).strip().split())
        normalized_name = prep.normalize_team_key(cleaned_name)
        if not normalized_name:
            return None, [], normalized_name

        canonical_matches = self.normalized_to_canonical.get(normalized_name, [])
        if len(canonical_matches) == 1:
            canonical_name = canonical_matches[0]
            return (
                ResolvedTeam(
                    canonical_name=canonical_name,
                    normalized_name=normalized_name,
                    sources=tuple(sorted(self.name_sources[canonical_name])),
                ),
                [],
                normalized_name,
            )

        suggestions = self.suggest(cleaned_name)
        return None, suggestions, normalized_name

    def suggest(self, raw_name: str) -> list[str]:
        cleaned_name = " ".join(str(raw_name).strip().split())
        normalized_name = prep.normalize_team_key(cleaned_name)

        exactish = difflib.get_close_matches(cleaned_name, self.canonical_names, n=SUGGESTION_LIMIT, cutoff=0.5)
        normalized_matches = difflib.get_close_matches(
            normalized_name,
            list(self.normalized_to_canonical),
            n=SUGGESTION_LIMIT,
            cutoff=0.45,
        )
        contains_matches = [
            canonical_name
            for canonical_name in self.canonical_names
            if normalized_name and normalized_name in prep.normalize_team_key(canonical_name)
        ][:SUGGESTION_LIMIT]

        ordered: list[str] = []
        for candidate in exactish:
            if candidate not in ordered:
                ordered.append(candidate)
        for normalized_candidate in normalized_matches:
            for canonical_name in self.normalized_to_canonical.get(normalized_candidate, []):
                if canonical_name not in ordered:
                    ordered.append(canonical_name)
        for candidate in contains_matches:
            if candidate not in ordered:
                ordered.append(candidate)
        return ordered[:SUGGESTION_LIMIT]

    def example_names(self, limit: int = EXAMPLE_LIMIT) -> list[str]:
        preferred = [
            "Duke",
            "Michigan",
            "Florida",
            "Houston",
            "Alabama",
            "Tennessee",
            "Auburn",
            "Arizona",
            "Saint Mary's",
            "Texas Tech",
            "Wisconsin",
            "UConn",
        ]
        examples = [name for name in preferred if name in self.name_sources]
        if len(examples) >= limit:
            return examples[:limit]

        for canonical_name in self.canonical_names:
            if canonical_name not in examples:
                examples.append(canonical_name)
            if len(examples) >= limit:
                break
        return examples

    def print_examples(self) -> None:
        print()
        print(f"Example {self.year} team names:")
        print("  " + " | ".join(self.example_names()))
        print()


def read_barttorvik_team_names(year: int) -> list[str]:
    path = DATA_BARTTORVIK_DIR / f"{year}_team_results.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, header=0)
    if "team" not in df.columns:
        return []
    return sorted(df["team"].dropna().astype(str).str.strip().unique().tolist())


def read_teamrankings_team_names(year: int) -> list[str]:
    path = DATA_TEAMRANKINGS_DIR / f"{year}-03-15_schedule_strength_by_other.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "team" not in df.columns:
        return []
    return sorted(df["team"].dropna().astype(str).str.strip().unique().tolist())


def build_team_catalog(year: int) -> TeamCatalog:
    teams_by_source = {
        "BartTorvik": read_barttorvik_team_names(year),
        "TeamRankings SOS": read_teamrankings_team_names(year),
    }
    if not any(teams_by_source.values()):
        raise FileNotFoundError(f"No {year} team-name sources were found in data-BartTorvik or data-TeamRankings.")
    return TeamCatalog(year=year, teams_by_source=teams_by_source)


def source_summary(resolved_team: ResolvedTeam) -> str:
    return ", ".join(resolved_team.sources)


def prompt_for_team(label: str, catalog: TeamCatalog, already_picked: str | None = None) -> ResolvedTeam:
    while True:
        raw_value = input(f"{label}: ").strip()
        lowered = raw_value.lower()

        if lowered in {"quit", "exit"}:
            raise KeyboardInterrupt
        if lowered == "examples":
            catalog.print_examples()
            continue
        if lowered == "list":
            print()
            print(f"First {min(40, len(catalog.canonical_names))} available names:")
            for name in catalog.canonical_names[:40]:
                print(f"  {name}")
            print()
            continue

        resolved, suggestions, normalized_name = catalog.resolve(raw_value)
        if resolved is None:
            print()
            print("No exact team match was found after normalization.")
            print(f"  Your normalized input: {normalized_name or '<empty>'}")
            if suggestions:
                print("  Similar names:")
                for suggestion in suggestions:
                    print(f"    - {suggestion}")
            else:
                print("  No close matches were found.")
            print("  Type one of the suggested names, or use 'examples' or 'list'.")
            print()
            continue

        if already_picked and resolved.canonical_name == already_picked:
            print()
            print("You picked the same team twice. Enter a different opponent.")
            print()
            continue

        print()
        print(f"Matched: {resolved.canonical_name}")
        print(f"  Sources: {source_summary(resolved)}")
        print()
        return resolved


def collect_missing_2026_prediction_inputs() -> list[str]:
    missing: list[str] = []

    year_checks = [
        ("Tournament Matchups.csv", DATA_KAGGLE_DIR / "Tournament Matchups.csv", "YEAR"),
        ("KenPom Barttorvik.csv", DATA_KAGGLE_DIR / "KenPom Barttorvik.csv", "YEAR"),
        ("TeamRankings.csv", DATA_KAGGLE_DIR / "TeamRankings.csv", "YEAR"),
        ("Resumes.csv", DATA_KAGGLE_DIR / "Resumes.csv", "YEAR"),
    ]

    for label, path, year_column in year_checks:
        if not path.exists():
            missing.append(f"{label} is missing")
            continue

        df = pd.read_csv(path)
        if year_column not in df.columns:
            missing.append(f"{label} does not have a {year_column} column")
            continue

        years = pd.to_numeric(df[year_column], errors="coerce").dropna().astype(int)
        if DEFAULT_YEAR not in set(years.tolist()):
            max_year = int(years.max()) if not years.empty else None
            missing.append(f"{label} does not include {DEFAULT_YEAR} data (max year found: {max_year})")

    return missing


def print_readiness_summary(team_a: ResolvedTeam, team_b: ResolvedTeam) -> None:
    missing_inputs = collect_missing_2026_prediction_inputs()

    print("Matchup summary")
    print(f"  {team_a.canonical_name} vs {team_b.canonical_name}")
    print()

    if not missing_inputs:
        print(f"{DEFAULT_YEAR} prediction readiness: ready")
        print("The team names resolved cleanly, and the required 2026 source tables appear to be present.")
        return

    print(f"{DEFAULT_YEAR} prediction readiness: not ready")
    print("The team names are valid, but your iter_4 pipeline still cannot build a full 2026 prediction row.")
    print("Missing current-year inputs:")
    for item in missing_inputs:
        print(f"  - {item}")
    print()
    print("What this means:")
    print("  - You can trust these canonical team names for future input.")
    print("  - You still need 2026 pregame team tables in the Kaggle-style sources before the model can score this game.")


def run_cli(year: int) -> None:
    if year != DEFAULT_YEAR:
        raise ValueError(f"This interface is intentionally locked to {DEFAULT_YEAR}.")

    catalog = build_team_catalog(year)

    print("=" * 72)
    print("March Madness Matchup Input Helper")
    print("=" * 72)
    print(f"Season year: {year}")
    print(f"Loaded team names: {len(catalog.canonical_names)}")
    print("Type two team names to validate them against your 2026 sources.")
    print("Helpful commands: examples | list | quit")
    catalog.print_examples()

    team_a = prompt_for_team("Team 1", catalog)
    team_b = prompt_for_team("Team 2", catalog, already_picked=team_a.canonical_name)
    print_readiness_summary(team_a, team_b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive 2026 team-name matcher for the March Madness Challenge pipeline."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"Season year to use. Only {DEFAULT_YEAR} is supported in this helper.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_cli(year=args.year)
    except KeyboardInterrupt:
        print()
        print("Exited without selecting a matchup.")


if __name__ == "__main__":
    main()
