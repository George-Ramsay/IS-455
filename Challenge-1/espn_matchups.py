from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import requests


SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/scoreboard"
)

TEAMS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/teams"
)

DEFAULT_TIMEOUT = 20
DEFAULT_YEAR = 2026


@dataclass
class TeamInfo:
    display_name: str
    short_display_name: str | None = None
    abbreviation: str | None = None
    location: str | None = None
    name: str | None = None
    espn_team_id: str | None = None


@dataclass
class Matchup:
    season: int
    date: str | None
    event_id: str | None
    event_name: str | None
    round_label: str | None
    status: str | None
    neutral_site: bool | None
    venue: str | None
    city: str | None
    state: str | None
    home_team: str | None
    away_team: str | None
    home_team_id: str | None
    away_team_id: str | None
    home_seed: str | None
    away_seed: str | None
    home_rank: str | None
    away_rank: str | None


def normalize_name(text: str) -> str:
    """
    Aggressive normalization for matching user input and source names.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()

    text = text.replace("&", " and ")
    text = text.replace("st.", "saint")
    text = text.replace("st ", "saint ")
    text = text.replace("mt.", "mount")
    text = text.replace("ucf", "central florida")
    text = text.replace("uconn", "connecticut")
    text = text.replace("ole miss", "mississippi")
    text = text.replace("usc", "southern california")

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def fetch_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_scoreboard(date_yyyymmdd: str, group: int | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"dates": date_yyyymmdd}
    if group is not None:
        params["groups"] = str(group)
    return fetch_json(SCOREBOARD_URL, params=params)


def fetch_all_teams() -> list[TeamInfo]:
    payload = fetch_json(TEAMS_URL)

    results: list[TeamInfo] = []
    sports = payload.get("sports", [])
    for sport in sports:
        for league in sport.get("leagues", []):
            for team_block in league.get("teams", []):
                team = team_block.get("team", {})
                results.append(
                    TeamInfo(
                        display_name=team.get("displayName", ""),
                        short_display_name=team.get("shortDisplayName"),
                        abbreviation=team.get("abbreviation"),
                        location=team.get("location"),
                        name=team.get("name"),
                        espn_team_id=team.get("id"),
                    )
                )
    return results


def build_team_alias_index(teams: Iterable[TeamInfo]) -> dict[str, TeamInfo]:
    """
    Map many normalized aliases to a canonical team record.
    """
    alias_map: dict[str, TeamInfo] = {}

    for team in teams:
        candidates = {
            team.display_name,
            team.short_display_name or "",
            team.abbreviation or "",
            team.location or "",
            team.name or "",
        }

        if team.location and team.name:
            candidates.add(f"{team.location} {team.name}")

        for candidate in candidates:
            normalized = normalize_name(candidate)
            if normalized and normalized not in alias_map:
                alias_map[normalized] = team

    return alias_map


def suggest_team_names(user_text: str, teams: list[TeamInfo], limit: int = 8) -> list[str]:
    canonical_names = sorted({team.display_name for team in teams})
    name_lookup = {normalize_name(name): name for name in canonical_names}

    matches = difflib.get_close_matches(
        normalize_name(user_text),
        list(name_lookup.keys()),
        n=limit,
        cutoff=0.55,
    )
    return [name_lookup[match] for match in matches]


def resolve_team_name(user_text: str, teams: list[TeamInfo], alias_index: dict[str, TeamInfo]) -> TeamInfo:
    normalized = normalize_name(user_text)

    if normalized in alias_index:
        return alias_index[normalized]

    substring_hits: list[TeamInfo] = []
    for team in teams:
        for candidate in filter(
            None,
            [
                team.display_name,
                team.short_display_name,
                team.abbreviation,
                team.location,
                team.name,
            ],
        ):
            candidate_norm = normalize_name(candidate)
            if normalized and normalized in candidate_norm:
                substring_hits.append(team)
                break

    unique_hits = {team.display_name: team for team in substring_hits}
    if len(unique_hits) == 1:
        return next(iter(unique_hits.values()))

    suggestions = suggest_team_names(user_text, teams)
    suggestion_text = ", ".join(suggestions[:8]) if suggestions else "No close matches found."
    raise ValueError(
        f"Could not resolve team '{user_text}'. "
        f"Similar names: {suggestion_text}"
    )


def extract_competitor_team(
    competitor: dict[str, Any],
) -> tuple[str | None, str | None, str | None, str | None]:
    team = competitor.get("team", {})
    display_name = team.get("displayName")
    team_id = team.get("id")
    seed = competitor.get("curatedRank", {}).get("current")
    rank = competitor.get("rank")
    if seed is not None:
        seed = str(seed)
    if rank is not None:
        rank = str(rank)
    return display_name, team_id, seed, rank


def parse_matchups(scoreboard_payload: dict[str, Any], season: int = DEFAULT_YEAR) -> list[Matchup]:
    events = scoreboard_payload.get("events", [])
    parsed: list[Matchup] = []

    for event in events:
        competitions = event.get("competitions", [])
        if not competitions:
            continue

        competition = competitions[0]
        competitors = competition.get("competitors", [])

        home = next((competitor for competitor in competitors if competitor.get("homeAway") == "home"), None)
        away = next((competitor for competitor in competitors if competitor.get("homeAway") == "away"), None)

        home_team, home_id, home_seed, home_rank = extract_competitor_team(home or {})
        away_team, away_id, away_seed, away_rank = extract_competitor_team(away or {})

        status = competition.get("status", {}).get("type", {}).get("description")
        venue = (competition.get("venue") or {}).get("fullName")
        address = (competition.get("venue") or {}).get("address") or {}
        round_label = (competition.get("type") or {}).get("text")

        parsed.append(
            Matchup(
                season=season,
                date=event.get("date"),
                event_id=event.get("id"),
                event_name=event.get("name"),
                round_label=round_label,
                status=status,
                neutral_site=competition.get("neutralSite"),
                venue=venue,
                city=address.get("city"),
                state=address.get("state"),
                home_team=home_team,
                away_team=away_team,
                home_team_id=home_id,
                away_team_id=away_id,
                home_seed=home_seed,
                away_seed=away_seed,
                home_rank=home_rank,
                away_rank=away_rank,
            )
        )

    return parsed


def write_matchups_csv(matchups: list[Matchup], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not matchups:
        fieldnames = [field.name for field in Matchup.__dataclass_fields__.values()]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
        return

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(matchups[0]).keys()))
        writer.writeheader()
        for matchup in matchups:
            writer.writerow(asdict(matchup))


def print_matchups(matchups: list[Matchup]) -> None:
    if not matchups:
        print("No matchups found.")
        return

    print()
    print(f"Found {len(matchups)} matchup(s):")
    print("-" * 100)
    for index, matchup in enumerate(matchups, start=1):
        left = matchup.away_team or "Unknown"
        right = matchup.home_team or "Unknown"
        seed_left = f" ({matchup.away_seed})" if matchup.away_seed else ""
        seed_right = f" ({matchup.home_seed})" if matchup.home_seed else ""
        round_text = f" | {matchup.round_label}" if matchup.round_label else ""
        venue_text = f" | {matchup.venue}" if matchup.venue else ""
        city_text = f" | {matchup.city}, {matchup.state}" if matchup.city or matchup.state else ""
        print(f"{index:>2}. {left}{seed_left} vs {right}{seed_right}{round_text}{venue_text}{city_text}")
    print("-" * 100)


def print_team_examples(teams: list[TeamInfo], limit: int = 20) -> None:
    examples = sorted({team.display_name for team in teams})[:limit]
    print("Example ESPN team names:")
    for name in examples:
        print(f"  - {name}")


def filter_matchups_by_teams(matchups: list[Matchup], team_a: str, team_b: str) -> list[Matchup]:
    normalized_a = normalize_name(team_a)
    normalized_b = normalize_name(team_b)

    filtered: list[Matchup] = []
    for matchup in matchups:
        home = normalize_name(matchup.home_team or "")
        away = normalize_name(matchup.away_team or "")
        if {normalized_a, normalized_b} == {home, away}:
            filtered.append(matchup)
    return filtered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pull current ESPN men's college basketball matchups."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Date in YYYYMMDD format, for example 20260319.",
    )
    parser.add_argument(
        "--group",
        type=int,
        default=None,
        help=(
            "Optional ESPN group filter. "
            "Example: 50 is commonly used for D-I scoreboard pages, "
            "but ESPN grouping can vary."
        ),
    )
    parser.add_argument(
        "--team",
        default=None,
        help="Optional first team name filter.",
    )
    parser.add_argument(
        "--opponent",
        default=None,
        help="Optional second team name filter.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to write parsed matchup rows.",
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Print raw JSON response for debugging.",
    )
    parser.add_argument(
        "--show-team-examples",
        action="store_true",
        help="Print example ESPN team names before pulling matchups.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        teams = fetch_all_teams()
        alias_index = build_team_alias_index(teams)

        if args.show_team_examples:
            print_team_examples(teams)

        team_a = None
        team_b = None
        if args.team:
            team_a = resolve_team_name(args.team, teams, alias_index).display_name
        if args.opponent:
            team_b = resolve_team_name(args.opponent, teams, alias_index).display_name

        payload = fetch_scoreboard(args.date, group=args.group)

        if args.dump_json:
            print(json.dumps(payload, indent=2))

        matchups = parse_matchups(payload, season=DEFAULT_YEAR)

        if team_a and team_b:
            matchups = filter_matchups_by_teams(matchups, team_a, team_b)

        print_matchups(matchups)

        if args.output_csv:
            output_path = Path(args.output_csv)
            write_matchups_csv(matchups, output_path)
            print(f"Wrote CSV to: {output_path}")

        if team_a and team_b and not matchups:
            print()
            print(f"No ESPN matchup found on {args.date} for: {team_a} vs {team_b}")

        return 0

    except requests.HTTPError as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Request error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 99


if __name__ == "__main__":
    raise SystemExit(main())
