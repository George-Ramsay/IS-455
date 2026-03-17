from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode

import requests


DEFAULT_URL = "https://www.teamrankings.com/ncaa-basketball/ranking/schedule-strength-by-other"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data-TeamRankings"
USER_AGENT = "Mozilla/5.0"


class TeamRankingsTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._capture_table = False
        self._table_depth = 0
        self._in_head = False
        self._in_body = False
        self._in_cell = False
        self._current_cell: list[str] = []
        self._current_row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "table" and not self._capture_table:
            classes = attrs_dict.get("class", "") or ""
            if "tr-table" in classes and "datatable" in classes:
                self._capture_table = True
                self._table_depth = 1
                return
        elif tag == "table" and self._capture_table:
            self._table_depth += 1

        if not self._capture_table:
            return
        if tag == "thead":
            self._in_head = True
        elif tag == "tbody":
            self._in_body = True
        elif tag == "tr" and (self._in_head or self._in_body):
            self._current_row = []
        elif tag in {"th", "td"} and (self._in_head or self._in_body):
            self._in_cell = True
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if self._capture_table and tag == "table":
            self._table_depth -= 1
            if self._table_depth == 0:
                self._capture_table = False
            return

        if not self._capture_table:
            return
        if tag == "thead":
            self._in_head = False
        elif tag == "tbody":
            self._in_body = False
        elif tag in {"th", "td"} and self._in_cell:
            value = " ".join("".join(self._current_cell).split())
            self._current_row.append(unescape(value))
            self._in_cell = False
            self._current_cell = []
        elif tag == "tr" and self._current_row:
            self.rows.append(self._current_row)
            self._current_row = []

    def handle_data(self, data: str) -> None:
        if self._capture_table and self._in_cell:
            self._current_cell.append(data)


def fetch_page(url: str, date: str | None) -> tuple[str, str]:
    params = {"date": date} if date else {}
    request_url = f"{url}?{urlencode(params)}" if params else url
    response = requests.get(
        request_url,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    return response.text, request_url


def parse_snapshot_date(html: str) -> str:
    match = re.search(r'data-query-param-name="date" value="([^"]+)"', html)
    if not match:
        raise ValueError("Could not find the page snapshot date.")
    return match.group(1)


def infer_season_year(snapshot_date: str) -> int:
    parsed = datetime.strptime(snapshot_date, "%m/%d/%Y")
    return parsed.year + 1 if parsed.month >= 10 else parsed.year


def extract_table_rows(html: str) -> list[list[str]]:
    parser = TeamRankingsTableParser()
    parser.feed(html)
    if len(parser.rows) < 2:
        raise ValueError("Could not find the TeamRankings data table.")
    return parser.rows


def split_team_and_record(value: str) -> tuple[str, str | None]:
    match = re.match(r"^(.*?)(?:\s+\((\d+-\d+)\))?$", value)
    if not match:
        return value.strip(), None
    return match.group(1).strip(), match.group(2)


def build_records(rows: list[list[str]], snapshot_date: str, source_url: str) -> list[dict[str, object]]:
    header, *data_rows = rows
    normalized_header = [column.strip().lower() for column in header]
    expected_header = ["rank", "team", "rating", "hi", "lo", "last"]
    if normalized_header != expected_header:
        raise ValueError(f"Unexpected table columns: {header}")

    season_year = infer_season_year(snapshot_date)
    records: list[dict[str, object]] = []
    for row in data_rows:
        if len(row) != len(header):
            continue
        team_name, record = split_team_and_record(row[1])
        wins = int(record.split("-")[0]) if record else None
        losses = int(record.split("-")[1]) if record else None
        records.append(
            {
                "season_year": season_year,
                "snapshot_date": snapshot_date,
                "rank": int(row[0]),
                "team": team_name,
                "record": record,
                "wins": wins,
                "losses": losses,
                "rating": float(row[2]),
                "hi": int(row[3]),
                "lo": int(row[4]),
                "last": int(row[5]),
                "source_url": source_url,
            }
        )
    return records


def make_output_path(output_dir: Path, snapshot_date: str) -> Path:
    file_date = datetime.strptime(snapshot_date, "%m/%d/%Y").strftime("%Y-%m-%d")
    return output_dir / f"{file_date}_schedule_strength_by_other.csv"


def write_csv(output_path: Path, records: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "season_year",
        "snapshot_date",
        "rank",
        "team",
        "record",
        "wins",
        "losses",
        "rating",
        "hi",
        "lo",
        "last",
        "source_url",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape TeamRankings NCAA basketball schedule strength rankings."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="TeamRankings rankings URL to scrape.",
    )
    parser.add_argument(
        "--date",
        help="Optional TeamRankings date parameter in MM/DD/YYYY format.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="Optional first season year to backfill using March 15 snapshots.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="Optional last season year to backfill using March 15 snapshots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the CSV snapshot will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.date and (args.start_year is not None or args.end_year is not None):
        raise ValueError("Use either --date or --start-year/--end-year, not both.")

    if args.start_year is not None or args.end_year is not None:
        start_year = args.start_year if args.start_year is not None else args.end_year
        end_year = args.end_year if args.end_year is not None else args.start_year
        if start_year is None or end_year is None:
            raise ValueError("Both --start-year and --end-year must resolve to a year.")
        for season_year in range(start_year, end_year + 1):
            requested_date = f"03/15/{season_year}"
            html, source_url = fetch_page(args.url, requested_date)
            snapshot_date = parse_snapshot_date(html)
            rows = extract_table_rows(html)
            records = build_records(rows, snapshot_date=snapshot_date, source_url=source_url)
            output_path = make_output_path(args.output_dir, snapshot_date)
            write_csv(output_path, records)
            print(f"Saved {len(records)} rows to {output_path}")
        return

    html, source_url = fetch_page(args.url, args.date)
    snapshot_date = parse_snapshot_date(html)
    rows = extract_table_rows(html)
    records = build_records(rows, snapshot_date=snapshot_date, source_url=source_url)
    output_path = make_output_path(args.output_dir, snapshot_date)
    write_csv(output_path, records)
    print(f"Saved {len(records)} rows to {output_path}")


if __name__ == "__main__":
    main()
