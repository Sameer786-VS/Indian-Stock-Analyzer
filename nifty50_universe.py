"""Load and validate the dated NIFTY 50 constituent universe.

The CSV is a dated snapshot, not a live feed. Update it from the official
NSE Indices constituent download when the index composition changes.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_UNIVERSE_PATH = PROJECT_DIR / "data" / "reference" / "nifty50_constituents.csv"
EXPECTED_CONSTITUENT_COUNT = 50
REQUIRED_COLUMNS = (
    "Company_Name",
    "NSE_Symbol",
    "Yahoo_Symbol",
    "NSE_Industry",
    "Index_Name",
    "Constituent_As_Of",
    "Source",
    "Source_URL",
    "Reference_URL",
)


def _normalise_key(value: str) -> str:
    value = str(value or "").strip().upper().replace(" AND ", " & ")
    value = re.sub(r"[^A-Z0-9&]+", " ", value)
    value = " ".join(value.split())
    for suffix in (" LIMITED", " LTD"):
        if value.endswith(suffix):
            value = value[: -len(suffix)].strip()
    return value


def load_nifty50_universe(
    path: str | Path | None = None,
    expected_count: int = EXPECTED_CONSTITUENT_COUNT,
) -> list[dict[str, str]]:
    """Read a NIFTY 50 CSV and validate its schema and membership rows."""
    file_path = Path(path) if path is not None else DEFAULT_UNIVERSE_PATH
    if not file_path.is_file():
        raise FileNotFoundError(f"NIFTY 50 constituent file not found: {file_path}")

    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        missing = [column for column in REQUIRED_COLUMNS if column not in headers]
        if missing:
            raise ValueError(f"NIFTY 50 CSV is missing required columns: {', '.join(missing)}")

        rows: list[dict[str, str]] = []
        for line_number, raw in enumerate(reader, start=2):
            row = {key: str(raw.get(key, "") or "").strip() for key in REQUIRED_COLUMNS}
            if not all(row[column] for column in REQUIRED_COLUMNS):
                raise ValueError(f"NIFTY 50 CSV has a blank required value on line {line_number}.")
            rows.append(row)

    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} NIFTY 50 constituents, found {len(rows)}. "
            "Review the dated constituent file before scanning."
        )

    yahoo_symbols = [row["Yahoo_Symbol"].upper() for row in rows]
    nse_symbols = [row["NSE_Symbol"].upper() for row in rows]
    if len(set(yahoo_symbols)) != len(yahoo_symbols):
        raise ValueError("NIFTY 50 CSV contains duplicate Yahoo symbols.")
    if len(set(nse_symbols)) != len(nse_symbols):
        raise ValueError("NIFTY 50 CSV contains duplicate NSE symbols.")
    invalid_suffixes = [symbol for symbol in yahoo_symbols if not symbol.endswith(".NS")]
    if invalid_suffixes:
        raise ValueError("Yahoo symbols must use the .NS suffix: " + ", ".join(invalid_suffixes))

    dates = {row["Constituent_As_Of"] for row in rows}
    if len(dates) != 1:
        raise ValueError("All NIFTY 50 rows must have the same Constituent_As_Of date.")
    indices = {row["Index_Name"] for row in rows}
    if indices != {"NIFTY 50"}:
        raise ValueError("The constituent file must contain only NIFTY 50 rows.")

    return rows


def load_nifty50_symbols(path: str | Path | None = None) -> list[str]:
    return [row["Yahoo_Symbol"] for row in load_nifty50_universe(path)]


def build_nifty50_alias_map(
    rows: Iterable[dict[str, str]] | None = None,
) -> dict[str, str]:
    """Map common full names and NSE/Yahoo symbols to Yahoo tickers."""
    universe = list(rows) if rows is not None else load_nifty50_universe()
    aliases: dict[str, str] = {}
    for row in universe:
        target = row["Yahoo_Symbol"].upper()
        for value in (row["Company_Name"], row["NSE_Symbol"], target):
            key = _normalise_key(value)
            if key:
                aliases[key] = target
        # Common user-facing company-name variants.
        company = row["Company_Name"].upper()
        if company.startswith("TATA MOTORS PASSENGER VEHICLES"):
            aliases["TATA MOTORS"] = target
            aliases["TATA MOTORS PASSENGER VEHICLES"] = target
        if company.startswith("MAHINDRA & MAHINDRA"):
            aliases["M AND M"] = target
    aliases["COAL INDIA"] = "COALINDIA.NS"
    aliases["HUL"] = "HINDUNILVR.NS"
    aliases["L AND T"] = "LT.NS"
    return aliases


def lookup_nifty50_symbol(value: str, aliases: dict[str, str] | None = None) -> str | None:
    """Resolve a NIFTY 50 company name or symbol to its Yahoo ticker."""
    mapping = aliases if aliases is not None else build_nifty50_alias_map()
    return mapping.get(_normalise_key(value))


def get_nifty50_metadata_by_yahoo_symbol(
    rows: Iterable[dict[str, str]] | None = None,
) -> dict[str, dict[str, str]]:
    universe = list(rows) if rows is not None else load_nifty50_universe()
    return {row["Yahoo_Symbol"].upper(): row for row in universe}
