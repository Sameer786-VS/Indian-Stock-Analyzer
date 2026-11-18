"""Reliable historical daily market-data download and cache layer.

This module is intended for the Indian Stock Research & Swing Analyzer.
It returns the same core OHLCV columns expected by the existing main.py.

Price-basis note
----------------
For Yahoo chart responses, OHLC is adjusted using Adj Close / raw Close, matching
main.py's existing analytical-price convention. The result is suitable for
continuous technical-indicator calculations. It is NOT an exchange-verified
as-traded execution-price series and must not replace the V2.1.3/V2.1.4
technical-versus-execution data workflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Callable, Optional
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

try:  # yfinance is a fallback, not required for the cache/test layer itself.
    import yfinance as yf
except ImportError:  # pragma: no cover - exercised only when yfinance is absent
    yf = None


REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_AGE_HOURS = 18
INDIA_TIMEZONE = "Asia/Kolkata"
PRICE_BASIS_NOTE = (
    "Yahoo adjusted analytical OHLC: Open/High/Low scaled by Adj Close / "
    "raw Close and Close set to Adj Close when available; not an as-traded "
    "execution-price series."
)
YAHOO_HOSTS = ("query2.finance.yahoo.com", "query1.finance.yahoo.com")
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}


@dataclass(frozen=True)
class CacheInfo:
    symbol: str
    status: str
    source: str
    downloaded_at_utc: str
    latest_market_date: str
    rows: int
    price_basis: str = PRICE_BASIS_NOTE


def _safe_symbol(symbol: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", str(symbol).strip()).strip("_")
    if not value:
        raise ValueError("A non-empty stock symbol is required.")
    return value


def _india_daily_index(index: pd.Index) -> pd.DatetimeIndex:
    """Convert dates to Indian exchange calendar dates and remove timezone."""
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    except Exception as exc:
        raise ValueError("Market data must have a datetime-like index.") from exc
    if dates.isna().any():
        raise ValueError("Market data contains invalid dates.")
    if dates.tz is not None:
        dates = dates.tz_convert(INDIA_TIMEZONE).tz_localize(None)
    return dates.normalize()


def normalize_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sorted daily rows; reject duplicate dates rather than hiding them."""
    if frame is None or frame.empty:
        raise ValueError("Market data is empty.")
    out = frame.copy()
    out.index = _india_daily_index(out.index)
    out.index.name = "Date"
    if out.index.has_duplicates:
        duplicates = out.index[out.index.duplicated()].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Duplicate daily dates found: {duplicates[:5]}")
    return out.sort_index()


def validate_ohlcv(frame: pd.DataFrame, min_rows: int = 1) -> pd.DataFrame:
    """Validate daily OHLCV without silently fabricating missing market values.

    Missing volume is permitted because index feeds may not publish volume.
    Missing/non-finite prices, non-positive prices, invalid OHLC relationships,
    and negative/non-finite volume values are rejected.
    """
    out = normalize_daily_frame(frame)
    missing = [column for column in REQUIRED_COLUMNS if column not in out.columns]
    if missing:
        raise ValueError(f"Required OHLCV columns are missing: {missing}")

    for column in REQUIRED_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    price_columns = ["Open", "High", "Low", "Close"]
    invalid = out[price_columns].isna().any(axis=1)
    invalid |= ~np.isfinite(out[price_columns].to_numpy(dtype=float)).all(axis=1)
    invalid |= (out[price_columns] <= 0).any(axis=1)
    invalid |= out["High"] < out["Low"]
    invalid |= out["High"] < out[["Open", "Close"]].max(axis=1)
    invalid |= out["Low"] > out[["Open", "Close"]].min(axis=1)

    volume = out["Volume"]
    invalid_volume = volume.notna() & (
        ~np.isfinite(volume.fillna(0).to_numpy(dtype=float)) | (volume < 0)
    )
    invalid |= invalid_volume

    if invalid.any():
        bad_dates = out.index[invalid].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Invalid OHLCV rows found; dates: {bad_dates[:8]}")
    if len(out) < int(min_rows):
        raise ValueError(
            f"Insufficient market history: {len(out)} rows; at least {min_rows} required."
        )
    return out


def merge_price_history(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge a refresh into cached data; new provider rows win on overlapping dates."""
    old = validate_ohlcv(existing)
    new = validate_ohlcv(incoming)
    merged = pd.concat([old, new]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return validate_ohlcv(merged)


def parse_yahoo_chart_payload(payload: dict) -> pd.DataFrame:
    """Parse Yahoo chart JSON and retain main.py's adjusted analytical OHLC basis."""
    chart = (payload or {}).get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo Finance returned: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise ValueError("Yahoo Finance returned no price results.")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = (result.get("indicators") or {}).get("quote") or []
    if not timestamps or not quotes:
        raise ValueError("Yahoo response is missing timestamps or OHLCV data.")

    row_count = len(timestamps)

    def fit_length(values) -> list:
        values = list(values or [])
        return (values + [None] * max(0, row_count - len(values)))[:row_count]

    index = pd.to_datetime(timestamps, unit="s", utc=True)
    exchange_timezone = (result.get("meta") or {}).get("exchangeTimezoneName")
    if exchange_timezone:
        try:
            index = index.tz_convert(exchange_timezone)
        except (TypeError, ValueError):
            index = index.tz_convert(INDIA_TIMEZONE)
    else:
        index = index.tz_convert(INDIA_TIMEZONE)
    index = index.tz_localize(None).normalize()

    quote_row = quotes[0]
    frame = pd.DataFrame(
        {
            "Open": fit_length(quote_row.get("open")),
            "High": fit_length(quote_row.get("high")),
            "Low": fit_length(quote_row.get("low")),
            "Close": fit_length(quote_row.get("close")),
            "Volume": fit_length(quote_row.get("volume")),
        },
        index=index,
    )

    adjusted_rows = (result.get("indicators") or {}).get("adjclose") or []
    raw_close = pd.to_numeric(frame["Close"], errors="coerce")
    if adjusted_rows:
        adjusted_close = pd.to_numeric(
            pd.Series(fit_length(adjusted_rows[0].get("adjclose")), index=frame.index),
            errors="coerce",
        )
        factor = adjusted_close / raw_close.replace(0, np.nan)
        factor = factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        for column in ("Open", "High", "Low"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce") * factor
        frame["Close"] = adjusted_close.where(adjusted_close.notna(), raw_close)
    else:
        frame["Close"] = raw_close

    frame.index.name = "Date"
    return validate_ohlcv(frame)


class MarketDataManager:
    """Download, validate and cache daily history with incremental refreshes.

    Args:
        cache_dir: Folder for per-symbol CSV cache and JSON metadata.
        max_cache_age_hours: A younger cache is returned without a network call.
        fetcher: Optional test/custom callable ``fetcher(symbol, start_date)``.
                 ``start_date=None`` requests a full initial history.
        request_timeout: Requests timeout as (connect, read) seconds.
        retry_attempts: Attempts per Yahoo host for temporary errors.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        max_cache_age_hours: float = DEFAULT_CACHE_AGE_HOURS,
        fetcher: Optional[Callable[[str, Optional[date]], pd.DataFrame]] = None,
        request_timeout: tuple[int, int] = (10, 25),
        retry_attempts: int = 2,
        sleep_seconds: float = 1.0,
    ) -> None:
        if float(max_cache_age_hours) < 0:
            raise ValueError("max_cache_age_hours cannot be negative.")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_age = timedelta(hours=float(max_cache_age_hours))
        self.fetcher = fetcher
        self.request_timeout = request_timeout
        self.retry_attempts = max(1, int(retry_attempts))
        self.sleep_seconds = max(0.0, float(sleep_seconds))
        self.last_cache_info: Optional[CacheInfo] = None

    def _paths(self, symbol: str) -> tuple[Path, Path]:
        safe = _safe_symbol(symbol)
        return self.cache_dir / f"{safe}.csv", self.cache_dir / f"{safe}.json"

    def _read_cache(self, symbol: str) -> Optional[tuple[pd.DataFrame, dict]]:
        csv_path, metadata_path = self._paths(symbol)
        if not csv_path.exists() or not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
                raise ValueError("Cache schema version is unsupported.")
            if metadata.get("symbol") != symbol:
                raise ValueError("Cache symbol does not match requested symbol.")
            if not metadata.get("downloaded_at_utc"):
                raise ValueError("Cache metadata is missing downloaded_at_utc.")
            # Reject malformed timestamps before deciding whether the cache is fresh.
            datetime.fromisoformat(metadata["downloaded_at_utc"])
            digest = sha256(csv_path.read_bytes()).hexdigest()
            if digest != metadata.get("csv_sha256"):
                raise ValueError("Cache checksum does not match its metadata.")
            data = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
            data = validate_ohlcv(data)
            return data, metadata
        except Exception as exc:
            raise ValueError(f"Could not trust cached data for {symbol}: {exc}") from exc

    def _atomic_write(self, path: Path, content_writer: Callable[[Path], None]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{path.stem}_", suffix=path.suffix,
                dir=path.parent, delete=False,
            ) as handle:
                temp_name = handle.name
            content_writer(Path(temp_name))
            os.replace(temp_name, path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    def _write_cache(self, symbol: str, data: pd.DataFrame, source: str) -> dict:
        csv_path, metadata_path = self._paths(symbol)
        data = validate_ohlcv(data)
        self._atomic_write(
            csv_path,
            lambda temporary: data.to_csv(
                temporary, index=True, index_label="Date", date_format="%Y-%m-%d",
                float_format="%.10g",
            ),
        )
        digest = sha256(csv_path.read_bytes()).hexdigest()
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "symbol": symbol,
            "source": source,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "first_market_date": data.index.min().date().isoformat(),
            "latest_market_date": data.index.max().date().isoformat(),
            "row_count": int(len(data)),
            "columns": list(data.columns),
            "csv_file": csv_path.name,
            "csv_sha256": digest,
            "price_basis": PRICE_BASIS_NOTE,
        }
        self._atomic_write(
            metadata_path,
            lambda temporary: temporary.write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            ),
        )
        return metadata

    def _request_yahoo(self, symbol: str, start_date: Optional[date]) -> pd.DataFrame:
        errors: list[str] = []
        encoded = quote(symbol, safe="")
        for host in YAHOO_HOSTS:
            url = f"https://{host}/v8/finance/chart/{encoded}"
            for attempt in range(1, self.retry_attempts + 1):
                try:
                    if start_date is None:
                        params = {"range": "10y", "interval": "1d", "events": "div,splits"}
                    else:
                        start_utc = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
                        end_utc = datetime.now(timezone.utc) + timedelta(days=2)
                        params = {
                            "period1": int(start_utc.timestamp()),
                            "period2": int(end_utc.timestamp()),
                            "interval": "1d",
                            "events": "div,splits",
                        }
                    response = requests.get(
                        url, params=params, headers=DEFAULT_HEADERS,
                        timeout=self.request_timeout,
                    )
                    if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retry_attempts:
                        time.sleep(self.sleep_seconds * attempt)
                        continue
                    response.raise_for_status()
                    return parse_yahoo_chart_payload(response.json())
                except Exception as exc:
                    errors.append(f"{host} attempt {attempt}: {exc}")
                    if attempt < self.retry_attempts:
                        time.sleep(self.sleep_seconds * attempt)
        raise RuntimeError("Yahoo chart download failed: " + " | ".join(errors))

    def _request_yfinance(self, symbol: str, start_date: Optional[date]) -> pd.DataFrame:
        if yf is None:
            raise RuntimeError("yfinance is not installed; fallback is unavailable.")
        ticker = yf.Ticker(symbol)
        if start_date is None:
            data = ticker.history(
                period="10y", interval="1d", auto_adjust=True,
                actions=False, timeout=self.request_timeout[1],
            )
        else:
            end_date = datetime.now(timezone.utc).date() + timedelta(days=2)
            data = ticker.history(
                start=start_date.isoformat(), end=end_date.isoformat(),
                interval="1d", auto_adjust=True, actions=False,
                timeout=self.request_timeout[1],
            )
        if data is None or data.empty:
            raise ValueError("yfinance returned no historical rows.")
        data = data.copy()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"yfinance is missing columns: {missing}")
        return validate_ohlcv(data[list(REQUIRED_COLUMNS)])

    def _fetch(self, symbol: str, start_date: Optional[date]) -> tuple[pd.DataFrame, str]:
        if self.fetcher is not None:
            data = self.fetcher(symbol, start_date)
            return validate_ohlcv(data), "custom_fetcher"
        errors = []
        try:
            return validate_ohlcv(self._request_yahoo(symbol, start_date)), "yahoo_chart_api"
        except Exception as exc:
            errors.append(f"Yahoo chart API: {exc}")
        try:
            return validate_ohlcv(self._request_yfinance(symbol, start_date)), "yfinance_auto_adjust"
        except Exception as exc:
            errors.append(f"yfinance fallback: {exc}")
        raise RuntimeError("All market-data providers failed. " + " | ".join(errors))

    @staticmethod
    def _attach_info(data: pd.DataFrame, info: CacheInfo) -> pd.DataFrame:
        result = data.copy()
        result.attrs["market_data"] = {
            "symbol": info.symbol,
            "cache_status": info.status,
            "source": info.source,
            "downloaded_at_utc": info.downloaded_at_utc,
            "latest_market_date": info.latest_market_date,
            "rows": info.rows,
            "price_basis": info.price_basis,
        }
        return result

    def get_history(
        self,
        symbol: str,
        min_rows: int = 61,
        force_refresh: bool = False,
        full_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return validated daily history, using cache and incremental updates.

        ``force_refresh`` bypasses cache age but still downloads only the recent
        overlap window. ``full_refresh`` deliberately requests the full 10-year
        history and replaces the cache after successful validation.
        """
        symbol = str(symbol).strip().upper()
        if not symbol:
            raise ValueError("Stock symbol cannot be empty.")

        cached = None
        cache_error = None
        try:
            cached = self._read_cache(symbol)
        except Exception as exc:
            cache_error = str(exc)
            print(f"Warning: {cache_error}; attempting a fresh download.")

        now = datetime.now(timezone.utc)
        if cached and not force_refresh and not full_refresh:
            data, metadata = cached
            downloaded_at = datetime.fromisoformat(metadata["downloaded_at_utc"])
            if downloaded_at.tzinfo is None:
                downloaded_at = downloaded_at.replace(tzinfo=timezone.utc)
            if now - downloaded_at <= self.max_cache_age:
                if len(data) < min_rows:
                    raise ValueError(
                        f"Cached history for {symbol} has {len(data)} rows; {min_rows} required."
                    )
                info = CacheInfo(
                    symbol=symbol, status="FRESH_CACHE", source=metadata["source"],
                    downloaded_at_utc=metadata["downloaded_at_utc"],
                    latest_market_date=data.index.max().date().isoformat(), rows=len(data),
                )
                self.last_cache_info = info
                return self._attach_info(data, info)

        fallback_data = cached[0] if cached else None
        fallback_metadata = cached[1] if cached else None
        if cached and not full_refresh:
            old_data, old_metadata = cached
            # Re-fetch a short overlap to catch corrected bars without redownloading 10 years.
            start_date = (old_data.index.max() - pd.Timedelta(days=7)).date()
        else:
            old_data, old_metadata = None, None
            start_date = None

        try:
            incoming, source = self._fetch(symbol, start_date)
            if incoming.empty:
                raise ValueError("Provider returned an empty refresh.")
            if old_data is not None:
                updated = merge_price_history(old_data, incoming)
                status = "INCREMENTAL_UPDATE"
            else:
                updated = validate_ohlcv(incoming)
                status = "FULL_REFRESH" if full_refresh and fallback_data is not None else "FULL_DOWNLOAD"
            updated = validate_ohlcv(updated, min_rows=min_rows)
            metadata = self._write_cache(symbol, updated, source)
            info = CacheInfo(
                symbol=symbol, status=status, source=source,
                downloaded_at_utc=metadata["downloaded_at_utc"],
                latest_market_date=metadata["latest_market_date"], rows=metadata["row_count"],
            )
            self.last_cache_info = info
            return self._attach_info(updated, info)
        except Exception as exc:
            if fallback_data is not None and len(fallback_data) >= min_rows:
                # A failed refresh does not destroy a previously validated cache.
                downloaded_at = fallback_metadata.get("downloaded_at_utc", "unknown")
                info = CacheInfo(
                    symbol=symbol, status="STALE_CACHE_FALLBACK", source=fallback_metadata.get("source", "unknown"),
                    downloaded_at_utc=downloaded_at,
                    latest_market_date=fallback_data.index.max().date().isoformat(), rows=len(fallback_data),
                )
                self.last_cache_info = info
                print(
                    f"Warning: Refresh failed for {symbol}; using validated but possibly stale cache. "
                    f"Reason: {exc}"
                )
                return self._attach_info(fallback_data, info)
            detail = f"; cache issue: {cache_error}" if cache_error else ""
            raise RuntimeError(f"Could not obtain valid history for {symbol}: {exc}{detail}") from exc
