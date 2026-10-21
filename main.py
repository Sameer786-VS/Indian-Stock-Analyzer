
from pathlib import Path
from datetime import datetime
from urllib.parse import quote
import os
import time
import traceback
import webbrowser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import yfinance as yf


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "data" / "processed"
CHART_FOLDER = BASE_DIR / "charts"
REPORT_ARCHIVE_FOLDER = DATA_FOLDER / "_archive"

SUPPORT_RESISTANCE_PERIOD = 60
VOLUME_AVERAGE_PERIOD = 20
HIGH_VOLUME_MULTIPLIER = 1.5
MAX_SCAN_STOCKS = 50

# Baseline backtest settings. Costs are an illustrative assumption, not a broker quote.
BACKTEST_HOLDING_SESSIONS = 5
BACKTEST_MIN_SETUP_SCORE = 4
BACKTEST_ROUND_TRIP_COST_PCT = 0.20
BACKTEST_REQUIRE_POSITIVE_RELATIVE_STRENGTH = True

# Keep backtest signals anchored to the same date across reruns.
# Extra historical data is downloaded to warm up the 200-session SMA.
BACKTEST_TEST_START_DATE = "2022-08-03"

# Try both Yahoo chart hosts.
YAHOO_HOSTS = [
    "query2.finance.yahoo.com",
    "query1.finance.yahoo.com"
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*"
}

DATA_FOLDER.mkdir(parents=True, exist_ok=True)
CHART_FOLDER.mkdir(parents=True, exist_ok=True)
REPORT_ARCHIVE_FOLDER.mkdir(parents=True, exist_ok=True)


def archive_existing_reports():
    """Move older timestamped scan/backtest reports out of the main folder.

    Keeps the latest convenience reports and per-stock historical data visible.
    Existing archive files are preserved; nothing is permanently deleted.
    """
    patterns = (
        "scanner_results_20*.csv",
        "scanner_errors_20*.csv",
        "backtest_summary_*_20*.csv",
        "backtest_trades_*_20*.csv",
        "backtest_equity_*_20*.csv",
    )

    moved = 0
    for pattern in patterns:
        for source in DATA_FOLDER.glob(pattern):
            if not source.is_file():
                continue

            destination = REPORT_ARCHIVE_FOLDER / source.name
            if destination.exists():
                # Avoid overwriting an older report with the same filename.
                counter = 1
                while True:
                    alternative = REPORT_ARCHIVE_FOLDER / (
                        f"{source.stem}_{counter}{source.suffix}"
                    )
                    if not alternative.exists():
                        destination = alternative
                        break
                    counter += 1

            try:
                source.replace(destination)
                moved += 1
            except OSError as error:
                print(
                    f"Warning: Could not archive {source.name}: {error}"
                )

    if moved:
        print(
            f"Archived {moved} old report file(s) to: "
            f"{REPORT_ARCHIVE_FOLDER.resolve()}"
        )


# =========================================================
# 2. STOCK NAME AND SYMBOL HANDLING
# =========================================================

ALIASES = {
    "HDFC": "HDFCBANK.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "HDFC BANK LIMITED": "HDFCBANK.NS",
    "HDCF BANK": "HDFCBANK.NS",
    "HDCFBANK": "HDFCBANK.NS",
    "HDCFBANK.NS": "HDFCBANK.NS",
    "ICICI": "ICICIBANK.NS",
    "ICI": "ICICIBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "ICICI BANK LIMITED": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "SBI BANK": "SBIN.NS",
    "STATE BANK OF INDIA": "SBIN.NS",
    "TCS": "TCS.NS",
    "TATA CONSULTANCY SERVICES": "TCS.NS",
    "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "INFOSYS": "INFY.NS",
    "LARSEN AND TOUBRO": "LT.NS",
    "L&T": "LT.NS",
    "HINDUSTAN UNILEVER": "HINDUNILVR.NS",
    "KOTAK BANK": "KOTAKBANK.NS",
    "KOTAK MAHINDRA BANK": "KOTAKBANK.NS",
    "AXIS BANK": "AXISBANK.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "WIPRO": "WIPRO.NS",
    "MARUTI SUZUKI": "MARUTI.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "POWER GRID": "POWERGRID.NS"
}


def normalize_symbol(raw_symbol):
    """Convert common company names or tickers into Yahoo symbols."""

    name = " ".join(raw_symbol.strip().upper().split())

    if not name:
        raise ValueError("Stock name or symbol cannot be empty.")

    if name in ALIASES:
        return ALIASES[name]

    symbol = name.replace(" ", "")

    # Preserve explicit exchange suffixes such as .NS or .BO.
    if "." not in symbol and not symbol.startswith("^"):
        symbol += ".NS"

    return symbol


def get_stock_symbol():
    name = input(
        "\nEnter stock name or ticker "
        "(example: HDFC Bank, ICICI Bank, TCS): "
    )

    return normalize_symbol(name)


def get_scan_symbols():
    raw_text = input(
        "\nEnter stock names/tickers separated by commas.\n"
        "Example: HDFC Bank, ICICI Bank, SBI, TCS\n"
        "Your watchlist: "
    ).strip()

    if not raw_text:
        raise ValueError("Please enter at least one stock.")

    symbols = []
    seen = set()

    for item in raw_text.split(","):
        if not item.strip():
            continue

        symbol = normalize_symbol(item)

        if symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)

    if not symbols:
        raise ValueError("No valid stock symbols were entered.")

    if len(symbols) > MAX_SCAN_STOCKS:
        raise ValueError(
            f"You can scan a maximum of {MAX_SCAN_STOCKS} stocks "
            "in one run."
        )

    return symbols


# =========================================================
# 3. DOWNLOAD DATA USING PYTHON REQUESTS
# =========================================================

def fetch_yahoo_chart_with_requests(symbol):
    """
    Try Yahoo's public chart endpoint with requests.
    Retry and try both query2 and query1 hosts.
    """

    encoded_symbol = quote(symbol, safe="")
    errors = []

    for host in YAHOO_HOSTS:
        url = (
            f"https://{host}/v8/finance/chart/"
            f"{encoded_symbol}"
        )

        for attempt in range(1, 3):
            try:
                print(
                    f"Trying {host} "
                    f"(attempt {attempt}/2)..."
                )

                response = requests.get(
                    url,
                    params={
                        "range": "10y",
                        "interval": "1d",
                        "events": "div,splits"
                    },
                    headers=HEADERS,
                    timeout=(10, 25)
                )

                # Retry temporary server/rate-limit errors.
                if response.status_code in {
                    429, 500, 502, 503, 504
                } and attempt < 2:
                    time.sleep(attempt * 2)
                    continue

                response.raise_for_status()

                payload = response.json()

                chart = payload.get("chart", {})
                chart_error = chart.get("error")

                if chart_error:
                    raise ValueError(
                        f"Yahoo Finance returned: {chart_error}"
                    )

                results = chart.get("result")

                if not results:
                    raise ValueError(
                        f"No price results returned for {symbol}."
                    )

                result = results[0]
                timestamps = result.get("timestamp", [])

                if not timestamps:
                    raise ValueError(
                        f"No historical dates returned for {symbol}."
                    )

                indicators = result.get("indicators", {})
                quote_data = indicators.get("quote", [])

                if not quote_data:
                    raise ValueError(
                        "The response did not contain price data."
                    )

                quote_row = quote_data[0]
                number_of_rows = len(timestamps)

                def fit_length(values):
                    """Match response arrays to the date count."""
                    values = list(values or [])

                    if len(values) < number_of_rows:
                        values += [None] * (
                            number_of_rows - len(values)
                        )

                    return values[:number_of_rows]

                # Build one row for each returned trading date.
                frame = pd.DataFrame(
                    {
                        "Open": fit_length(
                            quote_row.get("open")
                        ),
                        "High": fit_length(
                            quote_row.get("high")
                        ),
                        "Low": fit_length(
                            quote_row.get("low")
                        ),
                        "Close": fit_length(
                            quote_row.get("close")
                        ),
                        "Volume": fit_length(
                            quote_row.get("volume")
                        )
                    },
                    index=pd.to_datetime(
                        timestamps,
                        unit="s",
                        utc=True
                    )
                )

                # Apply Yahoo's adjusted-close factor to OHLC.
                # This makes historical prices more comparable
                # around dividends and stock splits.
                adjusted_rows = indicators.get("adjclose", [])
                adjusted_close = None

                if adjusted_rows:
                    adjusted_close = pd.to_numeric(
                        pd.Series(
                            fit_length(
                                adjusted_rows[0].get("adjclose")
                            ),
                            index=frame.index
                        ),
                        errors="coerce"
                    )

                raw_close = pd.to_numeric(
                    frame["Close"],
                    errors="coerce"
                )

                if adjusted_close is not None:
                    factor = (
                        adjusted_close
                        / raw_close.replace(0, np.nan)
                    )

                    factor = factor.replace(
                        [np.inf, -np.inf],
                        np.nan
                    ).fillna(1.0)

                    for column in ["Open", "High", "Low"]:
                        frame[column] = (
                            pd.to_numeric(
                                frame[column],
                                errors="coerce"
                            ) * factor
                        )

                    frame["Close"] = adjusted_close.where(
                        adjusted_close.notna(),
                        raw_close
                    )
                else:
                    frame["Close"] = raw_close

                # Convert dates to the exchange's local timezone.
                exchange_timezone = (
                    result.get("meta", {})
                    .get("exchangeTimezoneName")
                )

                if exchange_timezone:
                    try:
                        frame.index = frame.index.tz_convert(
                            exchange_timezone
                        )
                    except (TypeError, ValueError):
                        pass

                frame.index.name = "Date"

                for column in [
                    "Open", "High", "Low", "Close", "Volume"
                ]:
                    frame[column] = pd.to_numeric(
                        frame[column],
                        errors="coerce"
                    )

                frame = frame.dropna(
                    subset=["Open", "High", "Low", "Close"]
                )

                frame = frame[
                    ~frame.index.duplicated(keep="last")
                ].sort_index()

                if frame.empty:
                    raise ValueError(
                        "No valid price rows remained after cleaning."
                    )

                print(
                    f"Downloaded {len(frame)} sessions "
                    f"using {host}."
                )

                return frame

            except Exception as error:
                message = f"{host}, attempt {attempt}: {error}"
                errors.append(message)
                print(f"Download attempt failed: {error}")

                if attempt < 2:
                    time.sleep(attempt * 2)

    raise RuntimeError(
        "Requests could not download data. "
        + " | ".join(errors)
    )


# =========================================================
# 4. FALLBACK: YFINANCE
# =========================================================

def download_stock_data(symbol):
    """
    First try requests against both Yahoo hosts.
    If those requests fail, try yfinance as a fallback.
    """

    print(f"\nDownloading historical data for {symbol}...")

    request_error = None

    try:
        data = fetch_yahoo_chart_with_requests(symbol)

    except Exception as error:
        request_error = str(error)

        print("\nDirect download failed.")
        print("Trying yfinance fallback...")

        try:
            data = yf.Ticker(symbol).history(
                period="10y",
                interval="1d",
                auto_adjust=True,
                timeout=25
            )

        except Exception as fallback_error:
            raise RuntimeError(
                f"Could not download {symbol}.\n"
                f"Requests error: {request_error}\n"
                f"yfinance error: {fallback_error}\n"
                "Check your connection, proxy, firewall, "
                "and Yahoo Finance availability."
            ) from fallback_error

    if data is None or data.empty:
        raise ValueError(
            f"No historical data found for {symbol}. "
            "Check the symbol and data availability."
        )

    required = ["Open", "High", "Low", "Close", "Volume"]

    for column in required:
        if column not in data.columns:
            raise ValueError(
                f"Required column '{column}' is missing."
            )

    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    invalid_rows = (
        (data["High"] < data["Low"])
        | (data["High"] < data["Open"])
        | (data["High"] < data["Close"])
        | (data["Low"] > data["Open"])
        | (data["Low"] > data["Close"])
    )

    if invalid_rows.any():
        raise ValueError(
            "Downloaded data contains inconsistent OHLC prices."
        )

    if len(data) < SUPPORT_RESISTANCE_PERIOD + 1:
        raise ValueError(
            "At least 61 trading sessions are required "
            "for support/resistance analysis."
        )

    return data


# =========================================================
# 4A. NIFTY 50 RELATIVE-STRENGTH ANALYSIS
# =========================================================

def download_nifty50_data():
    """Download NIFTY 50 data for comparison with individual stocks."""
    try:
        return download_stock_data("^NSEI")
    except Exception as error:
        print(f"Warning: Could not download NIFTY 50 benchmark: {error}")
        return None


def add_relative_strength(stock_data, benchmark_data):
    """Compare a stock's 20-session return with the NIFTY 50 return."""
    stock = stock_data.copy()

    if benchmark_data is None or benchmark_data.empty:
        stock["NIFTY50_Return_20D_Pct"] = np.nan
        stock["Relative_Strength_20D_Pct"] = np.nan
        return stock

    benchmark = benchmark_data.copy()

    # Normalize dates so timezone differences do not prevent matching.
    for frame in (stock, benchmark):
        index = pd.DatetimeIndex(frame.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        frame.index = index.normalize()
        frame.sort_index(inplace=True)

    stock = stock[~stock.index.duplicated(keep="last")]
    benchmark = benchmark[~benchmark.index.duplicated(keep="last")]

    nifty_returns = benchmark["Close"].pct_change(20) * 100
    stock["NIFTY50_Return_20D_Pct"] = nifty_returns.reindex(stock.index)
    stock["Relative_Strength_20D_Pct"] = (
        stock["Return_20D_Pct"] - stock["NIFTY50_Return_20D_Pct"]
    )
    return stock


# =========================================================
# 5. CALCULATE TECHNICAL INDICATORS
# =========================================================

def calculate_indicators(data):
    data = data.copy()

    # Daily and 20-session returns.
    data["Daily_Return_Pct"] = (
        data["Close"].pct_change() * 100
    )

    data["Return_20D_Pct"] = (
        data["Close"].pct_change(20) * 100
    )

    # Moving averages.
    for period in [20, 50, 200]:
        data[f"SMA_{period}"] = (
            data["Close"].rolling(period).mean()
        )

    # RSI.
    change = data["Close"].diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["RSI_14"] = 100 - (100 / (1 + rs))

    data.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "RSI_14"
    ] = 100

    data.loc[
        (avg_loss == 0) & (avg_gain == 0),
        "RSI_14"
    ] = 50

    # ATR.
    previous_close = data["Close"].shift(1)

    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    data["ATR_14"] = true_range.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    data["ATR_Pct"] = (
        data["ATR_14"] / data["Close"] * 100
    )

    data["Volatility_20D"] = (
        data["Daily_Return_Pct"].rolling(20).std()
        * np.sqrt(252)
    )

    # Support and resistance based on previous 60 sessions.
    data["Support_60D"] = (
        data["Low"]
        .rolling(SUPPORT_RESISTANCE_PERIOD, min_periods=60)
        .min()
        .shift(1)
    )

    data["Resistance_60D"] = (
        data["High"]
        .rolling(SUPPORT_RESISTANCE_PERIOD, min_periods=60)
        .max()
        .shift(1)
    )

    # Average volume excludes today's session.
    data["Avg_Volume_20D"] = (
        data["Volume"]
        .rolling(VOLUME_AVERAGE_PERIOD, min_periods=20)
        .mean()
        .shift(1)
    )

    data["Relative_Volume_20D"] = (
        data["Volume"]
        / data["Avg_Volume_20D"].replace(0, np.nan)
    )

    data["High_Volume_Session"] = (
        data["Relative_Volume_20D"]
        >= HIGH_VOLUME_MULTIPLIER
    ).fillna(False)

    # Fresh breakout/breakdown crossings.
    previous_close = data["Close"].shift(1)

    previous_resistance = data["Resistance_60D"].shift(1)
    previous_support = data["Support_60D"].shift(1)

    data["Breakout_60D"] = (
        (data["Close"] > data["Resistance_60D"])
        & (previous_close <= previous_resistance)
    ).fillna(False)

    data["Breakdown_60D"] = (
        (data["Close"] < data["Support_60D"])
        & (previous_close >= previous_support)
    ).fillna(False)

    high_volume = (
        data["Relative_Volume_20D"]
        >= HIGH_VOLUME_MULTIPLIER
    ).fillna(False)

    data["Breakout_Confirmed"] = (
        data["Breakout_60D"] & high_volume
    )

    data["Breakdown_Confirmed"] = (
        data["Breakdown_60D"] & high_volume
    )

    return data


# =========================================================
# 6. RULE-BASED SCREENING SCORE
# =========================================================

def calculate_score_details(latest):
    """Return the score and the point contribution of each rule."""

    close = latest["Close"]
    sma20 = latest["SMA_20"]
    sma50 = latest["SMA_50"]
    sma200 = latest["SMA_200"]
    return20 = latest["Return_20D_Pct"]
    rsi = latest["RSI_14"]

    details = {
        "Score_Close_Above_SMA20": int(
            pd.notna(sma20) and close > sma20
        ),
        "Score_SMA20_Above_SMA50": int(
            pd.notna(sma20) and pd.notna(sma50) and sma20 > sma50
        ),
        "Score_Close_Above_SMA200": int(
            pd.notna(sma200) and close > sma200
        ),
        "Score_Positive_Return_20D": int(
            pd.notna(return20) and return20 > 0
        ),
        "Score_RSI_50_70": int(
            pd.notna(rsi) and 50 <= rsi <= 70
        ),
        "Score_Breakout_Bonus": (
            2 if bool(latest["Breakout_Confirmed"]) else 0
        ),
        "Score_Breakdown_Penalty": (
            -2 if bool(latest["Breakdown_Confirmed"]) else 0
        ),
    }
    details["Setup_Score"] = sum(details.values())
    return details


def calculate_setup_score(latest):
    """Keep the original scoring rules while exposing their breakdown."""
    return calculate_score_details(latest)["Setup_Score"]


def determine_assessment(latest, score):
    if bool(latest["Breakdown_Confirmed"]):
        return "Breakdown risk flag"

    if bool(latest["Breakout_Confirmed"]):
        return "Breakout flag - review"

    if score >= 4:
        return "Higher score - investigate"

    if score >= 2:
        return "Moderate score - monitor"

    return "Lower score - review manually"


def build_scan_record(symbol, data):
    latest = data.iloc[-1]

    close = latest["Close"]
    sma20 = latest["SMA_20"]
    sma50 = latest["SMA_50"]

    if pd.notna(sma20) and pd.notna(sma50):
        if close > sma20 > sma50:
            trend = "Positive alignment"
        elif close < sma20 < sma50:
            trend = "Negative alignment"
        else:
            trend = "Mixed alignment"
    else:
        trend = "Insufficient data"

    score_details = calculate_score_details(latest)
    score = score_details["Setup_Score"]

    return {
        "Symbol": symbol,
        "Latest_Date": data.index[-1].date().isoformat(),
        "Close": close,
        "Daily_Return_Pct": latest["Daily_Return_Pct"],
        "Return_20D_Pct": latest["Return_20D_Pct"],
        "NIFTY50_Return_20D_Pct": latest.get(
            "NIFTY50_Return_20D_Pct", np.nan
        ),
        "Relative_Strength_20D_Pct": latest.get(
            "Relative_Strength_20D_Pct", np.nan
        ),
        "SMA_20": latest["SMA_20"],
        "SMA_50": latest["SMA_50"],
        "SMA_200": latest["SMA_200"],
        "RSI_14": latest["RSI_14"],
        "ATR_14": latest["ATR_14"],
        "ATR_Pct": latest["ATR_Pct"],
        "Volatility_20D": latest["Volatility_20D"],
        "Support_60D": latest["Support_60D"],
        "Resistance_60D": latest["Resistance_60D"],
        "Volume": latest["Volume"],
        "Avg_Volume_20D": latest["Avg_Volume_20D"],
        "Relative_Volume_20D": latest["Relative_Volume_20D"],
        "High_Volume_Session": bool(
            latest["High_Volume_Session"]
        ),
        "Breakout_60D": bool(latest["Breakout_60D"]),
        "Breakdown_60D": bool(latest["Breakdown_60D"]),
        "Breakout_Confirmed": bool(
            latest["Breakout_Confirmed"]
        ),
        "Breakdown_Confirmed": bool(
            latest["Breakdown_Confirmed"]
        ),
        "Trend": trend,
        "Setup_Score": score,
        "Score_Close_Above_SMA20": score_details["Score_Close_Above_SMA20"],
        "Score_SMA20_Above_SMA50": score_details["Score_SMA20_Above_SMA50"],
        "Score_Close_Above_SMA200": score_details["Score_Close_Above_SMA200"],
        "Score_Positive_Return_20D": score_details["Score_Positive_Return_20D"],
        "Score_RSI_50_70": score_details["Score_RSI_50_70"],
        "Score_Breakout_Bonus": score_details["Score_Breakout_Bonus"],
        "Score_Breakdown_Penalty": score_details["Score_Breakdown_Penalty"],
        "Assessment": determine_assessment(latest, score)
    }


# =========================================================
# 7. DISPLAY INDIVIDUAL STOCK SUMMARY
# =========================================================

def show_summary(symbol, data):
    latest = data.iloc[-1]
    previous = data.iloc[-2]

    def show_number(column):
        value = latest.get(column, np.nan)
        return f"{value:,.2f}" if pd.notna(value) else "N/A"

    def show_volume(value):
        return f"{value:,.0f}" if pd.notna(value) else "N/A"

    close = latest["Close"]
    old_close = previous["Close"]
    change = close - old_close

    change_pct = (
        change / old_close * 100 if old_close != 0 else np.nan
    )

    print("\n" + "=" * 65)
    print(f"STOCK ANALYSIS: {symbol}")
    print("=" * 65)

    print(f"Latest data date:       {data.index[-1].date()}")
    print(f"Closing price:          Rs. {close:,.2f}")
    print(f"Previous close:         Rs. {old_close:,.2f}")
    print(
        f"Daily change:           Rs. {change:,.2f} "
        f"({change_pct:.2f}%)"
    )

    print("\n--- Technical Indicators ---")
    print(f"SMA 20:                 {show_number('SMA_20')}")
    print(f"SMA 50:                 {show_number('SMA_50')}")
    print(f"SMA 200:                {show_number('SMA_200')}")
    print(f"RSI 14:                 {show_number('RSI_14')}")
    print(f"ATR 14:                 {show_number('ATR_14')}")
    print(f"ATR percentage:         {show_number('ATR_Pct')}%")
    print(f"20-session return:      {show_number('Return_20D_Pct')}%")

    print("\n--- Relative Strength vs NIFTY 50 ---")
    print(
        f"NIFTY 50 return (20 sessions): "
        f"{show_number('NIFTY50_Return_20D_Pct')}%"
    )
    print(
        f"Relative performance: "
        f"{show_number('Relative_Strength_20D_Pct')} "
        "percentage points"
    )
    relative_strength = latest.get("Relative_Strength_20D_Pct", np.nan)
    if pd.isna(relative_strength):
        relative_status = "Unavailable"
    elif relative_strength > 0.5:
        relative_status = "Outperforming NIFTY 50"
    elif relative_strength < -0.5:
        relative_status = "Underperforming NIFTY 50"
    else:
        relative_status = "Performing broadly in line"
    print(f"Relative strength status: {relative_status}")

    print("\n--- Support and Resistance ---")
    print(f"Support:                Rs. {show_number('Support_60D')}")
    print(f"Resistance:             Rs. {show_number('Resistance_60D')}")

    print("\n--- Volume ---")
    print(f"Latest volume:          {show_volume(latest['Volume'])}")
    print(
        f"Prior average volume:   "
        f"{show_volume(latest['Avg_Volume_20D'])}"
    )
    print(f"Relative volume:        {show_number('Relative_Volume_20D')}x")

    print("\n--- Breakout / Breakdown ---")

    if bool(latest["Breakout_Confirmed"]):
        print("Volume-confirmed breakout.")
    elif bool(latest["Breakout_60D"]):
        print("Potential breakout; volume confirmation is absent.")
    elif bool(latest["Breakdown_Confirmed"]):
        print("Volume-confirmed breakdown.")
    elif bool(latest["Breakdown_60D"]):
        print("Potential breakdown; volume confirmation is absent.")
    else:
        print("No fresh 60-session breakout or breakdown.")

    record = build_scan_record(symbol, data)

    print("\n--- Research Screening ---")
    print(f"Trend:                  {record['Trend']}")
    print(f"Rule-based score:       {record['Setup_Score']}")
    print(f"Assessment:             {record['Assessment']}")
    print("Score breakdown (each trend/return/RSI rule is worth 1 point):")
    score_rows = [
        ("Close above SMA 20", "Score_Close_Above_SMA20"),
        ("SMA 20 above SMA 50", "Score_SMA20_Above_SMA50"),
        ("Close above SMA 200", "Score_Close_Above_SMA200"),
        ("Positive 20-session return", "Score_Positive_Return_20D"),
        ("RSI between 50 and 70", "Score_RSI_50_70"),
        ("Confirmed breakout bonus", "Score_Breakout_Bonus"),
        ("Confirmed breakdown penalty", "Score_Breakdown_Penalty"),
    ]
    for label, key in score_rows:
        print(f"  {label:<32} {record[key]:+d}")

    recent_columns = [
        "Close",
        "Volume",
        "Relative_Volume_20D",
        "SMA_20",
        "SMA_50",
        "RSI_14",
        "Support_60D",
        "Resistance_60D",
        "Breakout_60D",
        "Breakdown_60D"
    ]

    print("\n--- Last 5 Sessions ---")
    print(data[recent_columns].tail(5).round(2).to_string())

    print(
        "\nResearch indicators are not guaranteed trading signals "
        "or predictions."
    )
    print("=" * 65)


# =========================================================
# 8. SAVE CSV AND CHART
# =========================================================

def save_outputs(
    symbol,
    data,
    create_chart=True,
    open_chart=False
):
    safe_symbol = (
        symbol.replace(".", "_")
        .replace("^", "")
        .replace("/", "_")
        .replace("\\", "_")
    )

    csv_path = DATA_FOLDER / f"{safe_symbol}_history.csv"

    data.to_csv(csv_path)

    print(f"\nCSV saved:\n{csv_path.resolve()}")

    if not create_chart:
        return csv_path

    chart_path = CHART_FOLDER / f"{safe_symbol}_chart.png"
    chart_data = data.tail(252)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    price_ax = axes[0]
    volume_ax = axes[1]

    price_ax.plot(
        chart_data.index,
        chart_data["Close"],
        label="Close",
        linewidth=1.5
    )

    for period in [20, 50, 200]:
        price_ax.plot(
            chart_data.index,
            chart_data[f"SMA_{period}"],
            label=f"SMA {period}",
            linewidth=1
        )

    price_ax.plot(
        chart_data.index,
        chart_data["Support_60D"],
        label="Support",
        linestyle="--"
    )

    price_ax.plot(
        chart_data.index,
        chart_data["Resistance_60D"],
        label="Resistance",
        linestyle="--"
    )

    breakouts = chart_data[chart_data["Breakout_60D"]]
    breakdowns = chart_data[chart_data["Breakdown_60D"]]

    confirmed_up = chart_data[chart_data["Breakout_Confirmed"]]
    confirmed_down = chart_data[chart_data["Breakdown_Confirmed"]]

    price_ax.scatter(
        breakouts.index, breakouts["Close"],
        marker="^", s=55, label="Breakout"
    )

    price_ax.scatter(
        breakdowns.index, breakdowns["Close"],
        marker="v", s=55, label="Breakdown"
    )

    price_ax.scatter(
        confirmed_up.index, confirmed_up["Close"],
        marker="*", s=120, label="Volume-confirmed breakout"
    )

    price_ax.scatter(
        confirmed_down.index, confirmed_down["Close"],
        marker="X", s=65, label="Volume-confirmed breakdown"
    )

    price_ax.set_title(f"{symbol} - Price and Technical Analysis")
    price_ax.set_ylabel("Adjusted price (Rs.)")
    price_ax.legend(fontsize=8, ncol=2)
    price_ax.grid(True, alpha=0.3)

    volume_ax.bar(
        chart_data.index,
        chart_data["Volume"],
        label="Daily volume",
        width=0.8
    )

    volume_ax.plot(
        chart_data.index,
        chart_data["Avg_Volume_20D"],
        label="Previous 20-session average",
        linewidth=1.3
    )

    high_volume = chart_data[chart_data["High_Volume_Session"]]

    volume_ax.scatter(
        high_volume.index,
        high_volume["Volume"],
        marker="o",
        s=20,
        label="High-volume session"
    )

    volume_ax.set_title("Trading Volume")
    volume_ax.set_ylabel("Shares")
    volume_ax.set_xlabel("Date")
    volume_ax.legend(fontsize=8)
    volume_ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    print(f"Chart saved:\n{chart_path.resolve()}")

    if open_chart:
        try:
            if os.name == "nt":
                os.startfile(str(chart_path.resolve()))
            else:
                webbrowser.open(chart_path.resolve().as_uri())

            print("Opening chart...")

        except Exception as error:
            print(f"Automatic opening failed: {error}")
            print("Open the PNG manually from the charts folder.")

    return csv_path


# =========================================================
# 9. ANALYZE ONE STOCK
# =========================================================

def analyze_stock():
    try:
        symbol = get_stock_symbol()
        data = download_stock_data(symbol)
        data = calculate_indicators(data)
        benchmark_data = download_nifty50_data()
        data = add_relative_strength(data, benchmark_data)

        show_summary(symbol, data)

        save_outputs(
            symbol,
            data,
            create_chart=True,
            open_chart=True
        )

        print("\nSingle-stock analysis completed.")

    except Exception as error:
        print(f"\nCould not complete analysis: {error}")
        traceback.print_exc()


# =========================================================
# 10. SCAN MULTIPLE STOCKS
# =========================================================

def scan_multiple_stocks():
    try:
        symbols = get_scan_symbols()
    except ValueError as error:
        print(f"\nInput error: {error}")
        return

    records = []
    errors = []

    print(f"\nStarting scan of {len(symbols)} stocks.")
    print("Downloading NIFTY 50 benchmark once for the scan...")
    benchmark_data = download_nifty50_data()

    for position, symbol in enumerate(symbols, start=1):
        print(f"\n[{position}/{len(symbols)}] {symbol}")

        try:
            data = download_stock_data(symbol)
            data = calculate_indicators(data)
            data = add_relative_strength(data, benchmark_data)

            record = build_scan_record(symbol, data)
            records.append(record)

            # Save chart and CSV, without opening every chart.
            save_outputs(
                symbol,
                data,
                create_chart=True,
                open_chart=False
            )

            print(
                f"Close: Rs. {record['Close']:.2f} | "
                f"Score: {record['Setup_Score']} | "
                f"{record['Assessment']}"
            )

        except Exception as error:
            errors.append({
                "Symbol": symbol,
                "Error": str(error)
            })
            print(f"Failed to analyze {symbol}: {error}")

        # Small pause between stock requests.
        time.sleep(0.5)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if records:
        results = pd.DataFrame(records)

        results = results.sort_values(
            ["Setup_Score", "Return_20D_Pct"],
            ascending=[False, False],
            na_position="last"
        ).reset_index(drop=True)

        report_path = (
            REPORT_ARCHIVE_FOLDER / f"scanner_results_{timestamp}.csv"
        )
        latest_report = (
            DATA_FOLDER / "scanner_results_latest.csv"
        )

        results.to_csv(report_path, index=False)
        results.to_csv(latest_report, index=False)

        display_columns = [
            "Symbol",
            "Close",
            "Daily_Return_Pct",
            "Return_20D_Pct",
            "NIFTY50_Return_20D_Pct",
            "Relative_Strength_20D_Pct",
            "RSI_14",
            "Relative_Volume_20D",
            "Breakout_Confirmed",
            "Breakdown_Confirmed",
            "Setup_Score",
            "Score_Close_Above_SMA20",
            "Score_SMA20_Above_SMA50",
            "Score_Close_Above_SMA200",
            "Score_Positive_Return_20D",
            "Score_RSI_50_70",
            "Score_Breakout_Bonus",
            "Score_Breakdown_Penalty",
            "Assessment"
        ]

        print("\n" + "=" * 120)
        print("MULTI-STOCK SCANNER RESULTS")
        print("=" * 120)

        print(
            results[display_columns]
            .round(2)
            .to_string(index=False)
        )

        print(f"\nSuccessful stocks: {len(records)}")
        print(f"Failed stocks:     {len(errors)}")
        print(f"\nCombined report:\n{report_path.resolve()}")
        print(f"\nLatest report:\n{latest_report.resolve()}")

    else:
        print("\nNo stocks were successfully analyzed.")

    if errors:
        error_path = (
            REPORT_ARCHIVE_FOLDER / f"scanner_errors_{timestamp}.csv"
        )

        pd.DataFrame(errors).to_csv(error_path, index=False)
        print(f"\nError report:\n{error_path.resolve()}")

    print(f"\nLatest/current CSV folder: {DATA_FOLDER.resolve()}")
    print(f"Archived timestamped reports: {REPORT_ARCHIVE_FOLDER.resolve()}")
    print(f"Chart folder: {CHART_FOLDER.resolve()}")



# =========================================================
# 11. BASIC HISTORICAL BACKTEST
# =========================================================

def _plain_date_index(frame):
    """Return a copy with timezone-free normalized dates."""
    result = frame.copy()
    idx = pd.DatetimeIndex(result.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    result.index = idx.normalize()
    result = result[~result.index.duplicated(keep="last")]
    return result.sort_index()


def _calculate_aligned_benchmark(data, start_date, end_date, start_price_field="Open"):
    """Calculate buy-and-hold over an explicitly aligned reporting window.

    For a completed strategy, use the opening price on its first actual entry
    date and the closing price on its final exit date. For a no-trade run, the
    caller can request Close-to-Close over the available indicator-ready window.
    Missing dates/prices are reported as unavailable instead of silently using
    an unrelated first/last row in the downloaded dataset.
    """
    start_stamp = pd.Timestamp(start_date).tz_localize(None).normalize()
    end_stamp = pd.Timestamp(end_date).tz_localize(None).normalize()
    result = {
        "Benchmark_Start_Date": start_stamp.date().isoformat(),
        "Benchmark_End_Date": end_stamp.date().isoformat(),
        "Benchmark_Start_Price_Field": start_price_field,
        "Benchmark_End_Price_Field": "Close",
        "Benchmark_Start_Price": np.nan,
        "Benchmark_End_Price": np.nan,
        "Benchmark_Return_Method": (
            f"{start_price_field}-to-Close over the aligned reporting window"
        ),
        "Benchmark_Session_Count": np.nan,
        "Benchmark_Status": "Unavailable: date or price missing",
        "Benchmark_Investment_Exposure": (
            "Continuous buy-and-hold exposure; strategy may be in cash between trades"
        ),
        "Benchmark_Assumed_Round_Trip_Cost_Pct": BACKTEST_ROUND_TRIP_COST_PCT,
        "Stock_Buy_And_Hold_Return_Pct": np.nan,
        "Stock_Buy_And_Hold_Gross_Return_Pct": np.nan,
        "Stock_Buy_And_Hold_Net_Return_Pct": np.nan,
    }

    if end_stamp < start_stamp:
        result["Benchmark_Status"] = "Unavailable: end date precedes start date"
        return result

    normalized = _plain_date_index(data)
    if start_stamp not in normalized.index or end_stamp not in normalized.index:
        return result
    if start_price_field not in normalized.columns or "Close" not in normalized.columns:
        result["Benchmark_Status"] = "Unavailable: required price column missing"
        return result

    start_price = pd.to_numeric(normalized.at[start_stamp, start_price_field], errors="coerce")
    end_price = pd.to_numeric(normalized.at[end_stamp, "Close"], errors="coerce")
    if pd.isna(start_price) or pd.isna(end_price) or start_price <= 0 or end_price <= 0:
        return result

    start_position = int(normalized.index.get_loc(start_stamp))
    end_position = int(normalized.index.get_loc(end_stamp))
    result.update({
        "Benchmark_Start_Price": float(start_price),
        "Benchmark_End_Price": float(end_price),
        "Benchmark_Session_Count": end_position - start_position + 1,
        "Benchmark_Status": "Completed: dates aligned",
        "Stock_Buy_And_Hold_Return_Pct": float((end_price / start_price - 1) * 100),
        "Stock_Buy_And_Hold_Gross_Return_Pct": float((end_price / start_price - 1) * 100),
        "Stock_Buy_And_Hold_Net_Return_Pct": float(
            (end_price / start_price - 1) * 100 - BACKTEST_ROUND_TRIP_COST_PCT
        ),
    })
    return result


def _build_daily_mark_to_market_curve(data, trades_df, start_pos):
    """Build a close-to-close equity curve including open-trade P&L.

    The account is assumed to be fully invested in one stock per trade, with
    no overlapping trades. While a trade is open, daily equity is marked using
    that session's Close relative to the trade's Entry_Price. The configured
    round-trip cost is subtracted from the trade's marked return, matching the
    existing additive-cost convention used by the strategy report.
    """
    dates = data.index[start_pos:]
    curve = pd.DataFrame(index=dates)
    curve.index.name = "Date"
    curve["Close"] = data.loc[dates, "Close"].astype(float)
    curve["Strategy_Equity"] = np.nan
    curve["In_Position"] = False

    current_equity = 1.0
    cursor_pos = start_pos
    trade_close_equities = [1.0]

    for _, trade in trades_df.iterrows():
        entry_date = pd.Timestamp(trade["Entry_Date"]).normalize()
        exit_date = pd.Timestamp(trade["Exit_Date"]).normalize()
        entry_pos = int(data.index.get_loc(entry_date))
        exit_pos = int(data.index.get_loc(exit_date))

        if entry_pos < cursor_pos:
            raise ValueError(
                "Backtest trades overlap or are not ordered by entry date."
            )

        # Account remains in cash between completed trades and the next entry.
        if cursor_pos < entry_pos:
            cash_dates = data.index[cursor_pos:entry_pos]
            curve.loc[cash_dates, "Strategy_Equity"] = current_equity

        entry_price = float(trade["Entry_Price"])
        cost_fraction = (
            float(trade["Assumed_Round_Trip_Cost_Pct"]) / 100.0
        )

        # Daily close mark-to-market while the position is open.
        for position in range(entry_pos, exit_pos + 1):
            date = data.index[position]
            close_price = float(data.iloc[position]["Close"])
            marked_equity = current_equity * (
                close_price / entry_price - cost_fraction
            )
            curve.at[date, "Strategy_Equity"] = marked_equity
            curve.at[date, "In_Position"] = True

        # Realized equity after the trade uses the established net-return rule.
        net_return_fraction = float(trade["Net_Return_Pct"]) / 100.0
        current_equity *= 1.0 + net_return_fraction
        curve.at[exit_date, "Strategy_Equity"] = current_equity
        trade_close_equities.append(current_equity)
        cursor_pos = exit_pos + 1

    # Any period after the final trade is cash in this simplified model.
    if cursor_pos < len(data):
        cash_dates = data.index[cursor_pos:]
        curve.loc[cash_dates, "Strategy_Equity"] = current_equity

    curve["Strategy_Equity"] = curve["Strategy_Equity"].ffill().fillna(1.0)
    curve["Equity_Peak"] = curve["Strategy_Equity"].cummax()
    curve["Daily_Drawdown_Pct"] = (
        curve["Strategy_Equity"] / curve["Equity_Peak"] - 1.0
    ) * 100.0

    daily_mtm_drawdown = (
        float(curve["Daily_Drawdown_Pct"].min())
        if not curve.empty else 0.0
    )

    # Describe the worst peak-to-trough drawdown and whether equity recovered.
    drawdown_details = {
        "Max_Drawdown_Peak_Date": None,
        "Max_Drawdown_Trough_Date": None,
        "Max_Drawdown_Recovery_Date": None,
        "Max_Drawdown_Peak_To_Trough_Sessions": 0,
        "Max_Drawdown_Peak_To_Recovery_Sessions": None,
    }
    if not curve.empty and daily_mtm_drawdown < 0:
        trough_pos = int(curve["Daily_Drawdown_Pct"].to_numpy().argmin())
        peak_equity = float(curve.iloc[trough_pos]["Equity_Peak"])
        equity_before_trough = curve["Strategy_Equity"].iloc[:trough_pos + 1]
        peak_candidates = np.flatnonzero(
            np.isclose(equity_before_trough.to_numpy(), peak_equity, rtol=1e-10, atol=1e-12)
        )
        peak_pos = int(peak_candidates[-1]) if len(peak_candidates) else 0
        peak_date = pd.Timestamp(curve.index[peak_pos]).date().isoformat()
        trough_date = pd.Timestamp(curve.index[trough_pos]).date().isoformat()
        drawdown_details["Max_Drawdown_Peak_Date"] = peak_date
        drawdown_details["Max_Drawdown_Trough_Date"] = trough_date
        drawdown_details["Max_Drawdown_Peak_To_Trough_Sessions"] = trough_pos - peak_pos

        recovery_positions = np.flatnonzero(
            curve["Strategy_Equity"].iloc[trough_pos + 1:].to_numpy() >= peak_equity
        )
        if len(recovery_positions):
            recovery_pos = trough_pos + 1 + int(recovery_positions[0])
            drawdown_details["Max_Drawdown_Recovery_Date"] = (
                pd.Timestamp(curve.index[recovery_pos]).date().isoformat()
            )
            drawdown_details["Max_Drawdown_Peak_To_Recovery_Sessions"] = recovery_pos - peak_pos

    trade_close_series = pd.Series(trade_close_equities, dtype=float)
    trade_close_peaks = trade_close_series.cummax()
    trade_close_drawdown = (
        (trade_close_series / trade_close_peaks - 1.0) * 100.0
    )
    trade_close_max_drawdown = (
        float(trade_close_drawdown.min())
        if not trade_close_drawdown.empty else 0.0
    )

    return (
        curve.reset_index(), current_equity, daily_mtm_drawdown,
        trade_close_max_drawdown, drawdown_details
    )


def backtest_strategy(symbol, data):
    """
    Baseline test of a simple, explicit rule:
    - Signal after the close when score >= threshold
    - Require positive relative strength when enabled
    - Do not enter when a breakdown is confirmed
    - Enter next session at Open
    - Exit at the Close after the configured number of sessions
    - Subtract an illustrative round-trip cost

    Only one trade can be open at a time. This is educational research,
    not an assertion that the rule is profitable or suitable for live trading.
    """
    hold_sessions = BACKTEST_HOLDING_SESSIONS
    trades = []
    if data is None or data.empty:
        return pd.DataFrame(), {"Status": "No data"}

    data = data.copy()

    # Normalize dates for a stable comparison with the fixed test boundary.
    index = pd.DatetimeIndex(data.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    data.index = index.normalize()
    data = data[~data.index.duplicated(keep="last")].sort_index()

    if len(data) <= 200 + hold_sessions:
        return pd.DataFrame(), {
            "Status": "Insufficient historical sessions after warm-up"
        }

    # Require the main indicators to be available before evaluating signals.
    first_ready_positions = np.flatnonzero(data["SMA_200"].notna().to_numpy())
    if len(first_ready_positions) == 0:
        return pd.DataFrame(), {"Status": "SMA 200 is unavailable"}

    configured_start = pd.Timestamp(BACKTEST_TEST_START_DATE)
    fixed_start_pos = int(data.index.searchsorted(configured_start))
    start_pos = max(int(first_ready_positions[0]), fixed_start_pos)

    if start_pos >= len(data) - hold_sessions:
        return pd.DataFrame(), {
            "Status": (
                "Insufficient data on or after the fixed backtest "
                f"start date {BACKTEST_TEST_START_DATE}"
            )
        }

    last_exit_pos = -1
    last_possible_signal_pos = len(data) - hold_sessions - 1

    for signal_pos in range(start_pos, last_possible_signal_pos + 1):
        if signal_pos <= last_exit_pos:
            continue

        signal = data.iloc[signal_pos]
        score = calculate_setup_score(signal)
        relative_strength = signal.get("Relative_Strength_20D_Pct", np.nan)
        breakdown = bool(signal.get("Breakdown_Confirmed", False))

        if score < BACKTEST_MIN_SETUP_SCORE:
            continue
        if breakdown:
            continue
        if BACKTEST_REQUIRE_POSITIVE_RELATIVE_STRENGTH and (
            pd.isna(relative_strength) or relative_strength <= 0
        ):
            continue

        entry_pos = signal_pos + 1
        exit_pos = entry_pos + hold_sessions - 1
        if exit_pos >= len(data):
            break

        entry_row = data.iloc[entry_pos]
        exit_row = data.iloc[exit_pos]
        entry_price = entry_row["Open"]
        exit_price = exit_row["Close"]

        if (
            pd.isna(entry_price) or pd.isna(exit_price)
            or entry_price <= 0 or exit_price <= 0
        ):
            continue

        gross_return = (exit_price / entry_price - 1) * 100
        cost_pct = BACKTEST_ROUND_TRIP_COST_PCT
        net_return = gross_return - cost_pct

        score_details = calculate_score_details(signal)
        trades.append({
            "Symbol": symbol,
            "Signal_Date": pd.Timestamp(data.index[signal_pos]).date().isoformat(),
            "Entry_Date": pd.Timestamp(data.index[entry_pos]).date().isoformat(),
            "Exit_Date": pd.Timestamp(data.index[exit_pos]).date().isoformat(),
            "Entry_Price": float(entry_price),
            "Exit_Price": float(exit_price),
            "Holding_Sessions": hold_sessions,
            "Signal_Setup_Score": int(score),
            "Signal_Relative_Strength_20D_Pct": relative_strength,
            "Signal_RSI_14": signal.get("RSI_14", np.nan),
            "Signal_Relative_Volume_20D": signal.get("Relative_Volume_20D", np.nan),
            "Gross_Return_Pct": gross_return,
            "Assumed_Round_Trip_Cost_Pct": cost_pct,
            "Net_Return_Pct": net_return,
            "Outcome": "Win" if net_return > 0 else "Loss/Flat",
            **{key: value for key, value in score_details.items() if key != "Setup_Score"}
        })
        last_exit_pos = exit_pos

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        test_start = pd.Timestamp(data.index[start_pos]).date().isoformat()
        test_end = pd.Timestamp(data.index[-1]).date().isoformat()
        benchmark = _calculate_aligned_benchmark(
            data, test_start, test_end, start_price_field="Close"
        )
        benchmark["Benchmark_Return_Method"] = (
            "Close-to-Close over indicator-ready window (no qualifying trades)"
        )
        (
            equity_curve_df, final_equity, daily_mtm_drawdown,
            trade_close_drawdown, drawdown_details
        ) = _build_daily_mark_to_market_curve(data, trades_df, start_pos)
        summary = {
            "Symbol": symbol,
            "Status": "No qualifying trades found with current rules",
            "Configured_Backtest_Start_Date": BACKTEST_TEST_START_DATE,
            "Test_Start": test_start,
            "Test_End": test_end,
            "First_Signal_Date": None,
            "Holding_Sessions": hold_sessions,
            "Minimum_Setup_Score": BACKTEST_MIN_SETUP_SCORE,
            "Require_Positive_Relative_Strength": BACKTEST_REQUIRE_POSITIVE_RELATIVE_STRENGTH,
            "Assumed_Round_Trip_Cost_Pct": BACKTEST_ROUND_TRIP_COST_PCT,
            "Number_Of_Trades": 0,
            "Win_Rate_Pct": np.nan,
            "Average_Gross_Trade_Return_Pct": np.nan,
            "Average_Net_Trade_Return_Pct": np.nan,
            "Median_Net_Trade_Return_Pct": np.nan,
            "Profit_Factor": np.nan,
            "Compounded_Strategy_Return_Pct": 0.0,
            "Final_Strategy_Equity": 1.0,
            "Maximum_Trade_Close_Drawdown_Pct": trade_close_drawdown,
            "Maximum_Daily_Mark_To_Market_Drawdown_Pct": daily_mtm_drawdown,
            **drawdown_details,
            **benchmark,
        }
        return trades_df, summary, equity_curve_df

    # Build daily close mark-to-market equity, including open-trade losses.
    (
        equity_curve_df, equity, daily_mtm_drawdown,
        trade_close_drawdown, drawdown_details
    ) = _build_daily_mark_to_market_curve(data, trades_df, start_pos)

    net_returns = trades_df["Net_Return_Pct"].astype(float)
    gross_returns = trades_df["Gross_Return_Pct"].astype(float)
    losers = net_returns[net_returns < 0]
    winners = net_returns[net_returns > 0]
    if len(losers) and losers.abs().sum() > 0:
        profit_factor = float(winners.sum() / losers.abs().sum())
    else:
        profit_factor = np.nan

    # The reporting/benchmark window starts when the strategy first enters a
    # position and ends when its final position exits. Signals are recorded
    # separately because the first signal occurs before the actual entry.
    test_start = trades_df["Entry_Date"].iloc[0]
    test_end = trades_df["Exit_Date"].iloc[-1]
    benchmark = _calculate_aligned_benchmark(
        data, test_start, test_end, start_price_field="Open"
    )
    summary = {
        "Symbol": symbol,
        "Status": "Completed",
        "Configured_Backtest_Start_Date": BACKTEST_TEST_START_DATE,
        "Test_Start": test_start,
        "Test_End": test_end,
        "First_Signal_Date": trades_df["Signal_Date"].iloc[0],
        "Holding_Sessions": hold_sessions,
        "Minimum_Setup_Score": BACKTEST_MIN_SETUP_SCORE,
        "Require_Positive_Relative_Strength": BACKTEST_REQUIRE_POSITIVE_RELATIVE_STRENGTH,
        "Assumed_Round_Trip_Cost_Pct": BACKTEST_ROUND_TRIP_COST_PCT,
        "Number_Of_Trades": int(len(trades_df)),
        "Win_Rate_Pct": float((net_returns > 0).mean() * 100),
        "Average_Gross_Trade_Return_Pct": float(gross_returns.mean()),
        "Average_Net_Trade_Return_Pct": float(net_returns.mean()),
        "Median_Net_Trade_Return_Pct": float(net_returns.median()),
        "Profit_Factor": profit_factor,
        "Compounded_Strategy_Return_Pct": float((equity - 1) * 100),
        "Final_Strategy_Equity": float(equity),
        "Maximum_Trade_Close_Drawdown_Pct": trade_close_drawdown,
        "Maximum_Daily_Mark_To_Market_Drawdown_Pct": daily_mtm_drawdown,
        **drawdown_details,
        **benchmark,
    }
    return trades_df, summary, equity_curve_df


def backtest_one_stock():
    """Download one stock, run the baseline backtest, and save reports."""
    try:
        symbol = get_stock_symbol()
        data = download_stock_data(symbol)
        data = calculate_indicators(data)
        benchmark_data = download_nifty50_data()
        data = add_relative_strength(data, benchmark_data)

        trades_df, summary, equity_curve_df = backtest_strategy(symbol, data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_symbol = (
            symbol.replace(".", "_").replace("^", "")
            .replace("/", "_").replace("\\", "_")
        )
        trades_path = REPORT_ARCHIVE_FOLDER / f"backtest_trades_{safe_symbol}_{timestamp}.csv"
        summary_path = REPORT_ARCHIVE_FOLDER / f"backtest_summary_{safe_symbol}_{timestamp}.csv"
        equity_path = (
            REPORT_ARCHIVE_FOLDER
            / f"backtest_equity_{safe_symbol}_{timestamp}.csv"
        )
        latest_trades_path = DATA_FOLDER / "backtest_trades_latest.csv"
        latest_summary_path = DATA_FOLDER / "backtest_summary_latest.csv"
        latest_equity_path = DATA_FOLDER / "backtest_equity_latest.csv"

        trades_df.to_csv(trades_path, index=False)
        trades_df.to_csv(latest_trades_path, index=False)
        equity_curve_df.to_csv(equity_path, index=False)
        equity_curve_df.to_csv(latest_equity_path, index=False)
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        pd.DataFrame([summary]).to_csv(latest_summary_path, index=False)

        print("\n" + "=" * 64)
        print(f"BASELINE HISTORICAL BACKTEST: {symbol}")
        print("=" * 64)
        for key, value in summary.items():
            if isinstance(value, (float, np.floating)) and pd.notna(value):
                print(f"{key.replace('_', ' '):<43} {value:,.2f}")
            else:
                print(f"{key.replace('_', ' '):<43} {value}")

        if trades_df.empty:
            print("\nNo qualifying trades were generated. That is a result, not a download error.")
        else:
            show_cols = [
                "Signal_Date", "Entry_Date", "Exit_Date", "Entry_Price",
                "Exit_Price", "Signal_Setup_Score", "Gross_Return_Pct",
                "Assumed_Round_Trip_Cost_Pct", "Net_Return_Pct", "Outcome"
            ]
            print("\n--- Recent Backtest Trades ---")
            print(trades_df[show_cols].tail(15).round(2).to_string(index=False))

        print(f"\nTrade details CSV: {trades_path.resolve()}")
        print(f"Latest trade report: {latest_trades_path.resolve()}")
        print(f"Equity curve CSV: {equity_path.resolve()}")
        print(f"Latest equity curve: {latest_equity_path.resolve()}")
        print(f"Summary CSV: {summary_path.resolve()}")
        print(f"Latest summary: {latest_summary_path.resolve()}")
        print(f"Archived timestamped reports: {REPORT_ARCHIVE_FOLDER.resolve()}")
        print("\nImportant: this is a simplified historical simulation, not a prediction.")
        print("Costs are illustrative; taxes, slippage, liquidity, and live execution are not fully modelled.")

    except Exception as error:
        print(f"\nCould not complete backtest: {error}")
        traceback.print_exc()


# =========================================================
# 12. MAIN MENU
# =========================================================

def main():
    while True:
        print("\n" + "=" * 50)
        print("INDIAN STOCK RESEARCH & SWING ANALYZER")
        print("=" * 50)
        print("1. Analyze one stock")
        print("2. Scan multiple stocks")
        print("3. Backtest one stock (historical simulation)")
        print("4. Exit")

        choice = input("\nChoose an option (1-4): ").strip()

        if choice == "1":
            analyze_stock()

        elif choice == "2":
            scan_multiple_stocks()

        elif choice == "3":
            backtest_one_stock()

        elif choice == "4":
            print("\nProgram closed.")
            break

        else:
            print("\nInvalid choice. Enter 1, 2, 3, or 4.")


# =========================================================
# 13. START PROGRAM
# =========================================================

if __name__ == "__main__":
    archive_existing_reports()
    main()
