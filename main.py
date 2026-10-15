
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

SUPPORT_RESISTANCE_PERIOD = 60
VOLUME_AVERAGE_PERIOD = 20
HIGH_VOLUME_MULTIPLIER = 1.5
MAX_SCAN_STOCKS = 50

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


# =========================================================
# 2. STOCK NAME AND SYMBOL HANDLING
# =========================================================

ALIASES = {
    "HDFC": "HDFCBANK.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "HDFC BANK LIMITED": "HDFCBANK.NS",
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
                        "range": "5y",
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
                period="5y",
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

def calculate_setup_score(latest):
    score = 0

    close = latest["Close"]
    sma20 = latest["SMA_20"]
    sma50 = latest["SMA_50"]
    sma200 = latest["SMA_200"]
    return20 = latest["Return_20D_Pct"]
    rsi = latest["RSI_14"]

    if pd.notna(sma20) and close > sma20:
        score += 1

    if pd.notna(sma20) and pd.notna(sma50) and sma20 > sma50:
        score += 1

    if pd.notna(sma200) and close > sma200:
        score += 1

    if pd.notna(return20) and return20 > 0:
        score += 1

    if pd.notna(rsi) and 50 <= rsi <= 70:
        score += 1

    if bool(latest["Breakout_Confirmed"]):
        score += 2

    if bool(latest["Breakdown_Confirmed"]):
        score -= 2

    return score


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

    score = calculate_setup_score(latest)

    return {
        "Symbol": symbol,
        "Latest_Date": data.index[-1].date().isoformat(),
        "Close": close,
        "Daily_Return_Pct": latest["Daily_Return_Pct"],
        "Return_20D_Pct": latest["Return_20D_Pct"],
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

    for position, symbol in enumerate(symbols, start=1):
        print(f"\n[{position}/{len(symbols)}] {symbol}")

        try:
            data = download_stock_data(symbol)
            data = calculate_indicators(data)

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
            DATA_FOLDER / f"scanner_results_{timestamp}.csv"
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
            "RSI_14",
            "Relative_Volume_20D",
            "Breakout_Confirmed",
            "Breakdown_Confirmed",
            "Setup_Score",
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
            DATA_FOLDER / f"scanner_errors_{timestamp}.csv"
        )

        pd.DataFrame(errors).to_csv(error_path, index=False)
        print(f"\nError report:\n{error_path.resolve()}")

    print(f"\nCSV folder:   {DATA_FOLDER.resolve()}")
    print(f"Chart folder: {CHART_FOLDER.resolve()}")


# =========================================================
# 11. MAIN MENU
# =========================================================

def main():
    while True:
        print("\n" + "=" * 50)
        print("INDIAN STOCK RESEARCH & SWING ANALYZER")
        print("=" * 50)
        print("1. Analyze one stock")
        print("2. Scan multiple stocks")
        print("3. Exit")

        choice = input("\nChoose an option (1-3): ").strip()

        if choice == "1":
            analyze_stock()

        elif choice == "2":
            scan_multiple_stocks()

        elif choice == "3":
            print("\nProgram closed.")
            break

        else:
            print("\nInvalid choice. Enter 1, 2, or 3.")


# =========================================================
# 12. START PROGRAM
# =========================================================

if __name__ == "__main__":
    main()
