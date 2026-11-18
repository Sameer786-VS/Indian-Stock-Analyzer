
"""Stage 3.7: polished HTML research reports and charts.

Presentation-only module. It does not download data, recalculate indicators,
change the setup score, modify backtests, or generate trading instructions.
"""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import math
import os
import re
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------


def safe_filename(value: Any) -> str:
    """Return a filesystem-safe name for a stock symbol or report label."""
    text = str(value or "report").strip().upper()
    text = re.sub(r"[^A-Z0-9_-]+", "_", text)
    return text.strip("_") or "REPORT"


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _fmt(value: Any, suffix: str = "", decimals: int = 2) -> str:
    number = _number(value)
    if not math.isfinite(number):
        return "N/A"
    return f"{number:,.{decimals}f}{suffix}"


def _text(value: Any, default: str = "Not available") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    rendered = str(value).strip()
    if not rendered or rendered.lower() in {"nan", "none", "nat"}:
        return default
    return rendered


def _as_bool(value: Any) -> bool:
    """Convert common CSV/Pandas boolean representations safely."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value) if math.isfinite(float(value)) else False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _relative_asset(report_path: Path, asset_path: Path) -> str:
    return Path(os.path.relpath(asset_path, report_path.parent)).as_posix()


def _metric_card(label: str, value: str, note: str = "", tone: str = "") -> str:
    tone_class = f" metric-{escape(tone)}" if tone else ""
    return (
        f'<div class="metric{tone_class}">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{escape(value)}</div>'
        f'<div class="metric-note">{escape(note)}</div>'
        "</div>"
    )


_COLUMN_LABELS = {
    "Symbol": "Symbol",
    "Company_Name": "Company",
    "NSE_Industry": "Industry",
    "Close": "Close (₹)",
    "Return_20D_Pct": "20-session return (%)",
    "Relative_Strength_20D_Pct": "Relative strength (pp)",
    "Setup_Score": "Setup score",
    "Research_Conclusion": "Research conclusion",
    "Sector_Rank_20D": "Rank",
    "Scanned_Constituents": "Scanned stocks",
    "Coverage_Pct": "Coverage (%)",
    "Sector_Return_20D_Pct": "Group return (%)",
    "Sector_Relative_Strength_20D_Pct": "Relative strength (pp)",
    "Strength_Status": "Group status",
}


def _html_table(
    frame: pd.DataFrame,
    columns: list[str],
    max_rows: int | None = 12,
) -> str:
    available = [column for column in columns if column in frame.columns]
    if not available or frame.empty:
        return '<p class="muted">No report data available for this section.</p>'

    shown = frame[available].copy()
    if max_rows is not None:
        shown = shown.head(max_rows)

    header = "".join(
        f"<th>{escape(_COLUMN_LABELS.get(col, col.replace('_', ' ')))}</th>"
        for col in available
    )
    rows: list[str] = []
    for _, row in shown.iterrows():
        cells: list[str] = []
        for col in available:
            value = row[col]
            if isinstance(value, (float, np.floating, int, np.integer)):
                rendered = "—" if pd.isna(value) else f"{float(value):,.2f}"
                if isinstance(value, (int, np.integer)) and not pd.isna(value):
                    rendered = str(int(value))
            else:
                rendered = _text(value, "—")
            cells.append(f"<td>{escape(rendered)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        '<div class="table-wrap"><table><thead><tr>' + header
        + "</tr></thead><tbody>" + "".join(rows)
        + "</tbody></table></div>"
    )


def _base_html(title: str, subtitle: str, body: str) -> str:
    """Return a responsive, print-friendly HTML shell."""
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__REPORT_TITLE__</title>
<style>
:root { color-scheme: light; --ink:#14263a; --muted:#64748b; --line:#e2e8f0; --paper:#ffffff; --bg:#f3f6fa; --accent:#1d5d87; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:#17212b; font:14px/1.55 Inter,Segoe UI,Arial,sans-serif; }
main { max-width:1320px; margin:30px auto; padding:0 24px 48px; }
header { position:relative; overflow:hidden; background:linear-gradient(115deg,#102338,#1d3d58); color:#fff; padding:28px 32px; border-radius:16px; box-shadow:0 8px 24px rgba(15,35,56,.12); }
.eyebrow { margin:0 0 7px; color:#b7c9d8; font-size:10px; font-weight:700; letter-spacing:.15em; text-transform:uppercase; }
h1 { margin:0 0 7px; font-size:clamp(22px,3vw,30px); line-height:1.2; letter-spacing:-.02em; }
.header-subtitle { margin:0; color:#d4e0e9; font-size:12px; }
.header-rule { height:3px; width:56px; background:#68a9cf; border-radius:3px; margin-top:17px; }
h2 { margin:30px 0 12px; color:var(--ink); font-size:18px; line-height:1.3; letter-spacing:-.01em; }
h3 { margin:0 0 8px; font-size:14px; color:var(--ink); }
p { margin:7px 0; }
.metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); gap:12px; margin:18px 0 22px; }
.metric,.panel { background:var(--paper); border:1px solid var(--line); border-radius:12px; padding:15px 16px; box-shadow:0 2px 7px rgba(15,35,56,.025); }
.metric { min-height:102px; border-top:3px solid #cbd8e3; }
.metric-positive { border-top-color:#4b9b78; }
.metric-caution { border-top-color:#d09a45; }
.metric-negative { border-top-color:#c56d6d; }
.metric-label { color:#64748b; font-size:10px; font-weight:700; line-height:1.35; text-transform:uppercase; letter-spacing:.065em; }
.metric-value { color:var(--ink); font-size:22px; font-weight:750; letter-spacing:-.025em; margin:8px 0 3px; }
.metric-note,.muted { color:var(--muted); font-size:11px; }
.grid2 { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; align-items:start; }
.grid2 > section { min-width:0; }
.chart-panel { background:#fff; border:1px solid var(--line); border-radius:12px; padding:12px; }
img.chart { display:block; width:100%; height:auto; background:#fff; border:1px solid var(--line); border-radius:10px; }
.chart-caption { color:var(--muted); font-size:11px; margin:7px 2px 0; }
.table-wrap { overflow-x:auto; background:#fff; border:1px solid var(--line); border-radius:10px; }
table { width:100%; border-collapse:collapse; font-size:12px; }
th,td { text-align:left; vertical-align:top; padding:10px 12px; border-bottom:1px solid #edf1f5; }
th { background:#f7f9fc; color:#42566b; font-size:10px; text-transform:uppercase; letter-spacing:.045em; white-space:nowrap; }
tbody tr:last-child td { border-bottom:0; }
tbody tr:hover { background:#f8fbfd; }
.note { background:#fff; border:1px solid var(--line); border-left:4px solid #6c91ad; padding:14px 16px; border-radius:10px; }
.note-title { color:var(--ink); font-size:12px; font-weight:700; margin:0 0 4px; }
.status-pill { display:inline-block; border-radius:999px; padding:3px 9px; font-size:10px; font-weight:700; background:#edf3f7; color:#36566f; }
.status-good { background:#eaf5ef; color:#286346; }
.status-warn { background:#fff4df; color:#805b18; }
.status-bad { background:#fbecec; color:#8a3434; }
.section-intro { color:var(--muted); font-size:12px; margin:-5px 0 12px; }
.footer { margin-top:30px; padding-top:14px; border-top:1px solid var(--line); color:#718096; font-size:11px; }
@media(max-width:760px) { main { margin:14px auto; padding:0 12px 28px; } header { padding:22px 20px; } .grid2 { grid-template-columns:1fr; } .metrics { grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px; } .metric { min-height:92px; padding:12px; } .metric-value { font-size:19px; } }
@media(max-width:390px) { .metrics { grid-template-columns:1fr 1fr; } }
@media print { body { background:#fff; } main { margin:0; max-width:none; padding:0; } header { break-inside:avoid; box-shadow:none; } .panel,.metric,.chart-panel { break-inside:avoid; box-shadow:none; } }
</style></head><body><main>
<header><p class="eyebrow">Indian Stock Research &amp; Swing Analyzer · Stage 3.7</p><h1>__REPORT_TITLE__</h1><p class="header-subtitle">__REPORT_SUBTITLE__</p><div class="header-rule"></div></header>
__REPORT_BODY__
<footer class="footer">Generated by Indian Stock Research &amp; Swing Analyzer — Stage 3.7. Research information only; not investment advice or a trading recommendation.</footer>
</main></body></html>"""
    return (
        template
        .replace("__REPORT_TITLE__", escape(title))
        .replace("__REPORT_SUBTITLE__", escape(subtitle))
        .replace("__REPORT_BODY__", body)
    )


# ---------------------------------------------------------------------------
# Single-stock chart and report
# ---------------------------------------------------------------------------


def create_stock_chart(symbol: str, data: pd.DataFrame, chart_path: Path) -> Path:
    """Create a three-panel price/SMA, RSI and volume chart."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Stock chart requires a non-empty price DataFrame.")
    if "Close" not in data.columns:
        raise ValueError("Stock chart requires a Close column.")

    chart_data = data.tail(252).copy()
    chart_data.index = pd.to_datetime(chart_data.index)
    chart_data = chart_data.sort_index()
    chart_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        3, 1, figsize=(13, 9.5), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.15, 1.35]},
    )
    price_ax, rsi_ax, volume_ax = axes

    price_ax.plot(
        chart_data.index, chart_data["Close"],
        label="Close", linewidth=1.8,
    )
    for period in (20, 50, 200):
        column = f"SMA_{period}"
        if column in chart_data.columns:
            price_ax.plot(
                chart_data.index, chart_data[column],
                label=f"SMA {period}", linewidth=1.05,
            )
    for column, label in (
        ("Support_60D", "Prior 60-session support"),
        ("Resistance_60D", "Prior 60-session resistance"),
    ):
        if column in chart_data.columns:
            price_ax.plot(
                chart_data.index, chart_data[column],
                label=label, linestyle="--", linewidth=.9,
            )

    latest_close = _number(chart_data["Close"].iloc[-1])
    if math.isfinite(latest_close):
        price_ax.scatter(
            [chart_data.index[-1]], [latest_close],
            s=28, zorder=5, label="Latest close",
        )
    price_ax.set_title(f"{symbol} — price, moving averages and prior range", pad=10)
    price_ax.set_ylabel("Price (₹)")
    price_ax.grid(True, alpha=.22)
    price_ax.legend(fontsize=8, ncol=3, loc="best", frameon=True)

    if "RSI_14" in chart_data.columns:
        rsi_ax.plot(
            chart_data.index, chart_data["RSI_14"],
            label="RSI 14", linewidth=1.25,
        )
        rsi_ax.axhline(70, linestyle="--", linewidth=.8, label="70 / 30 guide")
        rsi_ax.axhline(50, linestyle=":", linewidth=.8)
        rsi_ax.axhline(30, linestyle="--", linewidth=.8)
        rsi_ax.set_ylim(0, 100)
        rsi_ax.legend(fontsize=7, loc="upper left")
    rsi_ax.set_ylabel("RSI")
    rsi_ax.grid(True, alpha=.22)

    if "Volume" in chart_data.columns:
        volume_ax.bar(
            chart_data.index, chart_data["Volume"],
            label="Daily volume", width=.8,
        )
    if "Avg_Volume_20D" in chart_data.columns:
        volume_ax.plot(
            chart_data.index, chart_data["Avg_Volume_20D"],
            label="Prior 20-session average", linewidth=1.2,
        )
    volume_ax.set_ylabel("Shares")
    volume_ax.set_xlabel("Date")
    volume_ax.grid(True, alpha=.22)
    volume_ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout(pad=1.5)
    fig.savefig(chart_path, dpi=165, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def _price_action_summary(
    symbol: str,
    data: pd.DataFrame,
    record: Mapping[str, Any],
) -> tuple[str, str, str]:
    """Explain current price location separately from a fresh crossing event."""
    latest = data.iloc[-1]
    close = _number(latest.get("Close"))
    resistance = _number(latest.get("Resistance_60D"))
    support = _number(latest.get("Support_60D"))
    latest_date = pd.Timestamp(data.index[-1]).date().isoformat()

    fresh_breakout = _as_bool(
        record.get("Breakout_60D", latest.get("Breakout_60D", False))
    )
    fresh_breakdown = _as_bool(
        record.get("Breakdown_60D", latest.get("Breakdown_60D", False))
    )
    confirmed_breakout = _as_bool(
        record.get("Breakout_Confirmed", latest.get("Breakout_Confirmed", False))
    )
    confirmed_breakdown = _as_bool(
        record.get("Breakdown_Confirmed", latest.get("Breakdown_Confirmed", False))
    )

    last_signal_date = ""
    if "Breakout_60D" in data.columns:
        signal_mask = data["Breakout_60D"].map(_as_bool)
        signal_dates = data.index[signal_mask]
        if len(signal_dates):
            last_signal_date = pd.Timestamp(signal_dates[-1]).date().isoformat()

    location = "The latest close could not be compared with the available support/resistance levels."
    level_value = "Reference levels are unavailable."
    status_class = "status-pill"

    if math.isfinite(close) and math.isfinite(resistance) and resistance > 0 and close > resistance:
        distance = (close / resistance - 1) * 100
        location = (
            f"Price is above the prior 60-session resistance by {_fmt(distance, '%')} "
            f"(close ₹{_fmt(close)}, reference ₹{_fmt(resistance)})."
        )
        level_value = "Above prior resistance"
        status_class += " status-good"
    elif math.isfinite(close) and math.isfinite(support) and support > 0 and close < support:
        distance = (close / support - 1) * 100
        location = (
            f"Price is below the prior 60-session support by {_fmt(abs(distance), '%')} "
            f"(close ₹{_fmt(close)}, reference ₹{_fmt(support)})."
        )
        level_value = "Below prior support"
        status_class += " status-bad"
    elif math.isfinite(close) and math.isfinite(support) and math.isfinite(resistance):
        location = (
            f"Price is inside the prior 60-session range "
            f"(support ₹{_fmt(support)}; resistance ₹{_fmt(resistance)})."
        )
        level_value = "Inside prior range"

    if fresh_breakout:
        event = (
            f"Fresh 60-session breakout crossing detected on {latest_date}. "
            + ("Volume confirmation is present." if confirmed_breakout
               else "The breakout does not have the model's volume confirmation.")
        )
        status_class += " status-good"
    elif fresh_breakdown:
        event = (
            f"Fresh 60-session breakdown crossing detected on {latest_date}. "
            + ("Volume confirmation is present." if confirmed_breakdown
               else "The breakdown does not have the model's volume confirmation.")
        )
        status_class += " status-bad"
    else:
        event = (
            "No fresh crossover was triggered on the latest session. "
            "The scanner's breakout flag records the day price crosses the rolling "
            "60-session level; it is not simply a test of whether price is above that level today."
        )
        if last_signal_date:
            event += f" Latest earlier breakout flag in the available history: {last_signal_date}."

    return level_value, location + " " + event, status_class


def _stock_report_body(
    symbol: str,
    data: pd.DataFrame,
    record: Mapping[str, Any],
    chart_rel: str,
) -> str:
    latest = data.iloc[-1]
    close = _number(latest.get("Close"))
    one_day = _number(latest.get("Daily_Return_Pct"))
    five_day = _number(latest.get("Return_5D_Pct"))
    twenty_day = _number(latest.get("Return_20D_Pct"))
    relative = _number(latest.get("Relative_Strength_20D_Pct"))
    latest_date = pd.Timestamp(data.index[-1]).date().isoformat()
    score = _fmt(record.get("Setup_Score"), decimals=0)

    rsi = _number(latest.get("RSI_14"))
    relative_tone = "positive" if math.isfinite(relative) and relative > 0 else ("negative" if math.isfinite(relative) and relative < 0 else "")

    cards = "".join([
        _metric_card("Latest close", f"₹ {_fmt(close)}", latest_date),
        _metric_card("1-session return", _fmt(one_day, "%"), tone="positive" if one_day > 0 else "negative" if math.isfinite(one_day) else ""),
        _metric_card("5-session return", _fmt(five_day, "%"), tone="positive" if five_day > 0 else "negative" if math.isfinite(five_day) else ""),
        _metric_card("20-session return", _fmt(twenty_day, "%"), tone="positive" if twenty_day > 0 else "negative" if math.isfinite(twenty_day) else ""),
        _metric_card("Relative strength vs NIFTY 50", _fmt(relative, " pp"), "20-session return difference", relative_tone),
        _metric_card("Existing setup score", score, "Original score; not a recommendation"),
    ])

    technical_rows_data = [
        ("SMA 20", f"₹ {_fmt(latest.get('SMA_20'))}"),
        ("SMA 50", f"₹ {_fmt(latest.get('SMA_50'))}"),
        ("SMA 200", f"₹ {_fmt(latest.get('SMA_200'))}"),
        ("RSI 14", _fmt(latest.get("RSI_14"))),
        ("ATR 14", f"₹ {_fmt(latest.get('ATR_14'))}"),
        ("ATR percentage", _fmt(latest.get("ATR_Pct"), "%")),
        ("60-session support", f"₹ {_fmt(latest.get('Support_60D'))}"),
        ("60-session resistance", f"₹ {_fmt(latest.get('Resistance_60D'))}"),
        ("Relative volume", _fmt(latest.get("Relative_Volume_20D"), "x")),
    ]
    technical_rows = "".join(
        f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>"
        for label, value in technical_rows_data
    )

    score_names = [
        ("Close above SMA 20", "Score_Close_Above_SMA20"),
        ("SMA 20 above SMA 50", "Score_SMA20_Above_SMA50"),
        ("Close above SMA 200", "Score_Close_Above_SMA200"),
        ("Positive 20-session return", "Score_Positive_Return_20D"),
        ("RSI between 50 and 70", "Score_RSI_50_70"),
        ("Confirmed breakout bonus", "Score_Breakout_Bonus"),
        ("Confirmed breakdown penalty", "Score_Breakdown_Penalty"),
    ]
    score_rows = "".join(
        f"<tr><td>{escape(label)}</td><td>{escape(_fmt(record.get(key), decimals=0))}</td></tr>"
        for label, key in score_names if key in record
    )

    research_keys = [
        ("Trend evidence", "Research_Trend_State"),
        ("Momentum evidence", "Research_Momentum_State"),
        ("Benchmark evidence", "Research_Benchmark_State"),
        ("Volume context", "Research_Volume_State"),
        ("Research conclusion", "Research_Conclusion"),
        ("Positive evidence", "Research_Positive_Evidence"),
        ("Cautions", "Research_Caution_Evidence"),
        ("Missing data", "Research_Missing_Evidence"),
        ("Stock vs other sector peers", "Stock_vs_Sector_20D_Pct"),
        ("Peer comparison quality", "Peer_Comparison_Quality"),
    ]
    research_cells = []
    for label, key in research_keys:
        if key not in record:
            continue
        value = record.get(key)
        if key == "Research_Missing_Evidence" and _text(value, "").lower() in {"", "none", "no missing data", "not available"}:
            value = "No missing fields reported by the research summary."
        if key == "Stock_vs_Sector_20D_Pct":
            value = _fmt(value, " pp")
        research_cells.append(
            f"<tr><td>{escape(label)}</td><td>{escape(_text(value))}</td></tr>"
        )
    research_rows = "".join(research_cells)

    location_label, location_detail, status_class = _price_action_summary(symbol, data, record)
    caution_note = (
        "RSI is at or above 70; review whether momentum is extended."
        if math.isfinite(rsi) and rsi >= 70
        else "RSI is below the 70 reference level; this alone does not establish a trade setup."
    )

    return f"""
<section class="metrics">{cards}</section>
<h2>Price and technical chart</h2>
<div class="chart-panel"><img class="chart" src="{escape(chart_rel)}" alt="{escape(symbol)} price, RSI and volume chart"><p class="chart-caption">Daily close and moving averages · RSI 14 · daily volume compared with the prior 20-session average. Chart displays up to 252 recent sessions.</p></div>
<h2>Price location and breakout check</h2>
<div class="panel"><p><span class="status-pill {status_class}">{escape(location_label)}</span></p><p>{escape(location_detail)}</p><p class="muted">A fresh breakout is an event (the crossing session). Price can remain above a prior resistance level on later sessions without generating another fresh-breakout flag.</p></div>
<div class="grid2"><section><h2>Technical snapshot</h2><div class="panel"><div class="table-wrap"><table><thead><tr><th>Indicator</th><th>Latest value</th></tr></thead><tbody>{technical_rows}</tbody></table></div></div></section>
<section><h2>Research interpretation</h2><div class="panel"><p><strong>{escape(_text(record.get('Research_Conclusion'), 'Research conclusion unavailable'))}</strong></p><p>{escape(_text(record.get('Research_Trend_State'), 'Trend evidence unavailable'))}</p><p>{escape(_text(record.get('Research_Momentum_State'), caution_note))}</p><p class="muted">{escape(caution_note)}</p></div></section></div>
<div class="grid2"><section><h2>Original score breakdown</h2><div class="panel"><p class="muted">The existing scoring rules are displayed as recorded by Stage 3.6. Stage 3.7 does not change them.</p><div class="table-wrap"><table><thead><tr><th>Rule</th><th>Points</th></tr></thead><tbody>{score_rows}</tbody></table></div></div></section>
<section><h2>Explainable research evidence</h2><div class="panel"><div class="table-wrap"><table><thead><tr><th>Evidence area</th><th>Reading</th></tr></thead><tbody>{research_rows}</tbody></table></div></div></section></div>
<h2>How to read this report</h2><div class="note"><p class="note-title">Research, not a trading instruction</p><p>Relative strength is the stock's 20-session return minus the NIFTY 50's 20-session return, expressed in percentage points. The setup score is a screening calculation, not a probability, forecast, or buy/sell instruction. Review data quality, liquidity, corporate actions, valuation and risk separately before making any decision.</p></div>
"""


def generate_stock_report(
    symbol: str,
    data: pd.DataFrame,
    record: Mapping[str, Any],
    report_dir: Path | str,
    chart_dir: Path | str,
    timestamp: str | None = None,
) -> dict[str, Path]:
    """Write timestamped and latest HTML research reports plus chart PNGs."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Stock report requires a non-empty price DataFrame.")

    report_dir, chart_dir = Path(report_dir), Path(chart_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = safe_filename(symbol)

    archive_chart = chart_dir / f"stock_{slug}_{stamp}.png"
    latest_chart = chart_dir / f"stock_{slug}_latest.png"
    create_stock_chart(symbol, data, archive_chart)
    if latest_chart != archive_chart:
        latest_chart.write_bytes(archive_chart.read_bytes())

    archive_html = report_dir / f"stock_{slug}_{stamp}.html"
    latest_html = report_dir / f"stock_{slug}_latest.html"
    title = f"Stock Research Report — {symbol}"
    company = _text(record.get("Company_Name"), symbol)
    latest_date = pd.Timestamp(data.index[-1]).date().isoformat()
    subtitle = f"{company} · {symbol} · Data date: {latest_date} · Generated: {stamp}"

    archive_body = _stock_report_body(
        symbol, data, record, _relative_asset(archive_html, archive_chart)
    )
    latest_body = _stock_report_body(
        symbol, data, record, _relative_asset(latest_html, latest_chart)
    )
    archive_html.write_text(_base_html(title, subtitle, archive_body), encoding="utf-8")
    latest_html.write_text(_base_html(title, subtitle, latest_body), encoding="utf-8")
    return {
        "archive_html": archive_html,
        "latest_html": latest_html,
        "archive_chart": archive_chart,
        "latest_chart": latest_chart,
    }


# ---------------------------------------------------------------------------
# Scan charts and dashboards
# ---------------------------------------------------------------------------


def _create_bar_chart(
    frame: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    path: Path,
    limit: int = 10,
) -> Path | None:
    if frame.empty or label_col not in frame.columns or value_col not in frame.columns:
        return None
    plot_data = frame[[label_col, value_col]].copy()
    plot_data[value_col] = pd.to_numeric(plot_data[value_col], errors="coerce")
    plot_data = plot_data.dropna().sort_values(value_col, ascending=False).head(limit)
    if plot_data.empty:
        return None

    plot_data = plot_data.sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, max(4.2, .42 * len(plot_data))))
    bars = ax.barh(plot_data[label_col].astype(str), plot_data[value_col])
    ax.axvline(0, linewidth=.9, linestyle="--")
    ax.set_title(title, pad=12, fontsize=12, weight="bold")
    ax.set_xlabel("Percentage points vs NIFTY 50")
    ax.grid(axis="x", alpha=.22)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=8)
    ax.margins(x=.15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)
    return path


def _create_score_chart(frame: pd.DataFrame, path: Path) -> Path | None:
    """Avoid a misleading histogram for tiny scans; show stock-level scores."""
    if frame.empty or "Setup_Score" not in frame.columns:
        return None
    scores = pd.to_numeric(frame["Setup_Score"], errors="coerce")
    valid = scores.dropna()
    if valid.empty:
        return None

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    if len(valid) < 6:
        labels = (
            frame.loc[valid.index, "Symbol"].astype(str)
            if "Symbol" in frame.columns
            else pd.Series([f"Stock {i + 1}" for i in range(len(valid))], index=valid.index)
        )
        plot = pd.DataFrame({"symbol": labels, "score": valid}).sort_values("score")
        bars = ax.barh(plot["symbol"], plot["score"])
        ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=9)
        ax.set_title("Setup score by stock — small sample", pad=12, fontsize=12, weight="bold")
        ax.set_xlabel("Existing setup score")
        min_score = min(0, int(math.floor(valid.min())))
        max_score = max(5, int(math.ceil(valid.max())))
        ax.set_xlim(min_score, max_score + 1)
        ax.set_xticks(range(min_score, max_score + 1))
        ax.set_ylabel("Stock")
        ax.text(
            .99, .03,
            "Small sample: compare stocks individually; do not infer a distribution.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        )
    else:
        counts = valid.round().astype(int).value_counts().sort_index()
        bars = ax.bar(counts.index.astype(str), counts.values)
        ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
        ax.set_title("Existing setup-score distribution", pad=12, fontsize=12, weight="bold")
        ax.set_xlabel("Existing setup score")
        ax.set_ylabel("Number of stocks")
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)
    return path


def _create_sector_chart(sector_frame: pd.DataFrame, path: Path) -> Path | None:
    value_col = "Sector_Relative_Strength_20D_Pct"
    label_col = "NSE_Industry"
    if sector_frame.empty or value_col not in sector_frame or label_col not in sector_frame:
        return None
    plot = sector_frame[[label_col, value_col]].copy()
    plot[value_col] = pd.to_numeric(plot[value_col], errors="coerce")
    plot = plot.dropna().sort_values(value_col, ascending=False).head(18)
    if plot.empty:
        return None
    plot = plot.sort_values(value_col, ascending=True)

    fig, ax = plt.subplots(figsize=(11, max(5, .40 * len(plot))))
    bars = ax.barh(plot[label_col].astype(str), plot[value_col])
    ax.axvline(0, linewidth=.9, linestyle="--")
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=8)
    ax.set_title("Industry-group relative strength vs NIFTY 50", pad=12, fontsize=12, weight="bold")
    ax.set_xlabel("Percentage points")
    ax.grid(axis="x", alpha=.22)
    ax.margins(x=.16)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_scan_report(
    frame: pd.DataFrame,
    sector_frame: pd.DataFrame | None,
    report_dir: Path | str,
    chart_dir: Path | str,
    scan_label: str = "Stock scan",
    timestamp: str | None = None,
) -> dict[str, Path]:
    """Create an HTML scan dashboard and supporting charts."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Scan report requires at least one analyzed stock.")

    report_dir, chart_dir = Path(report_dir), Path(chart_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    is_nifty = sector_frame is not None and not sector_frame.empty
    kind = "nifty50" if is_nifty else "scan"

    archive_assets: dict[str, Path] = {}
    latest_assets: dict[str, Path] = {}

    chart_jobs = [
        (
            "relative",
            chart_dir / f"{kind}_relative_{stamp}.png",
            chart_dir / f"{kind}_relative_latest.png",
            lambda path: _create_bar_chart(
                frame, "Symbol", "Relative_Strength_20D_Pct",
                "Top stocks by 20-session relative strength vs NIFTY 50",
                path, limit=10,
            ),
        ),
        (
            "score",
            chart_dir / f"{kind}_score_distribution_{stamp}.png",
            chart_dir / f"{kind}_score_distribution_latest.png",
            lambda path: _create_score_chart(frame, path),
        ),
    ]
    if is_nifty:
        chart_jobs.append((
            "sector",
            chart_dir / f"nifty50_sector_strength_{stamp}.png",
            chart_dir / "nifty50_sector_strength_latest.png",
            lambda path: _create_sector_chart(sector_frame, path),
        ))

    for key, archive_path, latest_path, builder in chart_jobs:
        created = builder(archive_path)
        if created:
            latest_path.write_bytes(archive_path.read_bytes())
            archive_assets[key] = archive_path
            latest_assets[key] = latest_path

    work = frame.copy()
    for column in (
        "Relative_Strength_20D_Pct", "Return_20D_Pct", "Setup_Score"
    ):
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    if "Relative_Strength_20D_Pct" in work.columns:
        leaders = work.sort_values(
            "Relative_Strength_20D_Pct", ascending=False, na_position="last"
        )
    elif "Setup_Score" in work.columns:
        leaders = work.sort_values("Setup_Score", ascending=False, na_position="last")
    else:
        leaders = work

    count = len(work)
    above_sma20 = (
        int(work["Close_Above_SMA20"].map(_as_bool).sum())
        if "Close_Above_SMA20" in work.columns else 0
    )
    positive_20d = (
        int((work["Return_20D_Pct"] > 0).sum())
        if "Return_20D_Pct" in work.columns else 0
    )
    outperform = (
        int((work["Relative_Strength_20D_Pct"] > .5).sum())
        if "Relative_Strength_20D_Pct" in work.columns else 0
    )
    median_return = work["Return_20D_Pct"].median() if "Return_20D_Pct" in work.columns else float("nan")
    median_relative = work["Relative_Strength_20D_Pct"].median() if "Relative_Strength_20D_Pct" in work.columns else float("nan")

    archive_html = report_dir / f"{kind}_research_{stamp}.html"
    latest_html = report_dir / f"{kind}_research_latest.html"

    def image_panel(key: str, heading: str, report_path: Path) -> str:
        is_archive = report_path.name.endswith(f"{stamp}.html")
        asset = (archive_assets if is_archive else latest_assets).get(key)
        if not asset:
            return ""
        return (
            f'<section><h2>{escape(heading)}</h2>'
            f'<div class="chart-panel"><img class="chart" '
            f'src="{escape(_relative_asset(report_path, asset))}" '
            f'alt="{escape(heading)}"></div></section>'
        )

    def make_body(report_path: Path) -> str:
        cards = "".join([
            _metric_card("Stocks analyzed", str(count)),
            _metric_card(
                "Above SMA 20", f"{above_sma20}/{count}",
                f"{above_sma20 / count * 100:.1f}%" if count else "",
            ),
            _metric_card("Positive 20-session return", f"{positive_20d}/{count}"),
            _metric_card("Outperforming NIFTY 50", f"{outperform}/{count}"),
            _metric_card("Median 20-session return", _fmt(median_return, "%")),
            _metric_card("Median relative strength", _fmt(median_relative, " pp")),
        ])

        charts_html = image_panel(
            "relative", "Relative-strength leaders", report_path
        )
        charts_html += image_panel("score", "Score distribution", report_path)
        if is_nifty:
            charts_html += image_panel(
                "sector", "Industry-group strength", report_path
            )

        leader_columns = [
            "Symbol", "Company_Name", "NSE_Industry", "Close",
            "Return_20D_Pct", "Relative_Strength_20D_Pct", "Setup_Score",
            "Research_Conclusion",
        ]
        # Show the complete scanned universe (up to 100 rows), not just 12 leaders.
        leader_table = _html_table(leaders, leader_columns, max_rows=100)

        sector_columns = [
            "Sector_Rank_20D", "NSE_Industry", "Scanned_Constituents",
            "Coverage_Pct", "Sector_Return_20D_Pct",
            "Sector_Relative_Strength_20D_Pct", "Strength_Status",
        ]
        if is_nifty and "Sector_Relative_Strength_20D_Pct" in sector_frame.columns:
            sector_sorted = sector_frame.sort_values(
                "Sector_Relative_Strength_20D_Pct", ascending=False,
                na_position="last",
            )
        else:
            sector_sorted = pd.DataFrame()
        sector_table = _html_table(sector_sorted, sector_columns, max_rows=30)

        data_date = "Not available"
        if "Latest_Date" in work.columns and work["Latest_Date"].notna().any():
            data_date = _text(work.loc[work["Latest_Date"].notna(), "Latest_Date"].iloc[0])

        method_note = (
            "Industry-group returns are equal-weighted averages of successfully scanned "
            "NIFTY 50 constituents grouped by NSE industry classification. These are "
            "research proxies, not official NSE sector-index returns or market-cap-weighted indices."
            if is_nifty else
            "This is a user-selected stock scan. The list may not represent the full market "
            "or a complete sector sample."
        )
        small_sample_note = (
            '<p class="section-intro">Small watchlists are shown stock by stock; '
            'a one- or two-stock scan should not be interpreted as a market-wide distribution.</p>'
            if count < 6 else ""
        )
        sector_section = (
            f"<h2>Industry-group comparison</h2>{sector_table}"
            if is_nifty else ""
        )

        return f"""
<section class="metrics">{cards}</section>
<div class="grid2">{charts_html}</div>
<h2>Stock-by-stock research results</h2>
<p class="section-intro">Sorted by 20-session relative strength where available. The table includes all scanned stocks in this report.</p>
{small_sample_note}{leader_table}
{sector_section}
<h2>Method and limitations</h2>
<div class="note"><p class="note-title">How to interpret these comparisons</p><p>{escape(method_note)} Relative strength is calculated against the NIFTY 50 20-session return. Setup scores and evidence labels are screening context only; they are not expected returns, probabilities, or trade instructions. Data coverage and provider-reported corporate-action adjustments can affect historical comparisons.</p></div>
<p class="muted">Scan type: {escape(scan_label)} · Data date: {escape(data_date)} · Generated: {escape(stamp)}</p>
"""

    title = "NIFTY 50 Research Dashboard" if is_nifty else "Stock Scan Research Dashboard"
    subtitle = f"{scan_label} · {count} stocks analyzed · Generated {stamp}"
    archive_html.write_text(
        _base_html(title, subtitle, make_body(archive_html)), encoding="utf-8"
    )
    latest_html.write_text(
        _base_html(title, subtitle, make_body(latest_html)), encoding="utf-8"
    )

    return {
        "archive_html": archive_html,
        "latest_html": latest_html,
        **{f"chart_{key}": path for key, path in archive_assets.items()},
    }
