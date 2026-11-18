
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


# Database location
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "indian_stock_research.db"


@contextmanager
def get_connection():
    """Open, manage and safely close a database connection."""
    connection = sqlite3.connect(DB_PATH, timeout=30)

    try:
        connection.execute("PRAGMA foreign_keys = ON")
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def test_connection():
    """Check database connection and integrity."""
    try:
        with get_connection() as connection:
            result = connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()

            print("Database connection successful!")
            print("Database integrity:", result[0])

    except sqlite3.Error as error:
        print("Database error:", error)


def get_all_stocks():
    """Retrieve all stocks."""
    with get_connection() as connection:
        cursor = connection.execute("""
            SELECT stock_id, symbol, company_name,
                   exchange, industry
            FROM stocks
            ORDER BY symbol
        """)

        return cursor.fetchall()


def add_stock(symbol, company_name, exchange="NSE", industry=None):
    """Add a stock or update its existing information."""

    symbol = symbol.strip().upper()
    company_name = company_name.strip()
    exchange = exchange.strip().upper()

    if not symbol or not company_name:
        raise ValueError("Symbol and company name are required.")

    if not exchange:
        exchange = "NSE"

    with get_connection() as connection:
        connection.execute("""
            INSERT INTO stocks (
                symbol, company_name, exchange, industry
            )
            VALUES (?, ?, ?, ?)

            ON CONFLICT(symbol) DO UPDATE SET
                company_name = excluded.company_name,
                exchange = excluded.exchange,
                industry = excluded.industry
        """, (symbol, company_name, exchange, industry))

        result = connection.execute(
            "SELECT stock_id FROM stocks WHERE symbol = ?",
            (symbol,)
        ).fetchone()

        return result[0]


def get_stock_prices(symbol):
    """Retrieve historical prices for a stock."""

    symbol = symbol.strip().upper()

    with get_connection() as connection:
        cursor = connection.execute("""
            SELECT
                s.symbol,
                h.price_date,
                h.open_price,
                h.high_price,
                h.low_price,
                h.close_price,
                h.volume
            FROM stocks s
            JOIN historical_prices h
                ON s.stock_id = h.stock_id
            WHERE s.symbol = ?
            ORDER BY h.price_date DESC
        """, (symbol,))

        return cursor.fetchall()


def save_historical_price(
    symbol,
    price_date,
    open_price,
    high_price,
    low_price,
    close_price,
    volume=0
):
    """Insert or update one daily stock price record."""

    symbol = symbol.strip().upper()

    # Validate trading date
    try:
        datetime.strptime(price_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Date must use YYYY-MM-DD format.")

    # Convert prices to numbers
    try:
        open_price = float(open_price)
        high_price = float(high_price)
        low_price = float(low_price)
        close_price = float(close_price)

        volume_number = float(volume)
    except (ValueError, TypeError):
        raise ValueError("Prices and volume must be numeric.")

    if min(open_price, high_price, low_price, close_price) <= 0:
        raise ValueError("Stock prices must be positive.")

    if high_price < max(open_price, close_price, low_price):
        raise ValueError("High price is inconsistent with OHLC values.")

    if low_price > min(open_price, close_price, high_price):
        raise ValueError("Low price is inconsistent with OHLC values.")

    if not volume_number.is_integer() or volume_number < 0:
        raise ValueError("Volume must be a non-negative whole number.")

    volume = int(volume_number)

    with get_connection() as connection:

        stock = connection.execute(
            "SELECT stock_id FROM stocks WHERE symbol = ?",
            (symbol,)
        ).fetchone()

        if stock is None:
            raise ValueError(
                f"{symbol} does not exist. Add the stock first."
            )

        stock_id = stock[0]

        connection.execute("""
            INSERT INTO historical_prices (
                stock_id,
                price_date,
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(stock_id, price_date) DO UPDATE SET
                open_price = excluded.open_price,
                high_price = excluded.high_price,
                low_price = excluded.low_price,
                close_price = excluded.close_price,
                volume = excluded.volume
        """, (
            stock_id,
            price_date,
            open_price,
            high_price,
            low_price,
            close_price,
            volume
        ))

    return True




if __name__ == "__main__":

    test_connection()

    print("\nTesting duplicate prevention:")

    for attempt in range(2):
        save_historical_price(
            "TCS",
            "2026-11-11",
            2500,
            2530,
            2480,
            2515,
            1500000
        )

    with get_connection() as connection:
        count = connection.execute("""
            SELECT COUNT(*)
            FROM historical_prices h
            JOIN stocks s ON s.stock_id = h.stock_id
            WHERE s.symbol = ?
            AND h.price_date = ?
        """, ("TCS", "2026-11-11")).fetchone()[0]

    print("Records for TCS on 2026-11-11:", count)
