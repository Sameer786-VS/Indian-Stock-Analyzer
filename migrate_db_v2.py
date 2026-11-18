
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "indian_stock_research.db"
BACKUP_PATH = BASE_DIR / "indian_stock_research_pre_v2_backup.db"

REQUIRED_TABLES = {
    "stocks",
    "historical_prices",
    "benchmarks",
    "benchmark_prices",
    "research_results",
    "scan_results",
    "sector_data",
}


def get_tables(connection):
    return {
        row[0]
        for row in connection.execute("""
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        """)
    }


def get_record_counts(connection):
    return {
        table: connection.execute(
            f'SELECT COUNT(*) FROM "{table}"'
        ).fetchone()[0]
        for table in REQUIRED_TABLES
    }


def create_verified_backup():
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    if BACKUP_PATH.exists():
        raise FileExistsError(
            f"Backup destination already exists: {BACKUP_PATH}"
        )

    source = sqlite3.connect(DB_PATH, timeout=30)
    destination = sqlite3.connect(BACKUP_PATH)

    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()

    check = sqlite3.connect(
        f"file:{BACKUP_PATH.as_posix()}?mode=ro",
        uri=True
    )
    try:
        integrity = check.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"Backup integrity check failed: {integrity}")

        if not REQUIRED_TABLES.issubset(get_tables(check)):
            raise RuntimeError("Backup is missing required tables.")

    finally:
        check.close()

    print("Fresh backup created and verified:")
    print(BACKUP_PATH)


def migrate_database():
    connection = sqlite3.connect(
        DB_PATH,
        timeout=30,
        isolation_level=None
    )

    try:
        connection.execute("PRAGMA foreign_keys = ON")

        version = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]

        if version != 0:
            raise RuntimeError(
                f"Expected database version 0, found {version}. "
                "Migration stopped."
            )

        tables = get_tables(connection)

        if not REQUIRED_TABLES.issubset(tables):
            raise RuntimeError("One or more original database tables are missing.")

        if "schema_migrations" in tables or "execution_prices" in tables:
            raise RuntimeError(
                "Migration-related tables already exist. "
                "Review the database before proceeding."
            )

        connection.execute("BEGIN IMMEDIATE")

        try:
            # 1. Database migration history
            connection.execute("""
                CREATE TABLE schema_migrations (
                    version INTEGER PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 2. NSE and Yahoo symbol mapping
            connection.execute(
                "ALTER TABLE stocks ADD COLUMN yahoo_symbol TEXT"
            )

            connection.execute("""
                UPDATE stocks
                SET yahoo_symbol = 'TCS.NS'
                WHERE UPPER(symbol) = 'TCS'
            """)

            connection.execute("""
                CREATE UNIQUE INDEX idx_stocks_yahoo_symbol
                ON stocks(yahoo_symbol)
                WHERE yahoo_symbol IS NOT NULL
                  AND TRIM(yahoo_symbol) <> ''
            """)

            connection.execute(
                "ALTER TABLE benchmarks ADD COLUMN provider_symbol TEXT"
            )

            connection.execute("""
                UPDATE benchmarks
                SET provider_symbol = '^NSEI'
                WHERE benchmark_symbol = 'NIFTY50'
            """)

            # 3. Price provenance for analytical historical prices
            for column in (
                "price_basis TEXT",
                "data_source TEXT",
                "retrieved_at_utc TEXT",
            ):
                connection.execute(
                    f"ALTER TABLE historical_prices ADD COLUMN {column}"
                )

            # The existing TCS record is illustrative, not verified market data.
            connection.execute("""
                UPDATE historical_prices
                SET price_basis = 'illustrative_sample',
                    data_source = 'manual_sample'
                WHERE price_date = '2026-11-11'
                  AND stock_id IN (
                      SELECT stock_id FROM stocks
                      WHERE UPPER(symbol) = 'TCS'
                  )
            """)

            # 4. Benchmark price provenance
            for column in (
                "price_basis TEXT",
                "data_source TEXT",
                "retrieved_at_utc TEXT",
            ):
                connection.execute(
                    f"ALTER TABLE benchmark_prices ADD COLUMN {column}"
                )

            # 5. Separate execution-price records
            connection.execute("""
                CREATE TABLE execution_prices (
                    execution_price_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_id INTEGER NOT NULL,
                    price_date TEXT NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER DEFAULT 0,
                    price_basis TEXT NOT NULL
                        DEFAULT 'as_traded_execution',
                    data_source TEXT NOT NULL,
                    retrieved_at_utc TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
                    UNIQUE (stock_id, price_date)
                )
            """)

            # 6. Research, scan and sector provenance
            for table in (
                "research_results",
                "scan_results",
                "sector_data",
            ):
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN data_source TEXT"
                )
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN price_basis TEXT"
                )

            # 7. Indexes
            connection.execute("""
                CREATE INDEX idx_historical_price_basis_date
                ON historical_prices(price_basis, price_date)
            """)

            connection.execute("""
                CREATE INDEX idx_execution_stock_date
                ON execution_prices(stock_id, price_date)
            """)

            connection.execute("""
                CREATE INDEX idx_benchmark_source_date
                ON benchmark_prices(data_source, price_date)
            """)

            # 8. Record successful migration
            connection.execute("""
                INSERT INTO schema_migrations (version, description)
                VALUES (
                    2,
                    'Symbol mapping, price provenance and execution-price separation'
                )
            """)

            connection.execute("PRAGMA user_version = 2")
            connection.execute("COMMIT")

        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    finally:
        connection.close()


def verify_database():
    connection = sqlite3.connect(DB_PATH)

    try:
        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        version = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]

        print("\n--- MIGRATION VERIFICATION ---")
        print("Database integrity:", integrity)
        print("Database version:", version)
        print("Tables:", sorted(get_tables(connection)))

        print("\nStock symbol mapping:")
        print(connection.execute("""
            SELECT symbol, yahoo_symbol
            FROM stocks
        """).fetchall())

        print("\nTCS sample price provenance:")
        print(connection.execute("""
            SELECT price_basis, data_source
            FROM historical_prices
            WHERE price_date = '2026-11-11'
        """).fetchall())

        if integrity != "ok" or version != 2:
            raise RuntimeError("Final database verification failed.")

    finally:
        connection.close()


if __name__ == "__main__":
    try:
        create_verified_backup()
        migrate_database()
        verify_database()
        print("\nSCHEMA VERSION 2 MIGRATION COMPLETED SUCCESSFULLY.")

    except Exception as error:
        print("\nMIGRATION STOPPED:", error)
        print("Check the database and backup before retrying.")
        raise
