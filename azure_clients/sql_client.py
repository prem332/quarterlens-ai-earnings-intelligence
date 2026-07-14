from __future__ import annotations

import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import pyodbc
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_DRIVER = "ODBC Driver 18 for SQL Server"

# Transient SQLSTATE / Azure error codes worth retrying while a serverless DB
# resumes from auto-pause (40613 = database currently unavailable/resuming).
_TRANSIENT_CODES = frozenset(
    {"40613", "40197", "40501", "49918", "49919", "49920",
     "08001", "08S01", "10928", "10929"}
)

_INSERT_COLUMNS = (
    "ticker", "cik", "accession", "form", "fiscal_label",
    "concept", "xbrl_tag", "value", "unit",
    "period_start", "period_end", "fy", "fp",
)


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val


class SQLClient:
    def __init__(self, *, connect_retries: int = 5, retry_backoff: float = 3.0):
        self._server = _require("AZURE_SQL_SERVER")        # FQDN, e.g. quarterlens-sqlserver.database.windows.net
        self._database = _require("AZURE_SQL_DATABASE")
        self._username = _require("AZURE_SQL_USERNAME")
        self._password = _require("AZURE_SQL_PASSWORD")
        self._connect_retries = connect_retries
        self._retry_backoff = retry_backoff

    @property
    def _conn_str(self) -> str:
        return (
            f"DRIVER={{{_DRIVER}}};"
            f"SERVER=tcp:{self._server},1433;"
            f"DATABASE={self._database};"
            f"UID={self._username};"
            f"PWD={self._password};"
            "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=60;"
        )

    def _connect(self) -> pyodbc.Connection:
        last: Exception | None = None
        for attempt in range(1, self._connect_retries + 1):
            try:
                return pyodbc.connect(self._conn_str)
            except pyodbc.Error as exc:
                code = str(exc.args[0]) if exc.args else ""
                if code not in _TRANSIENT_CODES or attempt == self._connect_retries:
                    raise
                wait = self._retry_backoff * attempt
                logger.warning(
                    "Transient SQL connect error %s (attempt %d/%d); "
                    "serverless DB likely resuming, retrying in %.0fs",
                    code, attempt, self._connect_retries, wait,
                )
                last = exc
                time.sleep(wait)
        assert last is not None  # unreachable; loop either returns or raises
        raise last

    @contextmanager
    def connection(self):
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def apply_schema(self, schema_path: str | Path) -> None:
        """Run a GO-batched .sql file. pyodbc can't parse the GO directive, so
        split on it and execute each batch separately."""
        ddl = Path(schema_path).read_text(encoding="utf-8")
        batches = [b.strip() for b in re.split(r"(?im)^\s*GO\s*$", ddl) if b.strip()]
        with self.connection() as conn:
            cur = conn.cursor()
            for batch in batches:
                cur.execute(batch)
            conn.commit()
        logger.info("Applied schema from %s (%d batch(es))", schema_path, len(batches))

    def load_facts(self, rows: Sequence[dict[str, Any]]) -> int:
        """Idempotent load: delete existing rows for the affected accessions,
        then bulk-insert. One transaction. The UNIQUE natural key will reject a
        duplicate (accession, concept, period_start, period_end) within `rows` —
        that's intentional: the fetcher must emit exactly one duration per fact
        (quarterly for 10-Q, annual for 10-K), so a collision means a bug, not
        data to silently drop.

        `value` should be a Decimal (fast_executemany infers column types from
        the first row; float can lose precision on DECIMAL(28,4)).
        """
        rows = list(rows)
        if not rows:
            return 0

        accessions = sorted({r["accession"] for r in rows})
        del_placeholders = ",".join("?" * len(accessions))
        insert_sql = (
            f"INSERT INTO dbo.financial_facts ({', '.join(_INSERT_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(_INSERT_COLUMNS))})"
        )
        values = [tuple(r.get(c) for c in _INSERT_COLUMNS) for r in rows]

        with self.connection() as conn:
            cur = conn.cursor()
            cur.fast_executemany = True
            try:
                cur.execute(
                    f"DELETE FROM dbo.financial_facts WHERE accession IN ({del_placeholders})",
                    *accessions,
                )
                cur.executemany(insert_sql, values)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        logger.info("Loaded %d facts across %d filing(s)", len(values), len(accessions))
        return len(values)

    def fetch_facts(
        self, cik: str, fiscal_label: str, concept: str | None = None
    ) -> list[dict[str, Any]]:
        """Read path used by the fetcher's post-run coverage assertion (and a
        convenience for later numeric-validation reads). Hits IX_..._lookup."""
        sql = (
            "SELECT concept, xbrl_tag, value, unit, period_start, period_end "
            "FROM dbo.financial_facts WHERE cik = ? AND fiscal_label = ?"
        )
        params: list[Any] = [cik, fiscal_label]
        if concept:
            sql += " AND concept = ?"
            params.append(concept)
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, *params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def count(self) -> int:
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM dbo.financial_facts")
            return int(cur.fetchone()[0])