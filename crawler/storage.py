"""
Async SQLite persistence layer for the crawler + search system.

All DB access in the system goes through the :class:`Storage` class.
Uses ``aiosqlite`` with WAL journal mode so readers (the searcher) do
not block writers (the crawl manager / workers).

Schema per design.md §3:
  - crawl_jobs
  - pages
  - crawl_queue
  - word_frequencies
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Sequence

import aiosqlite


# Default DB path: <project_root>/data/crawler.db, computed relative to this file.
# crawler/storage.py -> project_root = parent of the 'crawler' package.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
DEFAULT_DB_PATH = str(_PROJECT_ROOT / "data" / "crawler.db")


# Terminal job statuses (completed_at is stamped when transitioning into any of these).
_TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}


class Storage:
    """
    Async SQLite wrapper. All DB access in the system goes through this class.

    Responsibilities:
      - Open/configure the aiosqlite connection (WAL, pragmas).
      - Schema migration on startup (CREATE TABLE IF NOT EXISTS ...).
      - Provide typed methods for crawl-manager writes and searcher reads.
      - Thread-safe (single event loop) transactional batches.

    Not responsible for:
      - HTTP, parsing, scoring, ranking.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Store the path; does not open the connection yet."""
        self.db_path: str = db_path or DEFAULT_DB_PATH
        self._db: Optional[aiosqlite.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Open the aiosqlite connection, set pragmas (WAL, synchronous=NORMAL,
        foreign_keys=ON), and run schema migrations. Idempotent.
        """
        if self._db is not None:
            return

        # Ensure the parent directory of the DB file exists.
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._db = await aiosqlite.connect(self.db_path)
        # Row factory so SELECTs return dict-like rows.
        self._db.row_factory = aiosqlite.Row

        # Pragmas per design §3.5.
        await self._db.execute("PRAGMA journal_mode = WAL;")
        await self._db.execute("PRAGMA synchronous = NORMAL;")
        await self._db.execute("PRAGMA temp_store = MEMORY;")
        await self._db.execute("PRAGMA cache_size = -20000;")  # ~20 MB
        await self._db.execute("PRAGMA foreign_keys = ON;")

        await self._migrate()
        await self._db.commit()

    async def close(self) -> None:
        """Close the connection if open. Idempotent."""
        if self._db is not None:
            try:
                await self._db.close()
            finally:
                self._db = None

    # Alias requested by the harness spec (``initialize()``) -- same as ``connect()``.
    async def initialize(self) -> None:
        """Alias for :meth:`connect`."""
        await self.connect()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def _migrate(self) -> None:
        """Create tables and indexes if they don't already exist."""
        assert self._db is not None

        # crawl_jobs
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS crawl_jobs (
                job_id          TEXT PRIMARY KEY,
                origin_url      TEXT NOT NULL,
                max_depth       INTEGER NOT NULL,
                same_host_only  INTEGER NOT NULL,
                status          TEXT NOT NULL,
                created_at      REAL NOT NULL,
                completed_at    REAL,
                pages_fetched   INTEGER NOT NULL DEFAULT 0,
                pages_failed    INTEGER NOT NULL DEFAULT 0,
                config_json     TEXT
            );
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_crawl_jobs_origin_depth "
            "ON crawl_jobs(origin_url, max_depth);"
        )

        # pages
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                page_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id       TEXT NOT NULL,
                url          TEXT NOT NULL,
                origin_url   TEXT NOT NULL,
                depth        INTEGER NOT NULL,
                status_code  INTEGER NOT NULL,
                content_type TEXT,
                title        TEXT,
                content      TEXT,
                fetched_at   REAL NOT NULL,
                UNIQUE(job_id, url),
                FOREIGN KEY(job_id) REFERENCES crawl_jobs(job_id)
            );
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_pages_url ON pages(url);"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_pages_job ON pages(job_id);"
        )

        # crawl_queue
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS crawl_queue (
                queue_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id       TEXT NOT NULL,
                url          TEXT NOT NULL,
                depth        INTEGER NOT NULL,
                enqueued_at  REAL NOT NULL,
                state        TEXT NOT NULL,
                UNIQUE(job_id, url),
                FOREIGN KEY(job_id) REFERENCES crawl_jobs(job_id)
            );
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_queue_job_state "
            "ON crawl_queue(job_id, state);"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_queue_job_depth "
            "ON crawl_queue(job_id, depth);"
        )

        # word_frequencies
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS word_frequencies (
                page_id    INTEGER NOT NULL,
                job_id     TEXT NOT NULL,
                word       TEXT NOT NULL,
                frequency  INTEGER NOT NULL,
                UNIQUE(page_id, word),
                FOREIGN KEY(page_id) REFERENCES pages(page_id),
                FOREIGN KEY(job_id)  REFERENCES crawl_jobs(job_id)
            );
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_wf_job_word "
            "ON word_frequencies(job_id, word);"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_wf_word ON word_frequencies(word);"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self) -> aiosqlite.Connection:
        """Return the live connection or raise if not connected."""
        if self._db is None:
            raise RuntimeError("Storage.connect() must be called before use.")
        return self._db

    # ------------------------------------------------------------------
    # crawl_jobs
    # ------------------------------------------------------------------

    async def create_job(
        self,
        job_id: str,
        origin_url: str,
        max_depth: int,
        same_host_only: bool,
        config_json: str,
    ) -> None:
        """Insert a new crawl job row with status='pending'."""
        db = self._conn()
        await db.execute(
            """
            INSERT OR IGNORE INTO crawl_jobs
                (job_id, origin_url, max_depth, same_host_only, status,
                 created_at, completed_at, pages_fetched, pages_failed,
                 config_json)
            VALUES (?, ?, ?, ?, 'pending', ?, NULL, 0, 0, ?);
            """,
            (
                job_id,
                origin_url,
                int(max_depth),
                1 if same_host_only else 0,
                time.time(),
                config_json,
            ),
        )
        await db.commit()

    async def set_job_status(self, job_id: str, status: str) -> None:
        """Update status in-place. Sets completed_at when status is terminal."""
        db = self._conn()
        if status in _TERMINAL_JOB_STATUSES:
            await db.execute(
                "UPDATE crawl_jobs SET status=?, completed_at=? WHERE job_id=?;",
                (status, time.time(), job_id),
            )
        else:
            await db.execute(
                "UPDATE crawl_jobs SET status=? WHERE job_id=?;",
                (status, job_id),
            )
        await db.commit()

    async def get_job(self, job_id: str) -> Optional[dict]:
        """Return the job row as a dict, or None."""
        db = self._conn()
        async with db.execute(
            "SELECT * FROM crawl_jobs WHERE job_id=?;", (job_id,)
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def list_jobs(self, status: Optional[str] = None) -> list[dict]:
        """List all jobs, optionally filtered by status."""
        db = self._conn()
        if status is None:
            sql = "SELECT * FROM crawl_jobs ORDER BY created_at DESC;"
            params: tuple = ()
        else:
            sql = (
                "SELECT * FROM crawl_jobs WHERE status=? "
                "ORDER BY created_at DESC;"
            )
            params = (status,)
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def find_resumable_jobs(self) -> list[dict]:
        """
        Return all jobs with status in ('pending', 'running') -
        the set that resume_jobs() must reattach to.
        """
        db = self._conn()
        async with db.execute(
            "SELECT * FROM crawl_jobs WHERE status IN ('pending','running') "
            "ORDER BY created_at ASC;"
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def increment_job_counter(
        self, job_id: str, fetched: int = 0, failed: int = 0
    ) -> None:
        """Atomic counter bump for pages_fetched / pages_failed."""
        db = self._conn()
        await db.execute(
            """
            UPDATE crawl_jobs
               SET pages_fetched = pages_fetched + ?,
                   pages_failed  = pages_failed  + ?
             WHERE job_id = ?;
            """,
            (int(fetched), int(failed), job_id),
        )
        await db.commit()

    # ------------------------------------------------------------------
    # pages
    # ------------------------------------------------------------------

    async def save_page(
        self,
        job_id: str,
        url: str,
        origin_url: str,
        depth: int,
        status_code: int,
        content_type: Optional[str],
        title: Optional[str],
        content: Optional[str],
        word_frequencies: dict[str, int],
    ) -> None:
        """
        Atomic transaction: insert into pages, insert/update word_frequencies,
        increment crawl_jobs.pages_fetched, mark the corresponding
        crawl_queue row state='done'. On UNIQUE conflict in pages
        (already indexed), this is a no-op to support idempotent retries.
        """
        db = self._conn()

        # Single transaction for the whole batch.
        await db.execute("BEGIN;")
        try:
            cur = await db.execute(
                """
                INSERT OR IGNORE INTO pages
                    (job_id, url, origin_url, depth, status_code,
                     content_type, title, content, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    job_id,
                    url,
                    origin_url,
                    int(depth),
                    int(status_code),
                    content_type,
                    title,
                    content,
                    time.time(),
                ),
            )

            if cur.rowcount and cur.rowcount > 0:
                # The page row was freshly inserted; look up its id.
                async with db.execute(
                    "SELECT page_id FROM pages WHERE job_id=? AND url=?;",
                    (job_id, url),
                ) as pcur:
                    prow = await pcur.fetchone()
                page_id = int(prow["page_id"]) if prow is not None else None

                if page_id is not None and word_frequencies:
                    wf_rows = [
                        (page_id, job_id, word, int(freq))
                        for word, freq in word_frequencies.items()
                        if word  # skip empty-string tokens defensively
                    ]
                    if wf_rows:
                        await db.executemany(
                            """
                            INSERT OR REPLACE INTO word_frequencies
                                (page_id, job_id, word, frequency)
                            VALUES (?, ?, ?, ?);
                            """,
                            wf_rows,
                        )

                # Mark the queue row done (if it existed).
                await db.execute(
                    """
                    UPDATE crawl_queue
                       SET state='done'
                     WHERE job_id=? AND url=?;
                    """,
                    (job_id, url),
                )

                # Bump the pages_fetched counter.
                await db.execute(
                    "UPDATE crawl_jobs "
                    "SET pages_fetched = pages_fetched + 1 WHERE job_id=?;",
                    (job_id,),
                )
            # If the page already existed (UNIQUE conflict), no-op; retry idempotent.

            await db.commit()
        except Exception:
            await db.rollback()
            raise

    async def record_failure(
        self, job_id: str, url: str, depth: int, status_code: int
    ) -> None:
        """
        Atomic transaction: insert a pages row with content=NULL to record
        that we attempted and failed (so we never retry within this run),
        mark the queue row state='failed', and increment pages_failed.
        """
        db = self._conn()
        await db.execute("BEGIN;")
        try:
            await db.execute(
                """
                INSERT OR IGNORE INTO pages
                    (job_id, url, origin_url, depth, status_code,
                     content_type, title, content, fetched_at)
                VALUES (?, ?, '', ?, ?, NULL, NULL, NULL, ?);
                """,
                (job_id, url, int(depth), int(status_code), time.time()),
            )
            await db.execute(
                """
                UPDATE crawl_queue
                   SET state='failed'
                 WHERE job_id=? AND url=?;
                """,
                (job_id, url),
            )
            await db.execute(
                "UPDATE crawl_jobs "
                "SET pages_failed = pages_failed + 1 WHERE job_id=?;",
                (job_id,),
            )
            await db.commit()
        except Exception:
            await db.rollback()
            raise

    async def get_seen_urls(self, job_id: str) -> set[str]:
        """
        Return the union of URLs already in pages or crawl_queue for this
        job. Used by resume_jobs() to rebuild the in-memory dedup set.
        """
        db = self._conn()
        seen: set[str] = set()
        async with db.execute(
            "SELECT url FROM pages WHERE job_id=? "
            "UNION SELECT url FROM crawl_queue WHERE job_id=?;",
            (job_id, job_id),
        ) as cur:
            async for row in cur:
                seen.add(row["url"])
        return seen

    async def count_pages(self, job_id: str) -> int:
        """Return count of pages rows for a job (alias of total_pages)."""
        return await self.total_pages(job_id)

    async def search_pages(
        self, query: str, job_id: Optional[str] = None, limit: int = 500
    ) -> list[dict]:
        """
        LIKE-based candidate pre-filter over the pages table.

        Splits ``query`` on whitespace into terms; each term must match
        ``title`` OR ``content`` (case-insensitive LIKE), AND across terms.
        Fully parameterized -- user input is never string-concatenated
        into SQL.
        """
        db = self._conn()
        terms = [t for t in query.strip().split() if t]
        if not terms:
            return []

        where_clauses: list[str] = []
        params: list = []
        if job_id is not None:
            where_clauses.append("job_id = ?")
            params.append(job_id)

        # Each term: (title LIKE ? OR content LIKE ?)
        for term in terms:
            like = f"%{term}%"
            where_clauses.append("(title LIKE ? OR content LIKE ?)")
            params.append(like)
            params.append(like)

        sql = (
            "SELECT page_id, job_id, url, origin_url, depth, status_code, "
            "       content_type, title, content, fetched_at "
            "  FROM pages "
            " WHERE " + " AND ".join(where_clauses) +
            " LIMIT ?;"
        )
        params.append(int(limit))

        async with db.execute(sql, tuple(params)) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # crawl_queue
    # ------------------------------------------------------------------

    async def enqueue_urls(
        self, job_id: str, items: Sequence[tuple[str, int]]
    ) -> int:
        """
        Bulk-insert (url, depth) pairs with state='pending'. Uses
        INSERT OR IGNORE to skip duplicates (UNIQUE(job_id, url)).
        Returns the count of newly inserted rows.
        """
        if not items:
            return 0
        db = self._conn()
        now = time.time()
        rows = [
            (job_id, url, int(depth), now, "pending")
            for (url, depth) in items
        ]
        cur = await db.executemany(
            """
            INSERT OR IGNORE INTO crawl_queue
                (job_id, url, depth, enqueued_at, state)
            VALUES (?, ?, ?, ?, ?);
            """,
            rows,
        )
        await db.commit()
        return int(cur.rowcount or 0)

    async def claim_pending(self, job_id: str, limit: int) -> list[dict]:
        """
        Transactionally select up to ``limit`` rows where state='pending'
        for this job, flip them to state='in_flight', return them.
        Ordered by depth ASC, queue_id ASC (BFS).
        """
        db = self._conn()
        await db.execute("BEGIN IMMEDIATE;")
        try:
            async with db.execute(
                """
                SELECT queue_id, job_id, url, depth, enqueued_at, state
                  FROM crawl_queue
                 WHERE job_id=? AND state='pending'
                 ORDER BY depth ASC, queue_id ASC
                 LIMIT ?;
                """,
                (job_id, int(limit)),
            ) as cur:
                rows = await cur.fetchall()

            claimed: list[dict] = [dict(r) for r in rows]
            if claimed:
                ids = [r["queue_id"] for r in claimed]
                placeholders = ",".join("?" for _ in ids)
                await db.execute(
                    f"UPDATE crawl_queue SET state='in_flight' "
                    f"WHERE queue_id IN ({placeholders});",
                    tuple(ids),
                )
                # Reflect the new state on the returned dicts.
                for d in claimed:
                    d["state"] = "in_flight"
            await db.commit()
            return claimed
        except Exception:
            await db.rollback()
            raise

    async def get_pending_queue_items(
        self, job_id: str, limit: int
    ) -> list[dict]:
        """
        Read-only peek at pending queue items (does NOT claim them).
        Kept as a separate method from :meth:`claim_pending` per the
        architect note -- callers that want atomic claiming should use
        :meth:`claim_pending`.
        """
        db = self._conn()
        async with db.execute(
            """
            SELECT queue_id, job_id, url, depth, enqueued_at, state
              FROM crawl_queue
             WHERE job_id=? AND state='pending'
             ORDER BY depth ASC, queue_id ASC
             LIMIT ?;
            """,
            (job_id, int(limit)),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def claim_queue_items(self, job_id: str, limit: int) -> list[dict]:
        """Alias for :meth:`claim_pending` to match the contract naming."""
        return await self.claim_pending(job_id, limit)

    async def reset_in_flight(self, job_id: str) -> int:
        """
        Reset state='in_flight' -> 'pending' for all rows of this job.
        Called at startup for resumed jobs. Returns number of rows reset.
        """
        db = self._conn()
        cur = await db.execute(
            "UPDATE crawl_queue SET state='pending' "
            "WHERE job_id=? AND state='in_flight';",
            (job_id,),
        )
        await db.commit()
        return int(cur.rowcount or 0)

    # Alias matching the task prompt wording.
    async def reset_stale_in_flight(self, job_id: str) -> int:
        """Alias of :meth:`reset_in_flight`."""
        return await self.reset_in_flight(job_id)

    async def mark_queue_state(
        self, job_id: str, url: str, state: str
    ) -> None:
        """
        Transition a queue row to a new state. Legal states:
        ``pending``, ``in_flight``, ``done``, ``failed``, ``blocked``,
        ``skipped``.
        """
        db = self._conn()
        await db.execute(
            "UPDATE crawl_queue SET state=? WHERE job_id=? AND url=?;",
            (state, job_id, url),
        )
        await db.commit()

    async def queue_depth(self, job_id: str) -> int:
        """Count rows with state='pending' for this job."""
        db = self._conn()
        async with db.execute(
            "SELECT COUNT(*) AS c FROM crawl_queue "
            "WHERE job_id=? AND state='pending';",
            (job_id,),
        ) as cur:
            row = await cur.fetchone()
        return int(row["c"]) if row is not None else 0

    # ------------------------------------------------------------------
    # word_frequencies
    # ------------------------------------------------------------------

    async def insert_word_frequencies(
        self,
        tuples: Sequence[tuple[str, str, str, int, int, str]],
    ) -> int:
        """
        Bulk insert word frequency rows.

        Each tuple is ``(word, url, origin, depth, frequency, job_id)`` as
        specified in the task prompt. ``url`` is resolved to ``page_id``
        via the pages table; rows whose page does not exist are skipped.

        Uses INSERT OR REPLACE so repeated calls are idempotent.
        Returns the count of rows actually written.
        """
        if not tuples:
            return 0
        db = self._conn()

        # Group URLs per job for a single lookup batch.
        # (job_id, url) -> page_id
        needed: dict[tuple[str, str], Optional[int]] = {}
        for (_word, url, _origin, _depth, _freq, job_id) in tuples:
            needed[(job_id, url)] = None

        # Resolve page_ids.
        for (job_id, url) in list(needed.keys()):
            async with db.execute(
                "SELECT page_id FROM pages WHERE job_id=? AND url=?;",
                (job_id, url),
            ) as cur:
                row = await cur.fetchone()
            needed[(job_id, url)] = int(row["page_id"]) if row else None

        rows_to_write: list[tuple[int, str, str, int]] = []
        for (word, url, _origin, _depth, freq, job_id) in tuples:
            page_id = needed.get((job_id, url))
            if page_id is None or not word:
                continue
            rows_to_write.append((page_id, job_id, word, int(freq)))

        if not rows_to_write:
            return 0

        await db.executemany(
            """
            INSERT OR REPLACE INTO word_frequencies
                (page_id, job_id, word, frequency)
            VALUES (?, ?, ?, ?);
            """,
            rows_to_write,
        )
        await db.commit()
        return len(rows_to_write)

    async def search_by_word(
        self, word: str, limit: int = 50, job_id: Optional[str] = None
    ) -> list[dict]:
        """
        Return rows from ``word_frequencies`` for the given word, ordered
        by frequency DESC. Joined with ``pages`` so callers get
        ``(url, origin_url, depth, frequency)``.
        """
        db = self._conn()
        params: list = [word]
        sql = (
            "SELECT wf.page_id   AS page_id, "
            "       wf.job_id    AS job_id, "
            "       wf.word      AS word, "
            "       wf.frequency AS frequency, "
            "       p.url        AS url, "
            "       p.origin_url AS origin_url, "
            "       p.depth      AS depth, "
            "       p.title      AS title "
            "  FROM word_frequencies wf "
            "  JOIN pages p ON p.page_id = wf.page_id "
            " WHERE wf.word = ? "
        )
        if job_id is not None:
            sql += "   AND wf.job_id = ? "
            params.append(job_id)
        sql += " ORDER BY wf.frequency DESC LIMIT ?;"
        params.append(int(limit))

        async with db.execute(sql, tuple(params)) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Searcher reads
    # ------------------------------------------------------------------

    async def lookup_word(
        self, job_id: str, word: str
    ) -> list[tuple[int, int]]:
        """
        Return list of (page_id, frequency) for rows in word_frequencies
        matching this job_id and word.
        """
        db = self._conn()
        async with db.execute(
            "SELECT page_id, frequency FROM word_frequencies "
            "WHERE job_id=? AND word=?;",
            (job_id, word),
        ) as cur:
            rows = await cur.fetchall()
        return [(int(r["page_id"]), int(r["frequency"])) for r in rows]

    async def fetch_pages(
        self, job_id: str, page_ids: Sequence[int]
    ) -> list[dict]:
        """
        Return page rows (page_id, url, origin_url, depth, title, content,
        fetched_at) for the given ids. Batched to avoid huge IN clauses.
        """
        if not page_ids:
            return []
        db = self._conn()

        out: list[dict] = []
        # SQLite default SQLITE_MAX_VARIABLE_NUMBER is 999 (older) /
        # 32766 (newer). Stay conservative with 500-id batches.
        BATCH = 500
        ids = list(page_ids)
        for i in range(0, len(ids), BATCH):
            chunk = ids[i : i + BATCH]
            placeholders = ",".join("?" for _ in chunk)
            sql = (
                "SELECT page_id, url, origin_url, depth, title, content, "
                "       fetched_at "
                "  FROM pages "
                " WHERE job_id=? AND page_id IN (" + placeholders + ");"
            )
            params = (job_id, *chunk)
            async with db.execute(sql, params) as cur:
                rows = await cur.fetchall()
            out.extend(dict(r) for r in rows)
        return out

    async def total_pages(self, job_id: str) -> int:
        """Count indexed pages for a job (for IDF denominator)."""
        db = self._conn()
        async with db.execute(
            "SELECT COUNT(*) AS c FROM pages WHERE job_id=?;",
            (job_id,),
        ) as cur:
            row = await cur.fetchone()
        return int(row["c"]) if row is not None else 0

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    async def export_to_pdata(self, filepath: str) -> int:
        """
        Flat-file dump of ``word_frequencies``.

        Each line is tab-separated:
            word<TAB>url<TAB>origin_url<TAB>depth<TAB>frequency<TAB>job_id

        Returns the number of rows written.
        """
        db = self._conn()

        # Ensure parent dir exists.
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)

        count = 0
        async with db.execute(
            """
            SELECT wf.word       AS word,
                   p.url         AS url,
                   p.origin_url  AS origin_url,
                   p.depth       AS depth,
                   wf.frequency  AS frequency,
                   wf.job_id     AS job_id
              FROM word_frequencies wf
              JOIN pages p ON p.page_id = wf.page_id
            ORDER BY wf.job_id, wf.word, wf.frequency DESC;
            """
        ) as cur:
            with open(filepath, "w", encoding="utf-8") as fh:
                async for row in cur:
                    fh.write(
                        f"{row['word']}\t{row['url']}\t{row['origin_url']}\t"
                        f"{row['depth']}\t{row['frequency']}\t{row['job_id']}\n"
                    )
                    count += 1
        return count
