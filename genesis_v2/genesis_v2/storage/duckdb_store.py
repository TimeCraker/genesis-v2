"""DuckDB telemetry store for Genesis v2."""

from __future__ import annotations

import threading
from pathlib import Path

import duckdb

from genesis_v2.agent.agent import Agent


class DuckDBStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._conn = duckdb.connect(str(self._path))
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generations (
                generation INTEGER,
                island_id INTEGER,
                alive_count INTEGER,
                mean_fitness DOUBLE,
                mean_energy DOUBLE,
                best_fitness DOUBLE,
                PRIMARY KEY (generation, island_id)
            );
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ticks (
                tick_id    BIGINT,
                agent_id   VARCHAR,
                island_id  INTEGER,
                energy     DOUBLE,
                fitness    DOUBLE,
                pred_err   DOUBLE,
                compression DOUBLE,
                bvar       DOUBLE,
                node_count INTEGER,
                edge_count INTEGER,
                messages_received INTEGER DEFAULT 0,
                mean_trust DOUBLE DEFAULT 0.0
            );
            """
        )
        # Migrate: add social columns if missing (for existing DBs)
        for col, dtype in [("messages_received", "INTEGER"), ("mean_trust", "DOUBLE")]:
            try:
                self._conn.execute(f"ALTER TABLE ticks ADD COLUMN {col} {dtype} DEFAULT 0;")
            except Exception:
                pass  # column already exists

    def record_tick(
        self,
        tick_id: int,
        agent: Agent,
        messages_received: int = 0,
        mean_trust: float = 0.0,
    ) -> None:
        g = agent.genome
        assert g is not None
        with self._lock:
            self._conn.execute(
                "INSERT INTO ticks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                [
                    int(tick_id),
                    str(agent.id),
                    int(agent.island_id),
                    float(agent.energy),
                    float(agent.fitness),
                    float(agent.prediction_error),
                    float(agent.compression),
                    float(agent.behavioral_variance),
                    int(g.node_count()),
                    int(g.edge_count()),
                    int(messages_received),
                    float(mean_trust),
                ],
            )

    def record_generation(
        self,
        generation: int,
        island_id: int,
        alive_count: int,
        mean_fitness: float,
        mean_energy: float,
        best_fitness: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO generations VALUES (?, ?, ?, ?, ?, ?);",
                [generation, island_id, alive_count, mean_fitness, mean_energy, best_fitness],
            )

    def flush(self) -> None:
        pass  # DuckDB auto-commits

    def tick_count(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM ticks;")
            return int(cur.fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
