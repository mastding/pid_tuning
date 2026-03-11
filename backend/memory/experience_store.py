from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


MEMORY_ROOT = Path(__file__).resolve().parent / "records"
EXPERIENCE_FILE = MEMORY_ROOT / "pid_experiences.jsonl"
INDEX_FILE = MEMORY_ROOT / "pid_experiences.db"


def ensure_experience_store() -> Path:
    MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXPERIENCE_FILE.exists():
        EXPERIENCE_FILE.touch()
    _ensure_index_schema()
    return EXPERIENCE_FILE


def _get_connection() -> sqlite3.Connection:
    ensure_experience_store()
    conn = sqlite3.connect(INDEX_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_index_schema() -> None:
    MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(INDEX_FILE) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiences (
                experience_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                loop_name TEXT,
                loop_type TEXT,
                model_type TEXT,
                loop_uri TEXT,
                data_source TEXT,
                model_k REAL,
                model_t REAL,
                model_l REAL,
                normalized_rmse REAL,
                r2_score REAL,
                final_strategy TEXT,
                final_rating REAL,
                performance_score REAL,
                passed INTEGER,
                tags_json TEXT,
                record_json TEXT NOT NULL
            )
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(experiences)").fetchall()}
        if "model_type" not in columns:
            conn.execute("ALTER TABLE experiences ADD COLUMN model_type TEXT")
        if "hit_count" not in columns:
            conn.execute("ALTER TABLE experiences ADD COLUMN hit_count INTEGER DEFAULT 0")
        if "follow_up_success_count" not in columns:
            conn.execute("ALTER TABLE experiences ADD COLUMN follow_up_success_count INTEGER DEFAULT 0")
        if "last_hit_at" not in columns:
            conn.execute("ALTER TABLE experiences ADD COLUMN last_hit_at TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_loop_type ON experiences(loop_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_model_type ON experiences(model_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_created_at ON experiences(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_final_rating ON experiences(final_rating DESC)")
        conn.commit()


def append_experience_record(record: Dict[str, Any]) -> str:
    store_path = ensure_experience_store()
    experience_id = str(record.get("experience_id") or "")
    with store_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    model = record.get("model") or {}
    strategy = record.get("strategy") or {}
    evaluation = record.get("evaluation") or {}
    tags = record.get("tags") or []
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO experiences (
                experience_id, created_at, loop_name, loop_type, model_type, loop_uri, data_source,
                model_k, model_t, model_l, normalized_rmse, r2_score,
                final_strategy, final_rating, performance_score, passed, tags_json, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experience_id,
                str(record.get("created_at", "")),
                str(record.get("loop_name", "")),
                str(record.get("loop_type", "")),
                str(model.get("model_type", "FOPDT")),
                str(record.get("loop_uri", "")),
                str(record.get("data_source", "")),
                float(model.get("K", 0.0) or 0.0),
                float(model.get("T", 0.0) or 0.0),
                float(model.get("L", 0.0) or 0.0),
                float(model.get("normalized_rmse", 0.0) or 0.0),
                float(model.get("r2_score", 0.0) or 0.0),
                str(strategy.get("final", "")),
                float(evaluation.get("final_rating", 0.0) or 0.0),
                float(evaluation.get("performance_score", 0.0) or 0.0),
                1 if bool(evaluation.get("passed", False)) else 0,
                json.dumps(tags, ensure_ascii=False),
                json.dumps(record, ensure_ascii=False),
            ),
        )
        conn.commit()
    return experience_id


def rebuild_experience_index() -> Dict[str, Any]:
    ensure_experience_store()
    records = load_experience_records()
    with sqlite3.connect(INDEX_FILE) as conn:
        conn.execute("DELETE FROM experiences")
        conn.commit()
    rebuilt_count = 0
    for record in records:
        try:
            append_experience_record(record)
            rebuilt_count += 1
        except Exception:
            continue
    return {"rebuilt": True, "rebuilt_count": rebuilt_count}


def clear_experience_store() -> Dict[str, Any]:
    ensure_experience_store()
    before = get_experience_stats()

    if EXPERIENCE_FILE.exists():
        EXPERIENCE_FILE.write_text("", encoding="utf-8")

    with sqlite3.connect(INDEX_FILE) as conn:
        conn.execute("DELETE FROM experiences")
        conn.commit()

    after = get_experience_stats()
    return {
        "cleared": True,
        "cleared_count": int(before.get("total_count", 0) or 0),
        "before": before,
        "after": after,
    }


def load_experience_records(limit: int | None = None) -> List[Dict[str, Any]]:
    store_path = ensure_experience_store()
    records: List[Dict[str, Any]] = []
    with store_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None and limit > 0:
        return records[-limit:]
    return records


def load_indexed_experience_candidates(loop_type: str, limit: int = 100) -> List[Dict[str, Any]]:
    ensure_experience_store()
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT record_json
            FROM experiences
            WHERE (? = '' OR loop_type = ?)
            ORDER BY passed DESC, final_rating DESC, created_at DESC
            LIMIT ?
            """,
            (loop_type, loop_type, max(limit, 1)),
        ).fetchall()
    records: List[Dict[str, Any]] = []
    for row in rows:
        try:
            records.append(json.loads(row["record_json"]))
        except Exception:
            continue
    return records


def get_experience_stats() -> Dict[str, Any]:
    ensure_experience_store()
    with _get_connection() as conn:
        summary = conn.execute(
            """
            SELECT
                COUNT(*) AS total_count,
                SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) AS passed_count,
                AVG(final_rating) AS avg_final_rating,
                SUM(COALESCE(hit_count, 0)) AS total_hits,
                SUM(COALESCE(follow_up_success_count, 0)) AS total_follow_up_success
            FROM experiences
            """
        ).fetchone()
        strategy_rows = conn.execute(
            """
            SELECT final_strategy, COUNT(*) AS cnt
            FROM experiences
            GROUP BY final_strategy
            ORDER BY cnt DESC, final_strategy ASC
            LIMIT 5
            """
        ).fetchall()
        loop_rows = conn.execute(
            """
            SELECT loop_type, COUNT(*) AS cnt
            FROM experiences
            GROUP BY loop_type
            ORDER BY cnt DESC, loop_type ASC
            """
        ).fetchall()
        model_rows = conn.execute(
            """
            SELECT model_type, COUNT(*) AS cnt
            FROM experiences
            GROUP BY model_type
            ORDER BY cnt DESC, model_type ASC
            """
        ).fetchall()
    return {
        "total_count": int(summary["total_count"] or 0),
        "passed_count": int(summary["passed_count"] or 0),
        "avg_final_rating": float(summary["avg_final_rating"] or 0.0),
        "total_hits": int(summary["total_hits"] or 0),
        "total_follow_up_success": int(summary["total_follow_up_success"] or 0),
        "top_strategies": [
            {"strategy": str(row["final_strategy"] or "-"), "count": int(row["cnt"] or 0)}
            for row in strategy_rows
        ],
        "loop_type_distribution": [
            {"loop_type": str(row["loop_type"] or "-"), "count": int(row["cnt"] or 0)}
            for row in loop_rows
        ],
        "model_type_distribution": [
            {"model_type": str(row["model_type"] or "FOPDT"), "count": int(row["cnt"] or 0)}
            for row in model_rows
        ],
    }


def list_experiences(
    *,
    loop_type: str = "",
    model_type: str = "",
    passed: str = "",
    strategy: str = "",
    keyword: str = "",
    limit: int = 50,
) -> List[Dict[str, Any]]:
    ensure_experience_store()
    conditions: list[str] = []
    params: list[Any] = []
    if loop_type:
        conditions.append("loop_type = ?")
        params.append(loop_type)
    if model_type:
        conditions.append("model_type = ?")
        params.append(model_type)
    if passed in {"true", "false"}:
        conditions.append("passed = ?")
        params.append(1 if passed == "true" else 0)
    if strategy:
        conditions.append("final_strategy = ?")
        params.append(strategy)
    if keyword:
        conditions.append("(loop_name LIKE ? OR loop_uri LIKE ? OR experience_id LIKE ?)")
        like = f"%{keyword}%"
        params.extend([like, like, like])
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"""
        SELECT experience_id, created_at, loop_name, loop_type, model_type, loop_uri, final_strategy,
               model_k, model_t, model_l, normalized_rmse, r2_score,
               hit_count, follow_up_success_count, last_hit_at,
               final_rating, performance_score, passed, tags_json
        FROM experiences
        {where_clause}
        ORDER BY created_at DESC
        LIMIT ?
    """
    params.append(max(limit, 1))
    with _get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        {
            "experience_id": str(row["experience_id"]),
            "created_at": str(row["created_at"]),
            "loop_name": str(row["loop_name"] or ""),
            "loop_type": str(row["loop_type"] or ""),
            "model_type": str(row["model_type"] or "FOPDT"),
            "loop_uri": str(row["loop_uri"] or ""),
            "final_strategy": str(row["final_strategy"] or ""),
            "model": {
                "model_type": str(row["model_type"] or "FOPDT"),
                "K": float(row["model_k"] or 0.0),
                "T": float(row["model_t"] or 0.0),
                "L": float(row["model_l"] or 0.0),
                "normalized_rmse": float(row["normalized_rmse"] or 0.0),
                "r2_score": float(row["r2_score"] or 0.0),
            },
            "evaluation": {
                "final_rating": float(row["final_rating"] or 0.0),
                "performance_score": float(row["performance_score"] or 0.0),
                "passed": bool(row["passed"]),
            },
            "reuse": {
                "hit_count": int(row["hit_count"] or 0),
                "follow_up_success_count": int(row["follow_up_success_count"] or 0),
                "last_hit_at": str(row["last_hit_at"] or ""),
            },
            "tags": json.loads(row["tags_json"] or "[]"),
        }
        for row in rows
    ]


def get_experience_detail(experience_id: str) -> Dict[str, Any] | None:
    ensure_experience_store()
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT record_json FROM experiences WHERE experience_id = ?",
            (experience_id,),
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["record_json"])
    except Exception:
        return None


def register_experience_references(
    referenced_experience_ids: List[str],
    *,
    hit_time: str,
    follow_up_passed: bool,
    follow_up_final_rating: float,
) -> Dict[str, Any]:
    ensure_experience_store()
    ids = [str(item) for item in referenced_experience_ids if str(item)]
    if not ids:
        return {"updated": 0}

    updated = 0
    with _get_connection() as conn:
        for experience_id in ids:
            row = conn.execute(
                "SELECT record_json, hit_count, follow_up_success_count FROM experiences WHERE experience_id = ?",
                (experience_id,),
            ).fetchone()
            if not row:
                continue
            try:
                record = json.loads(row["record_json"])
            except Exception:
                record = {}

            reuse = dict(record.get("reuse") or {})
            reuse["hit_count"] = int(reuse.get("hit_count", row["hit_count"] or 0) or 0) + 1
            reuse["last_hit_at"] = hit_time
            reuse["last_follow_up_passed"] = bool(follow_up_passed)
            reuse["last_follow_up_final_rating"] = float(follow_up_final_rating or 0.0)
            if follow_up_passed:
                reuse["follow_up_success_count"] = int(
                    reuse.get("follow_up_success_count", row["follow_up_success_count"] or 0) or 0
                ) + 1
            else:
                reuse["follow_up_success_count"] = int(
                    reuse.get("follow_up_success_count", row["follow_up_success_count"] or 0) or 0
                )
            record["reuse"] = reuse

            conn.execute(
                """
                UPDATE experiences
                SET hit_count = ?,
                    follow_up_success_count = ?,
                    last_hit_at = ?,
                    record_json = ?
                WHERE experience_id = ?
                """,
                (
                    int(reuse["hit_count"]),
                    int(reuse["follow_up_success_count"]),
                    str(hit_time),
                    json.dumps(record, ensure_ascii=False),
                    experience_id,
                ),
            )
            updated += 1
        conn.commit()
    return {"updated": updated}
