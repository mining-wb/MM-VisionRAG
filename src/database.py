# ====== SQLite 会话与对话历史 ======
# 大模型无状态，对话状态由后端存这里，拼进 Prompt 用

import sqlite3
from pathlib import Path
from typing import Optional

# 默认库放在 data 目录
_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "chat.db"


def _get_conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    """拿连接，库文件不存在会先建表。"""
    path = db_path or str(_DEFAULT_DB)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """建会话表与消息表。"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
    """)
    conn.commit()


def ensure_session(session_id: str, db_path: Optional[str] = None) -> None:
    """没有则插入一条会话。"""
    conn = _get_conn(db_path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
            (session_id,),
        )
        conn.commit()
    finally:
        conn.close()


def add_message(session_id: str, role: str, content: str, db_path: Optional[str] = None) -> None:
    """追加一条消息。role 为 user / assistant。"""
    ensure_session(session_id, db_path)
    conn = _get_conn(db_path)
    try:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_messages(
    session_id: str,
    limit: int = 6,
    db_path: Optional[str] = None,
) -> list[dict]:
    """取最近 N 条消息（按时间正序），用于拼 Prompt。例如 limit=6 即最近 3 轮。"""
    conn = _get_conn(db_path)
    try:
        cur = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    # 倒回来成时间正序
    out = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    return out
