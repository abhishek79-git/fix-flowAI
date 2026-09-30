"""
Database repository module.
"""
import sqlite3
from typing import Dict, Any, List, Optional
import json

def save_complaint(conn: sqlite3.Connection, complaint_data: Dict[str, Any]) -> int:
    """Save a complaint to DB."""
    cursor = conn.cursor()
    cursor.execute("INSERT INTO complaints (text) VALUES (?)", (complaint_data.get('text', ''),))
    conn.commit()
    return cursor.lastrowid or 0

def get_complaints(conn: sqlite3.Connection, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Get complaints based on filters."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, text FROM complaints")
    return [{"id": row[0], "text": row[1]} for row in cursor.fetchall()]

def save_prediction(conn: sqlite3.Connection, complaint_id: int, prediction: Any) -> None:
    """Save prediction for a complaint."""
    cursor = conn.cursor()
    cursor.execute("INSERT INTO predictions (complaint_id, prediction) VALUES (?, ?)", (complaint_id, str(prediction)))
    conn.commit()

def save_benchmark_result(conn: sqlite3.Connection, result: Any) -> None:
    """Save benchmark result."""
    cursor = conn.cursor()
    cursor.execute("INSERT INTO benchmark_runs (metrics) VALUES (?)", (str(result),))
    conn.commit()

def get_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Get overall statistics."""
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM complaints")
    row = cursor.fetchone()
    count = row[0] if row else 0
    return {
        "total_complaints": count,
        "severity_distribution": {}
    }
