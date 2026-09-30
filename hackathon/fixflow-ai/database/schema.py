"""
Database schema module.
"""
import sqlite3
from typing import Optional

def get_connection(db_path: str = 'fixflow.db') -> sqlite3.Connection:
    """Get SQLite database connection."""
    return sqlite3.connect(db_path)

def create_tables(db_path: str = 'fixflow.db') -> None:
    """Create all required tables."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id INTEGER,
            prediction TEXT,
            FOREIGN KEY(complaint_id) REFERENCES complaints(id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS priorities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id INTEGER,
            score REAL,
            FOREIGN KEY(complaint_id) REFERENCES complaints(id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS optimization_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            details TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metrics TEXT,
            run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
