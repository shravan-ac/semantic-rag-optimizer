import sqlite3
from datetime import datetime,timezone
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parents[2]
DB_PATH=BASE_DIR/"data"/"rag_analytics.db"

def initialize_database():
	DB_PATH.parent.mkdir(parents=True,exist_ok=True)
	with sqlite3.connect(DB_PATH) as connection:
		connection.execute("""
		CREATE TABLE IF NOT EXISTS request_logs(
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TEXT NOT NULL,
			question TEXT NOT NULL,
			model TEXT NOT NULL,
			prompt_tokens INTEGER DEFAULT 0,
			completion_tokens INTEGER DEFAULT 0,
			total_tokens INTEGER DEFAULT 0,
			latency_ms REAL NOT NULL,
			estimated_cost_usd REAL,
			cache_hit INTEGER DEFAULT 0,
			route TEXT
			)
		""")
		connection.commit()

def log_request(
		question:str,
		model:str,
		prompt_tokens:int,
		completion_tokens:int,
		total_tokens:int,
		latency_ms:float,
		estimated_cost_usd:float|None=None,
		cache_hit: bool =False,
		route:str="default"
	):
	initialize_database()
	with sqlite3.connect(DB_PATH) as connection:
		connection.execute("""
		INSERT INTO request_logs(
		timestamp,
		question,
		model,
		prompt_tokens,
		completion_tokens,
		total_tokens,
		latency_ms,
		estimated_cost_usd,
		cache_hit,
		route
		)
		VALUES (?,?,?,?,?,?,?,?,?,?)
		""",(
			datetime.now(timezone.utc).isoformat(),
			question,
			model,
			prompt_tokens,
			completion_tokens,
			total_tokens,
			latency_ms,
			estimated_cost_usd,
			int(cache_hit),
			route
		))

		connection.commit()

if __name__ == "__main__":
	initialize_database()
	print(f"Analytics database intializer: {DB_PATH}")
