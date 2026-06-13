import sqlite3
import pandas as pd

conn = sqlite3.connect(
    "customers.db"
)

query = """
SELECT *
FROM customers
LIMIT 5
"""

result = pd.read_sql(
    query,
    conn
)

print(result)

conn.close()
