import sqlite3
import pandas as pd

# Load Dataset

df = pd.read_csv(
    "data/segmented_customers.csv"
)

# Create Database

conn = sqlite3.connect(
    "customers.db"
)

# Save Data

df.to_sql(
    "customers",
    conn,
    if_exists="replace",
    index=False
)

print("Database Created Successfully!")

conn.close()
