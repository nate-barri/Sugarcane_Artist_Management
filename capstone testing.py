import psycopg2

POSTGRES_CONFIG = {
    "host": "localhost",
    "database": "capstone",
    "user": "postgres",  # Update with your actual username
    "password": "admin",  # Update with your actual password
    "port": "5432"
}

try:
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    print("✅ Connected to PostgreSQL successfully!")
    conn.close()
except Exception as e:
    print(f"❌ Error: {e}")
