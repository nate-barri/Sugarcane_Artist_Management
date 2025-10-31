import pandas as pd
import psycopg2
import emoji
import sys
import os
from datetime import datetime
from psycopg2.extras import execute_values

db_params = {
    'dbname': os.environ.get('PGDATABASE', 'neondb'),
    'user': os.environ.get('PGUSER', 'neondb_owner'),
    'password': os.environ.get('PGPASSWORD'),
    'host': os.environ.get('PGHOST'),
    'port': os.environ.get('PGPORT', '5432')
}

# Function to remove emojis
def remove_emojis(text):
    return emoji.replace_emoji(text, "") if isinstance(text, str) else text

# Function to load and clean CSV data
def load_csv(file_path):
    try:
        # Attempt to read the CSV with a different encoding (e.g., ISO-8859-1)
        df = pd.read_csv(file_path, encoding='ISO-8859-1')  # or 'latin1'
    except UnicodeDecodeError:
        print(f"[ERROR] Error reading {file_path}: Unable to decode the file.")
        return None
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(r'[^a-z0-9_]', '_', regex=True)
    df.columns = df.columns.str.replace(r'^[0-9]', 'col_', regex=True)  # Prefix numbers with "col_"
    
    # Rename specific columns for consistency
    column_mappings = {
        'negative_feedback_from_users__hide': 'negative_feedback',
        'unique_negative_feedback_from_users__hide': 'unique_negative_feedback'
    }
    df.rename(columns=column_mappings, inplace=True)
    
    # Replace NaNs with None
    df = df.where(pd.notna(df), None)
    
    # Remove emojis from text columns
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].apply(remove_emojis)
    
    # Convert publish_time to datetime and extract components
    if 'publish_time' in df.columns:
        df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
        df['year'] = df['publish_time'].dt.year
        df['month'] = df['publish_time'].dt.month
        df['day'] = df['publish_time'].dt.day
        df['time'] = df['publish_time'].dt.strftime("%H:%M:%S")
    
    # Convert numeric columns to DECIMAL(10,2)
    decimal_columns = ['duration_sec', 'seconds_viewed', 'average_seconds_viewed']
    
    for col in decimal_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0).astype(float).round(2)
    
    # Convert boolean fields
    boolean_columns = ['is_crosspost', 'is_share']
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    # Set impressions to reach for photo posts, otherwise NULL
    if 'post_type' in df.columns and 'reach' in df.columns:
        df['impressions'] = df.apply(lambda row: row['reach'] if row['post_type'] == 'photo' else None, axis=1)
    
    # Convert important numeric fields to NUMERIC (instead of VARCHAR)
    numeric_columns = ['post_id', 'reach', 'shares', 'comments', 'reactions', 'impressions']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove duplicate rows based on post_id before insert
    df = df.drop_duplicates(subset=['post_id'], keep='last')
    
    return df

# Insert data into PostgreSQL (NeonDB)
def insert_data(df, table_name):
    if df.empty:
        print("[WARNING] No data to insert.")
        return 0
    
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        # Get table column names
        cur.execute(f"""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = '{table_name}'
        """)
        table_columns = {row[0] for row in cur.fetchall()}
        
        # Keep only columns that exist in the table
        df = df[[col for col in df.columns if col in table_columns]]
        
        # Insert data with explicit handling in SQL
        columns = ', '.join(df.columns)
        sql = f"""
        INSERT INTO {table_name} ({columns}) 
        VALUES %s
        ON CONFLICT (post_id) DO UPDATE
        SET {', '.join([f'{col} = EXCLUDED.{col}' for col in df.columns if col != 'post_id'])};
        """
        
        execute_values(cur, sql, df.values.tolist())
        conn.commit()
        cur.close()
        print(f"[SUCCESS] Data successfully inserted! {len(df)} records processed.")
        return len(df)
    except Exception as e:
        print("[ERROR] Error inserting data:", e)
        return 0
    finally:
        if conn:
            conn.close()

# Create table function for NeonDB
def create_table():
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS facebook_data_set(
            post_id NUMERIC PRIMARY KEY,
            page_id VARCHAR,
            page_name TEXT,
            title TEXT,
            description TEXT,
            post_type TEXT,
            duration_sec NUMERIC(10,2) DEFAULT 0,
            publish_time TIMESTAMP,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            time TEXT,
            permalink TEXT,
            is_crosspost BOOLEAN,
            is_share BOOLEAN,
            funded_content_status TEXT,
            reach NUMERIC(20,2),
            shares NUMERIC(20,2),
            comments NUMERIC(20,2),
            reactions NUMERIC(20,2),
            seconds_viewed NUMERIC(10,2),
            average_seconds_viewed NUMERIC(10,2) DEFAULT 0,
            impressions NUMERIC(20,2) DEFAULT NULL
        );
        """
        
        cur.execute(create_table_sql)
        conn.commit()
        cur.close()
        print("[SUCCESS] Table ensured to exist!")
    except Exception as e:
        print("[ERROR] Error creating table:", e)
    finally:
        if conn:
            conn.close()

# Function to ensure platform exists in dw.dim_platform
def ensure_platform_exists():
    """Ensure 'facebook' platform exists in dw.dim_platform table, create if missing"""
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'dw'")
        if not cur.fetchone():
            print("[WARNING] Schema 'dw' does not exist. Creating it...")
            cur.execute("CREATE SCHEMA IF NOT EXISTS dw")
            conn.commit()
        
        cur.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'dw' AND table_name = 'dim_platform'
        """)
        if not cur.fetchone():
            print("[WARNING] Table 'dw.dim_platform' does not exist. Creating it...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dw.dim_platform (
                    platform_sk SERIAL PRIMARY KEY,
                    platform_code VARCHAR(50) UNIQUE NOT NULL,
                    platform_name VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        
        cur.execute("SELECT platform_sk FROM dw.dim_platform WHERE platform_code = 'facebook'")
        result = cur.fetchone()
        
        if not result:
            print("[WARNING] Platform 'facebook' not found. Creating it automatically...")
            cur.execute("""
                INSERT INTO dw.dim_platform (platform_code, platform_name) 
                VALUES ('facebook', 'Facebook')
                ON CONFLICT (platform_code) DO NOTHING
                RETURNING platform_sk
            """)
            result = cur.fetchone()
            conn.commit()
            print(f"[SUCCESS] Platform 'facebook' created with platform_sk={result[0]}")
        else:
            print(f"[SUCCESS] Platform 'facebook' found with platform_sk={result[0]}")
        
        cur.close()
        return True
        
    except Exception as e:
        print(f"[ERROR] Error ensuring platform exists: {e}")
        return False
    finally:
        if conn:
            conn.close()

# Main function
def ingest_data(file_path):
    if not ensure_platform_exists():
        print("[ERROR] Cannot proceed with data insertion. Please fix the platform configuration first.")
        sys.exit(1)
    
    table_name = "facebook_data_set"
    df = load_csv(file_path)
    if df is not None:
        return insert_data(df, table_name)
    return 0

if __name__ == "__main__":
    create_table()
    if len(sys.argv) > 1:
        records = ingest_data(sys.argv[1])
    else:
        records = ingest_data("Facebook/FULL_SET_FB.csv")
    print(f"RECORDS: {records}")
