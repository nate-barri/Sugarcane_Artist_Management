import re, hashlib
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import emoji
import sys
import os

# Get database connection from environment variables
db_params = {
    'dbname': os.environ.get('PGDATABASE', 'neondb'),
    'user': os.environ.get('PGUSER', 'neondb_owner'),
    'password': os.environ.get('PGPASSWORD'),
    'host': os.environ.get('PGHOST'),
    'port': os.environ.get('PGPORT', '5432')
}

# ---------- Helpers ----------
def remove_emoji(text):
    if not isinstance(text, str):
        return text
    return emoji.replace_emoji(text, replace="").strip()

YOUTUBE_ID_RE = re.compile(r"(?:v=|/shorts/|/embed/|youtu\.be/|/watch\?v=)([A-Za-z0-9_-]{11})")

def extract_video_id_from_any(row):
    id_cols = ["Video ID","Video Id","VideoId","External video ID","External Video ID","External Video Id","External Video Id "]
    url_cols = ["Video URL","URL","Watch Page","Watch URL","Video link","Link","Video"]
    for c in id_cols:
        if c in row and pd.notna(row[c]):
            s = str(row[c]).strip()
            if len(s) == 11 and re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
                return s
            m = YOUTUBE_ID_RE.search(s)
            if m: return m.group(1)
    for c in url_cols:
        if c in row and pd.notna(row[c]):
            s = str(row[c]).strip()
            m = YOUTUBE_ID_RE.search(s)
            if m: return m.group(1)
    for k, v in row.items():
        if isinstance(v, str):
            m = YOUTUBE_ID_RE.search(v)
            if m: return m.group(1)
    return None

def make_surrogate_id(row):
    parts = [
        str(row.get("Video title") or "").strip().lower(),
        str(row.get("Publish_Year") or ""),
        str(row.get("Publish_Month") or ""),
        str(row.get("Publish_Day") or ""),
        str(row.get("Duration") or "")
    ]
    key = "|".join(parts)
    return f"s_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:20]}"

def parse_duration_to_hms(duration_str):
    if not isinstance(duration_str, str):
        return 0, 0, 0
    try:
        parts = [int(p) for p in duration_str.split(":")]
        if len(parts) == 3: return parts[0], parts[1], parts[2]
        if len(parts) == 2: return 0, parts[0], parts[1]
        if len(parts) == 1: return 0, 0, parts[0]
    except:
        pass
    return 0, 0, 0

def pct_to_float(val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return None
    s = str(val).strip()
    if s.endswith("%"): s = s[:-1]
    try: return float(s)/100.0
    except: return None

def safe_div(a, b):
    try:
        if a is None or b in (None, 0): return None
        return float(a) / float(b)
    except:
        return None

def read_csv_robust(path):
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            df_local = pd.read_csv(path, encoding=enc, engine="python")
            return df_local
        except UnicodeDecodeError:
            continue
    df_local = pd.read_csv(path, encoding="latin1", engine="python", sep=None)
    return df_local

def process_youtube_data(file_path):
    # ---------- Extract ----------
    df = read_csv_robust(file_path)
    df.columns = [c.strip() for c in df.columns]

    # ---------- Transform ----------
    # Publish time → parts
    df["Video publish time"] = df["Video publish time"].apply(lambda x: pd.to_datetime(x, errors="coerce", utc=True))
    df["Publish_Day"] = df["Video publish time"].dt.day
    df["Publish_Month"] = df["Video publish time"].dt.month
    df["Publish_Year"] = df["Video publish time"].dt.year
    df.drop(columns=["Video publish time"], inplace=True)

    # Average view duration → H/M/S
    df["Avg_Dur_Hours"], df["Avg_Dur_Minutes"], df["Avg_Dur_Seconds"] = zip(
        *df["Average view duration"].map(parse_duration_to_hms)
    )
    df.drop(columns=["Average view duration"], inplace=True)

    # CTR % → 0..1
    if "Impressions click-through rate (%)" in df.columns:
        df["Impressions_ctr"] = df["Impressions click-through rate (%)"].map(pct_to_float)
    else:
        df["Impressions_ctr"] = None

    # Real YouTube ID or surrogate
    df["Video_ID_real"] = df.apply(extract_video_id_from_any, axis=1)
    df["Video_ID"] = df["Video_ID_real"]
    df.loc[df["Video_ID"].isna(), "Video_ID"] = df[df["Video_ID"].isna()].apply(make_surrogate_id, axis=1)

    # Compute Average views per viewer if not provided
    if "Average views per viewer" not in df.columns:
        df["Average views per viewer"] = df.apply(
            lambda r: safe_div(r.get("Views"), r.get("Unique viewers")),
            axis=1
        )

    # Emoji strip on all text columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(remove_emoji)

    # Replace pandas NA with None for Postgres
    df = df.replace({np.nan: None, pd.NaT: None})
    
    return df

def create_table():
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        cur.execute(""" 
        CREATE TABLE IF NOT EXISTS yt_video_etl (
            video_id TEXT PRIMARY KEY,
            content TEXT,
            video_title TEXT,
            publish_day INT,
            publish_month INT,
            publish_year INT,
            duration TEXT,
            impressions BIGINT,
            impressions_ctr FLOAT,
            avg_views_per_viewer FLOAT,
            likes BIGINT,
            dislikes BIGINT,
            shares BIGINT,
            comments_added BIGINT,
            views BIGINT,
            watch_time_hours FLOAT,
            unique_viewers BIGINT,
            category TEXT,
            avg_dur_hours INT,
            avg_dur_minutes INT,
            avg_dur_seconds INT
        );
        """)
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"Error creating table: {e}", file=sys.stderr)
    finally:
        if conn:
            conn.close()

def insert_data(df):
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        insert_sql = """
        INSERT INTO yt_video_etl (
            video_id, content, video_title,
            publish_day, publish_month, publish_year,
            duration, impressions, impressions_ctr,
            avg_views_per_viewer, likes, dislikes, shares, comments_added, views,
            watch_time_hours, unique_viewers, category,
            avg_dur_hours, avg_dur_minutes, avg_dur_seconds
        ) VALUES %s
        ON CONFLICT (video_id) DO UPDATE SET
            content = EXCLUDED.content,
            video_title = EXCLUDED.video_title,
            publish_day = EXCLUDED.publish_day,
            publish_month = EXCLUDED.publish_month,
            publish_year = EXCLUDED.publish_year,
            duration = EXCLUDED.duration,
            impressions = EXCLUDED.impressions,
            impressions_ctr = EXCLUDED.impressions_ctr,
            avg_views_per_viewer = EXCLUDED.avg_views_per_viewer,
            likes = EXCLUDED.likes,
            dislikes = EXCLUDED.dislikes,
            shares = EXCLUDED.shares,
            comments_added = EXCLUDED.comments_added,
            views = EXCLUDED.views,
            watch_time_hours = EXCLUDED.watch_time_hours,
            unique_viewers = EXCLUDED.unique_viewers,
            category = EXCLUDED.category,
            avg_dur_hours = EXCLUDED.avg_dur_hours,
            avg_dur_minutes = EXCLUDED.avg_dur_minutes,
            avg_dur_seconds = EXCLUDED.avg_dur_seconds;
        """

        def g(row, name): return row.get(name)

        records = []
        for _, r in df.iterrows():
            records.append((
                g(r,"Video_ID"),
                g(r,"Content"),
                g(r,"Video title"),
                g(r,"Publish_Day"), g(r,"Publish_Month"), g(r,"Publish_Year"),
                g(r,"Duration"),
                g(r,"Impressions"),
                g(r,"Impressions_ctr"),
                g(r,"Average views per viewer"),
                g(r,"Likes"),
                g(r,"Dislikes"),
                g(r,"Shares"),
                g(r,"Comments added"),
                g(r,"Views"),
                g(r,"Watch time (hours)"),
                g(r,"Unique viewers"),
                g(r,"Category"),
                g(r,"Avg_Dur_Hours"),
                g(r,"Avg_Dur_Minutes"),
                g(r,"Avg_Dur_Seconds"),
            ))

        if records:
            execute_values(cur, insert_sql, records, page_size=1000)
            conn.commit()
            print(f"SUCCESS: Inserted {len(records)} records")
            return len(records)
        else:
            print("No rows to insert.", file=sys.stderr)
            return 0
            
    except Exception as e:
        print(f"Error inserting data: {e}", file=sys.stderr)
        return 0
    finally:
        if conn:
            conn.close()

def ensure_platform_exists():
    """Ensure 'youtube' platform exists in dw.dim_platform table, create if missing"""
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        # Check if dw schema exists, create if missing
        cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'dw'")
        if not cur.fetchone():
            print("[WARNING] Schema 'dw' does not exist. Creating it...")
            cur.execute("CREATE SCHEMA IF NOT EXISTS dw")
            conn.commit()
        
        # Check if dim_platform table exists, create if missing
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
        
        # Check if 'youtube' platform exists, insert if missing
        cur.execute("SELECT platform_sk FROM dw.dim_platform WHERE platform_code = 'youtube'")
        result = cur.fetchone()
        
        if not result:
            print("[WARNING] Platform 'youtube' not found. Creating it automatically...")
            cur.execute("""
                INSERT INTO dw.dim_platform (platform_code, platform_name) 
                VALUES ('youtube', 'YouTube')
                ON CONFLICT (platform_code) DO NOTHING
                RETURNING platform_sk
            """)
            result = cur.fetchone()
            conn.commit()
            print(f"[SUCCESS] Platform 'youtube' created with platform_sk={result[0]}")
        else:
            print(f"[SUCCESS] Platform 'youtube' found with platform_sk={result[0]}")
        
        cur.close()
        return True
        
    except Exception as e:
        print(f"[ERROR] Error ensuring platform exists: {e}")
        return False
    finally:
        if conn:
            conn.close()

def ingest_data(file_path):
    if not ensure_platform_exists():
        print("[ERROR] Cannot proceed with data insertion. Please fix the platform configuration first.")
        sys.exit(1)
    
    create_table()
    df = process_youtube_data(file_path)
    return insert_data(df)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python youtube_etl.py <csv_file_path>", file=sys.stderr)
        sys.exit(1)
    
    file_path = sys.argv[1]
    records = ingest_data(file_path)
    print(f"RECORDS: {records}")
