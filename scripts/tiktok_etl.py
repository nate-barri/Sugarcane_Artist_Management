import re, hashlib
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import emoji
from datetime import datetime, timezone
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

# ================= HELPERS =================
def read_csv_robust(path):
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc, engine="python")
            print(f"[SUCCESS] Loaded TikTok CSV with encoding: {enc}")
            return df
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(path, encoding="latin1", engine="python", sep=None)
    print("[SUCCESS] Loaded TikTok CSV with fallback: latin1 + auto-delimiter")
    return df

def remove_emoji(text):
    if not isinstance(text, str):
        return text
    return emoji.replace_emoji(text, replace="").strip()

def to_int_or_none(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        s = str(x).strip()
        if s == "":
            return None
        return int(float(s))
    except:
        return None

def normalize_post_type(pt):
    if pt is None or (isinstance(pt, float) and np.isnan(pt)):
        return None
    pt = str(pt).lower().strip()
    if pt == "":
        return None
    
    if pt in ['photos', 'pictures', 'picture']:
        return 'photo'
    elif pt in ['videos']:
        return 'video'
    
    return pt

def parse_duration_to_hms(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return (0, 0, 0, None)
    s = str(s).strip().lower()
    if s == "":
        return (0, 0, 0, None)

    if ":" in s:
        parts = s.split(":")
        try:
            parts = [int(p) for p in parts]
            if len(parts) == 3:
                h, m, sec = parts
            elif len(parts) == 2:
                h, m, sec = 0, parts[0], parts[1]
            elif len(parts) == 1:
                h, m, sec = 0, 0, parts[0]
            else:
                h, m, sec = 0, 0, 0
            total = h*3600 + m*60 + sec
            return (h, m, sec, total)
        except:
            pass

    h = m = sec = 0
    mh = re.search(r"(\d+)\s*h", s)
    mm = re.search(r"(\d+)\s*m", s)
    ms = re.search(r"(\d+)\s*s", s)
    if mh or mm or ms:
        if mh:
            h = int(mh.group(1))
        if mm:
            m = int(mm.group(1))
        if ms:
            sec = int(ms.group(1))
        return (h, m, sec, h*3600 + m*60 + sec)

    if re.fullmatch(r"\d+(\.\d+)?", s):
        try:
            sec = int(float(s))
            return (0, 0, sec, sec)
        except:
            return (0, 0, 0, None)

    return (0, 0, 0, None)

def make_publish_time(y, m, d):
    y = to_int_or_none(y)
    m = to_int_or_none(m)
    d = to_int_or_none(d)
    if all(v is not None for v in (y, m, d)):
        try:
            return datetime(int(y), int(m), int(d), tzinfo=timezone.utc)
        except:
            return None
    return None

# ================= EXTRACT =================
def process_tiktok_data(file_path):
    df = read_csv_robust(file_path)
    df.columns = [c.strip() for c in df.columns]

    # ================= TRANSFORM =================
    work = pd.DataFrame()
    work["video_id"]   = df.get("tiktok_video_id")
    work["url"]        = df.get("content_link")
    work["title"]      = df.get("video_title")
    work["sound_used"] = df.get("sound_used")
    work["saves"]      = df.get("saves")

    if "post_type" in df.columns:
        work["post_type"] = df["post_type"].apply(normalize_post_type)
    else:
        work["post_type"] = None

    for src, dst in [
        ("likes","likes"), ("shares","shares"),
        ("comments_added","comments_added"), ("views","views"),
        ("saves","saves")
    ]:
        if src in df.columns:
            work[dst] = df[src].apply(to_int_or_none)
        else:
            work[dst] = None

    duration_col = "duration"
    if duration_col in df.columns:
        hms = df[duration_col].apply(parse_duration_to_hms)
        work["dur_hours"]   = hms.apply(lambda t: to_int_or_none(t[0]))
        work["dur_minutes"] = hms.apply(lambda t: to_int_or_none(t[1]))
        work["dur_seconds"] = hms.apply(lambda t: to_int_or_none(t[2]))
        work["duration_sec"]= hms.apply(lambda t: to_int_or_none(t[3]))
        work["duration"]    = df[duration_col].astype(str)
    else:
        work["dur_hours"] = work["dur_minutes"] = work["dur_seconds"] = work["duration_sec"] = work["duration"] = None

    work["publish_year"]  = df.get("publish_year").apply(to_int_or_none) if "publish_year" in df.columns else None
    work["publish_month"] = df.get("publish_month").apply(to_int_or_none) if "publish_month" in df.columns else None
    work["publish_day"]   = df.get("publish_day").apply(to_int_or_none) if "publish_day" in df.columns else None
    work["publish_time"]  = [
        make_publish_time(y, m, d) for y, m, d in zip(work["publish_year"], work["publish_month"], work["publish_day"])
    ]

    for txt_col in ["title","url","sound_used","post_type"]:
        if txt_col in work.columns:
            work[txt_col] = work[txt_col].apply(remove_emoji)

    work = work.replace({np.nan: None, pd.NaT: None})

    rows_before = len(work)
    work = work[work["video_id"].notna() & (work["video_id"].astype(str).str.strip() != "")]
    rows_after = len(work)
    if rows_after < rows_before:
        print(f"[WARNING] Dropped {rows_before-rows_after} rows with empty tiktok_video_id.")

    return work

def create_table():
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        cur.execute("""
        CREATE SCHEMA IF NOT EXISTS dw;
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS dw.dim_platform (
            platform_sk SERIAL PRIMARY KEY,
            platform_code VARCHAR(50) UNIQUE NOT NULL,
            platform_name VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        cur.execute("""
        INSERT INTO dw.dim_platform (platform_code, platform_name)
        VALUES ('tiktok', 'TikTok')
        ON CONFLICT (platform_code) DO NOTHING;
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS public.tt_video_etl (
            video_id TEXT PRIMARY KEY,
            url TEXT,
            title TEXT,
            publish_time TIMESTAMPTZ,
            publish_year INT,
            publish_month INT,
            publish_day INT,
            duration TEXT,
            dur_hours INT,
            dur_minutes INT,
            dur_seconds INT,
            duration_sec INT,
            likes BIGINT,
            shares BIGINT,
            comments_added BIGINT,
            views BIGINT,
            saves BIGINT,
            sound_used TEXT,
            post_type TEXT
        );
        """)
        conn.commit()
        cur.close()
        print("[SUCCESS] Table ensured to exist!")
    except Exception as e:
        print(f"[ERROR] Error creating table: {e}", file=sys.stderr)
    finally:
        if conn:
            conn.close()

def insert_data(df):
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        insert_sql = """
        INSERT INTO public.tt_video_etl (
            video_id, url, title,
            publish_time, publish_year, publish_month, publish_day,
            duration, dur_hours, dur_minutes, dur_seconds, duration_sec,
            likes, shares, comments_added, views, saves,
            sound_used, post_type
        ) VALUES %s
        ON CONFLICT (video_id) DO UPDATE
        SET url            = COALESCE(EXCLUDED.url,            public.tt_video_etl.url),
            title          = COALESCE(EXCLUDED.title,          public.tt_video_etl.title),
            publish_time   = COALESCE(EXCLUDED.publish_time,   public.tt_video_etl.publish_time),
            publish_year   = COALESCE(EXCLUDED.publish_year,   public.tt_video_etl.publish_year),
            publish_month  = COALESCE(EXCLUDED.publish_month,  public.tt_video_etl.publish_month),
            publish_day    = COALESCE(EXCLUDED.publish_day,    public.tt_video_etl.publish_day),
            duration       = COALESCE(EXCLUDED.duration,       public.tt_video_etl.duration),
            dur_hours      = COALESCE(EXCLUDED.dur_hours,      public.tt_video_etl.dur_hours),
            dur_minutes    = COALESCE(EXCLUDED.dur_minutes,    public.tt_video_etl.dur_minutes),
            dur_seconds    = COALESCE(EXCLUDED.dur_seconds,    public.tt_video_etl.dur_seconds),
            duration_sec   = COALESCE(EXCLUDED.duration_sec,   public.tt_video_etl.duration_sec),
            likes          = COALESCE(EXCLUDED.likes,          public.tt_video_etl.likes),
            shares         = COALESCE(EXCLUDED.shares,         public.tt_video_etl.shares),
            comments_added = COALESCE(EXCLUDED.comments_added, public.tt_video_etl.comments_added),
            views          = COALESCE(EXCLUDED.views,          public.tt_video_etl.views),
            saves          = COALESCE(EXCLUDED.saves,          public.tt_video_etl.saves),
            sound_used     = COALESCE(EXCLUDED.sound_used,     public.tt_video_etl.sound_used),
            post_type      = COALESCE(EXCLUDED.post_type,      public.tt_video_etl.post_type);
        """

        records = []
        for _, r in df.iterrows():
            records.append((
                r.get("video_id"),
                r.get("url"),
                r.get("title"),
                r.get("publish_time"),
                r.get("publish_year"),
                r.get("publish_month"),
                r.get("publish_day"),
                r.get("duration"),
                r.get("dur_hours"),
                r.get("dur_minutes"),
                r.get("dur_seconds"),
                r.get("duration_sec"),
                r.get("likes"),
                r.get("shares"),
                r.get("comments_added"),
                r.get("views"),
                r.get("saves"),
                r.get("sound_used"),
                r.get("post_type"),
            ))

        if records:
            execute_values(cur, insert_sql, records, page_size=1000)
            conn.commit()
            print(f"[SUCCESS] Inserted/Upserted {len(records)} rows into public.tt_video_etl.")
            return len(records)
        else:
            print("[WARNING] No rows to insert (no valid video_id).")
            return 0
            
    except Exception as e:
        print(f"[ERROR] Error inserting data: {e}", file=sys.stderr)
        return 0
    finally:
        if conn:
            conn.close()

def ingest_data(file_path):
    create_table()
    df = process_tiktok_data(file_path)
    return insert_data(df)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tiktok_etl.py <csv_file_path>", file=sys.stderr)
        sys.exit(1)
    
    file_path = sys.argv[1]
    records = ingest_data(file_path)
    print(f"RECORDS: {records}")
