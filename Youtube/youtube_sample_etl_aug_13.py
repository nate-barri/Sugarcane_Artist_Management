import re, hashlib
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import emoji  # pip install emoji

# === DB connection parameters ===
db_params = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host': 'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port': '5432'
}

# === Config ===
CSV_PATH = r"C:\Sugarcane_Artist_Management\Youtube\final_full_set_yt.csv"
ENSURE_TRIGGER = True  # reattach the statement-level trigger if the function exists

# ---------- Helpers ----------
def remove_emoji(text):
    if not isinstance(text, str):
        return text
    return emoji.replace_emoji(text, replace="").strip()

YOUTUBE_ID_RE = re.compile(r"(?:v=|/shorts/|/embed/|youtu\.be/|/watch\?v=)([A-Za-z0-9_-]{11})")

def extract_video_id_from_any(row):
    # Try ID columns then URL columns, then scan all strings
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
            print(f"📄 Loaded CSV with encoding: {enc}")
            return df_local
        except UnicodeDecodeError:
            continue
    # final fallback with delimiter sniffing
    df_local = pd.read_csv(path, encoding="latin1", engine="python", sep=None)
    print("📄 Loaded CSV with fallback: latin1 + auto-delimiter")
    return df_local

# ---------- Extract ----------
df = read_csv_robust(CSV_PATH)
# trim any stray spaces in headers
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
missing_real = df["Video_ID_real"].isna().sum()
if missing_real:
    print(f"ℹ️ {missing_real} rows lack a real YouTube ID; generating surrogate IDs.")
df["Video_ID"] = df["Video_ID_real"]
df.loc[df["Video_ID"].isna(), "Video_ID"] = df[df["Video_ID"].isna()].apply(make_surrogate_id, axis=1)

# Compute Average views per viewer if not provided: Views / Unique viewers
if "Average views per viewer" not in df.columns:
    df["Average views per viewer"] = df.apply(
        lambda r: safe_div(r.get("Views"), r.get("Unique viewers")),
        axis=1
    )

# Emoji strip on all text columns (incl. Category, Content, Video title)
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].apply(remove_emoji)

# Replace pandas NA with None for Postgres
df = df.replace({np.nan: None, pd.NaT: None})

# ---------- Load (landing) ----------
conn = psycopg2.connect(**db_params)
cur = conn.cursor()

# Ensure table exists (same datatypes as before)
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
    avg_dur_hours INT,
    avg_dur_minutes INT,
    avg_dur_seconds INT,
    category TEXT
);
""")
conn.commit()

# Add new columns for your new dataset (no changes to existing types)
cur.execute("ALTER TABLE yt_video_etl ADD COLUMN IF NOT EXISTS unique_viewers BIGINT;")
conn.commit()

# Ensure the statement-level trigger is attached (if the trigger function exists)
if ENSURE_TRIGGER:
    cur.execute("""
    DO $$
    BEGIN
      IF EXISTS (
        SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
        WHERE n.nspname='dw' AND p.proname='trg_yt_sync_stmt'
      ) THEN
        IF NOT EXISTS (
          SELECT 1 FROM pg_trigger t
          JOIN pg_class c ON c.oid=t.tgrelid
          JOIN pg_namespace ns ON ns.oid=c.relnamespace
          WHERE ns.nspname='public' AND c.relname='yt_video_etl' AND t.tgname='yt_sync_stmt_insupd'
        ) THEN
          EXECUTE 'CREATE TRIGGER yt_sync_stmt_insupd
                   AFTER INSERT OR UPDATE ON public.yt_video_etl
                   FOR EACH STATEMENT
                   EXECUTE FUNCTION dw.trg_yt_sync_stmt()';
        END IF;
      ELSE
        RAISE NOTICE 'dw.trg_yt_sync_stmt() not found; skipping trigger creation.';
      END IF;
    END$$;
    """)
    conn.commit()

# Fresh load while keeping the trigger
cur.execute("TRUNCATE TABLE yt_video_etl;")
conn.commit()

insert_sql = """
INSERT INTO yt_video_etl (
    video_id, content, video_title,
    publish_day, publish_month, publish_year,
    duration, impressions, impressions_ctr,
    avg_views_per_viewer, likes, dislikes, shares, comments_added, views,
    watch_time_hours, unique_viewers, category,
    avg_dur_hours, avg_dur_minutes, avg_dur_seconds
) VALUES %s
ON CONFLICT (video_id) DO NOTHING;
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

# Execute the insertion logic for the updated records
if records:
    execute_values(cur, insert_sql, records, page_size=1000)
    conn.commit()
    print(f"✅ Inserted {len(records)} rows into landing table (yt_video_etl).")
else:
    print("⚠️ No rows to insert.")

cur.close()
conn.close()

print("ETL complete: encoding-robust CSV load, new columns handled, CTR converted to 0–1, IDs extracted/surrogated, trigger will auto-sync DW after the bulk INSERT.")
