import re, hashlib
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import emoji  # pip install emoji

# --- DB connection parameters ---
db_params = {
    'dbname': 'test1',
    'user': 'postgres',
    'password': 'admin',
    'host': 'localhost',
    'port': '5432'
}

# ---------- Helpers ----------
def remove_emoji(text):
    if not isinstance(text, str):
        return text
    return emoji.replace_emoji(text, replace="").strip()

YOUTUBE_ID_RE = re.compile(r"(?:v=|/shorts/|/embed/|youtu\.be/|/watch\?v=)([A-Za-z0-9_-]{11})")

def extract_video_id_from_any(row):
    """Try to extract a real 11-char YouTube ID from common columns/URLs."""
    candidate_id_cols = [
        "Video ID","Video Id","VideoId",
        "External video ID","External Video ID","External Video Id","External Video Id "
    ]
    candidate_url_cols = ["Video URL","URL","Watch Page","Watch URL","Video link","Link","Video"]
    # direct id columns
    for c in candidate_id_cols:
        if c in row and pd.notna(row[c]):
            s = str(row[c]).strip()
            if len(s) == 11 and re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
                return s
            m = YOUTUBE_ID_RE.search(s)
            if m: return m.group(1)
    # url-like columns
    for c in candidate_url_cols:
        if c in row and pd.notna(row[c]):
            s = str(row[c]).strip()
            m = YOUTUBE_ID_RE.search(s)
            if m: return m.group(1)
    # scan all string fields
    for k, v in row.items():
        if isinstance(v, str):
            m = YOUTUBE_ID_RE.search(v)
            if m: return m.group(1)
    return None

def make_surrogate_id(row):
    """Deterministic ID when real YouTube ID is missing (prevents duplicates)."""
    parts = [
        str(row.get("Video title") or "").strip().lower(),
        str(row.get("Publish_Year") or ""),
        str(row.get("Publish_Month") or ""),
        str(row.get("Publish_Day") or ""),
        str(row.get("Duration") or "")
    ]
    key = "|".join(parts)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    return f"s_{digest}"

def parse_duration(duration_str):
    if not isinstance(duration_str, str):
        return 0, 0, 0
    try:
        parts = [int(p) for p in duration_str.split(":")]
        if len(parts) == 3: return parts[0], parts[1], parts[2]
        if len(parts) == 2: return 0, parts[0], parts[1]
        if len(parts) == 1: return 0, 0, parts[0]
    except: pass
    return 0, 0, 0

def pct_to_float(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s.endswith("%"): s = s[:-1]
    try: return float(s) / 100.0
    except: return None

# ---------- Extract ----------
df = pd.read_csv(r"C:\Sugarcane_Artist_Management\Youtube\YOUTUBE - SUGARCANE CONTENT DATA.csv")

# ---------- Transform ----------
# Parse publish time quietly (handles mixed formats; avoid warnings)
df["Video publish time"] = df["Video publish time"].apply(
    lambda x: pd.to_datetime(x, errors="coerce", utc=True)
)

# Date parts, drop raw datetime
df["Publish_Day"] = df["Video publish time"].dt.day
df["Publish_Month"] = df["Video publish time"].dt.month
df["Publish_Year"] = df["Video publish time"].dt.year
df.drop(columns=["Video publish time"], inplace=True)

# Parse duration to H/M/S, drop original
df["Avg_Dur_Hours"], df["Avg_Dur_Minutes"], df["Avg_Dur_Seconds"] = zip(
    *df["Average view duration"].map(parse_duration)
)
df.drop(columns=["Average view duration"], inplace=True)

# CTR to float 0–1 (create if missing to keep schema)
if "Impressions click-through rate (%)" in df.columns:
    df["Impressions_ctr"] = df["Impressions click-through rate (%)"].map(pct_to_float)
else:
    df["Impressions_ctr"] = None

# Derive Video_ID; fallback to deterministic surrogate if missing
df["Video_ID_real"] = df.apply(extract_video_id_from_any, axis=1)
missing_real = df["Video_ID_real"].isna().sum()
if missing_real:
    print(f"Info: {missing_real} rows lack a real YouTube ID; generating surrogate IDs.")
df["Video_ID"] = df["Video_ID_real"]
df.loc[df["Video_ID"].isna(), "Video_ID"] = df[df["Video_ID"].isna()].apply(make_surrogate_id, axis=1)

# Emoji strip on all text/object cols
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].apply(remove_emoji)

# Nulls for Postgres
df = df.replace({np.nan: None, pd.NaT: None})

# ---------- Load ----------
conn = psycopg2.connect(**db_params)
cursor = conn.cursor()

# Recreate table to ensure correct schema (avoids 'video_id does not exist')
cursor.execute("DROP TABLE IF EXISTS yt_video_etl;")
conn.commit()

cursor.execute("""
CREATE TABLE yt_video_etl (
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
    new_viewers BIGINT,
    subscribers_gained BIGINT,
    subscribers_lost BIGINT,
    likes BIGINT,
    shares BIGINT,
    comments_added BIGINT,
    views BIGINT,
    watch_time_hours FLOAT,
    avg_dur_hours INT,
    avg_dur_minutes INT,
    avg_dur_seconds INT
);
""")
conn.commit()

insert_query = """
INSERT INTO yt_video_etl (
    video_id, content, video_title,
    publish_day, publish_month, publish_year,
    duration, impressions, impressions_ctr,
    avg_views_per_viewer, new_viewers,
    subscribers_gained, subscribers_lost, likes, shares,
    comments_added, views, watch_time_hours,
    avg_dur_hours, avg_dur_minutes, avg_dur_seconds
) VALUES %s
ON CONFLICT (video_id) DO NOTHING;
"""

def get(row, name): return row.get(name)

data_tuples = [
    (
        get(row, "Video_ID"),
        get(row, "Content"),
        get(row, "Video title"),
        get(row, "Publish_Day"),
        get(row, "Publish_Month"),
        get(row, "Publish_Year"),
        get(row, "Duration"),
        get(row, "Impressions"),
        get(row, "Impressions_ctr"),
        get(row, "Average views per viewer"),
        get(row, "New viewers"),
        get(row, "Subscribers gained"),
        get(row, "Subscribers lost"),
        get(row, "Likes"),
        get(row, "Shares"),
        get(row, "Comments added"),
        get(row, "Views"),
        get(row, "Watch time (hours)"),
        get(row, "Avg_Dur_Hours"),
        get(row, "Avg_Dur_Minutes"),
        get(row, "Avg_Dur_Seconds"),
    )
    for _, row in df.iterrows()
]

if data_tuples:
    execute_values(cursor, insert_query, data_tuples)
    conn.commit()
else:
    print("No rows to insert.")

cursor.close()
conn.close()

print("ETL complete: table recreated, emojis removed, CTR converted, real/surrogate Video_ID used, duplicates skipped (ON CONFLICT video_id).")
