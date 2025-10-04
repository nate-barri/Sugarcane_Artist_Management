#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, hashlib
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import emoji
from datetime import datetime, timezone

# ================= DB CONNECTION (Neon) =================
db_params = {
    'dbname':   'neondb',
    'user':     'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host':     'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port':     '5432',
    'sslmode':  'require'
}

# ================= CONFIG =================
CSV_PATH = r"TikTok/tiktok_data.csv"   # change to your local path if needed

# ================= HELPERS =================
def read_csv_robust(path):
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc, engine="python")
            print(f"📄 Loaded TikTok CSV with encoding: {enc}")
            return df
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(path, encoding="latin1", engine="python", sep=None)
    print("📄 Loaded TikTok CSV with fallback: latin1 + auto-delimiter")
    return df

def remove_emoji(text):
    if not isinstance(text, str): return text
    return emoji.replace_emoji(text, replace="").strip()

def to_int_or_none(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return None
        s = str(x).strip()
        if s == "": return None
        return int(float(s))
    except:
        return None

def parse_duration_to_hms(s):
    """
    Accepts 'hh:mm:ss', 'mm:ss', 'ss', or things like '15s', '1m30s'.
    Returns (h, m, s, total_seconds) with ints (None-safe).
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return (0, 0, 0, None)
    s = str(s).strip().lower()
    if s == "":
        return (0, 0, 0, None)

    # 1) hh:mm:ss or mm:ss or ss
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

    # 2) patterns like '1h2m3s', '90s', '2m'
    h = m = sec = 0
    mh = re.search(r"(\d+)\s*h", s)
    mm = re.search(r"(\d+)\s*m", s)
    ms = re.search(r"(\d+)\s*s", s)
    if mh or mm or ms:
        if mh: h = int(mh.group(1))
        if mm: m = int(mm.group(1))
        if ms: sec = int(ms.group(1))
        return (h, m, sec, h*3600 + m*60 + sec)

    # 3) plain seconds (number)
    if re.fullmatch(r"\d+(\.\d+)?", s):
        try:
            sec = int(float(s))
            return (0, 0, sec, sec)
        except:
            return (0, 0, 0, None)

    # fallback
    return (0, 0, 0, None)

def make_publish_time(y, m, d):
    y = to_int_or_none(y); m = to_int_or_none(m); d = to_int_or_none(d)
    if all(v is not None for v in (y, m, d)):
        try:
            return datetime(int(y), int(m), int(d), tzinfo=timezone.utc)
        except:
            return None
    return None

# ================= EXTRACT =================
df = read_csv_robust(CSV_PATH)
# trim headers and enforce exact known names
df.columns = [c.strip() for c in df.columns]

# Expected columns from you:
expected = {
    "tiktok_video_id", "content_link", "video_title",
    "publish_day", "publish_month", "publish_year",
    "duration", "likes", "shares", "comments_added",
    "views", "saves", "sound_used", "post_type"
}
missing = [c for c in expected if c not in set(df.columns)]
if missing:
    print(f"⚠️ Missing expected columns: {missing} — continuing, but those will be NULL.")

# ================= TRANSFORM =================
work = pd.DataFrame()
work["video_id"]   = df.get("tiktok_video_id")
work["url"]        = df.get("content_link")
work["title"]      = df.get("video_title")
work["post_type"]  = df.get("post_type")
work["sound_used"] = df.get("sound_used")
work["saves"]      = df.get("saves")

# numeric counts
for src, dst in [
    ("likes","likes"), ("shares","shares"),
    ("comments_added","comments_added"), ("views","views"),
    ("saves","saves")
]:
    if src in df.columns:
        work[dst] = df[src].apply(to_int_or_none)
    else:
        work[dst] = None

# duration → h/m/s/total (FIXED)
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

# publish parts (normalize to ints) + publish_time (UTC midnight)
work["publish_year"]  = df.get("publish_year").apply(to_int_or_none) if "publish_year" in df.columns else None
work["publish_month"] = df.get("publish_month").apply(to_int_or_none) if "publish_month" in df.columns else None
work["publish_day"]   = df.get("publish_day").apply(to_int_or_none) if "publish_day" in df.columns else None
work["publish_time"]  = [
    make_publish_time(y, m, d) for y, m, d in zip(work["publish_year"], work["publish_month"], work["publish_day"])
]

# text cleanup
for txt_col in ["title","url","sound_used","post_type"]:
    if txt_col in work.columns:
        work[txt_col] = work[txt_col].apply(remove_emoji)

# Replace pandas NA with None for Postgres
work = work.replace({np.nan: None, pd.NaT: None})

# Basic sanity: drop rows without a usable id
rows_before = len(work)
work = work[work["video_id"].notna() & (work["video_id"].astype(str).str.strip() != "")]
rows_after = len(work)
if rows_after < rows_before:
    print(f"⚠️ Dropped {rows_before-rows_after} rows with empty tiktok_video_id.")

print(f"✅ Ready to insert {len(work)} rows.")

# ================= LOAD (landing) =================
conn = psycopg2.connect(**db_params)
cur = conn.cursor()

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

records = [
    (
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
    )
    for _, r in work.iterrows()
]

if records:
    execute_values(cur, insert_sql, records, page_size=1000)
    conn.commit()
    print(f"✅ Inserted/Upserted {len(records)} rows into public.tt_video_etl.")
else:
    print("⚠️ No rows to insert (no valid video_id).")

cur.close()
conn.close()
print("🎉 TikTok ETL complete.")