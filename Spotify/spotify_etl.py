import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime
import emoji

# ================= DB CONNECTION =================
db_params = {
    'dbname':   'neondb',
    'user':     'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host':     'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port':     '5432',
    'sslmode':  'require'
}

# ================= HELPER FUNCTIONS =================
def remove_emojis(text):
    if pd.isna(text):
        return text
    return emoji.replace_emoji(str(text), replace='')

def normalize_date(value):
    """Convert multiple date formats to YYYY-MM-DD."""
    if pd.isna(value):
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None

def clean_dataframe(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(remove_emojis)
    if 'date' in df.columns:
        df['date'] = df['date'].astype(str).apply(normalize_date)
    if 'release_date' in df.columns:
        df['release_date'] = df['release_date'].astype(str).apply(normalize_date)
    return df

# ================= EXTRACT =================
songs_path = r"C:\Sugarcane_Artist_Management\Spotify\Sugarcane-songs-all.csv"
stats_path = r"C:\Sugarcane_Artist_Management\Spotify\2023-2025.csv"

print("→ Reading Spotify CSV files...")
songs_df = pd.read_csv(songs_path)
stats_df = pd.read_csv(stats_path)
print(f"✓ Loaded {len(songs_df)} songs and {len(stats_df)} stats records")

# ================= TRANSFORM =================
print("→ Cleaning and normalizing data...")
songs_df = clean_dataframe(songs_df)
stats_df = clean_dataframe(stats_df)
print("✓ Data cleaned successfully")

# ================= LOAD =================
print("→ Connecting to Neon PostgreSQL...")
conn = psycopg2.connect(**db_params)
cur = conn.cursor()

# ---- ENABLE UUID EXTENSION ----
cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

# ---- CREATE TABLES IF NOT EXISTS ----
cur.execute("""
CREATE TABLE IF NOT EXISTS spotify_songs (
    song_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    song TEXT UNIQUE NOT NULL,
    listeners BIGINT,
    streams BIGINT,
    saves BIGINT,
    release_date DATE
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS spotify_stats (
    stat_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    song TEXT REFERENCES spotify_songs(song) ON DELETE CASCADE,
    date DATE NOT NULL,
    listeners BIGINT,
    streams BIGINT,
    followers BIGINT,
    UNIQUE (song, date)
);
""")
conn.commit()

# ---- UPSERT INTO spotify_songs ----
print("→ Inserting/updating songs...")
songs_data = [
    (
        row.get('song'),
        row.get('listeners'),
        row.get('streams'),
        row.get('saves'),
        row.get('release_date')
    )
    for _, row in songs_df.iterrows()
    if row.get('song') and str(row.get('song')).strip() not in ['', 'nan']
]

execute_batch(cur, """
    INSERT INTO spotify_songs (song, listeners, streams, saves, release_date)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (song) DO UPDATE
    SET listeners = EXCLUDED.listeners,
        streams = EXCLUDED.streams,
        saves = EXCLUDED.saves,
        release_date = EXCLUDED.release_date;
""", songs_data, page_size=100)
conn.commit()
print(f"✓ Songs loaded/updated ({len(songs_data)} rows)")

# ---- ENSURE PLACEHOLDER SONG EXISTS ----
cur.execute("""
    INSERT INTO spotify_songs (song, listeners, streams, saves, release_date)
    VALUES ('All Songs', 0, 0, 0, NULL)
    ON CONFLICT (song) DO NOTHING;
""")
conn.commit()

# ---- UPSERT INTO spotify_stats ----
print("→ Inserting/updating global stats...")
stats_data = [
    (
        'All Songs',
        row.get('date'),
        row.get('listeners'),
        row.get('streams'),
        row.get('followers')
    )
    for _, row in stats_df.iterrows()
    if row.get('date') and str(row.get('date')).strip() not in ['NaT', 'nan', '']
]

execute_batch(cur, """
    INSERT INTO spotify_stats (song, date, listeners, streams, followers)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (song, date) DO UPDATE
    SET listeners = EXCLUDED.listeners,
        streams = EXCLUDED.streams,
        followers = EXCLUDED.followers;
""", stats_data, page_size=100)
conn.commit()

cur.close()
conn.close()

print("\n✅ Spotify ETL completed successfully!")
print(f"✓ {len(songs_data)} songs processed")
print(f"✓ {len(stats_data)} global stats processed")
print("✓ Future CSV uploads will auto-update existing records.")
