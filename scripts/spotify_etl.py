import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime
import emoji
import sys
import os

if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    import io
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ================= DB CONNECTION =================
db_params = {
    'dbname': os.environ.get('PGDATABASE', 'neondb'),
    'user': os.environ.get('PGUSER', 'neondb_owner'),
    'password': os.environ.get('PGPASSWORD'),
    'host': os.environ.get('PGHOST'),
    'port': os.environ.get('PGPORT', '5432'),
    'sslmode': 'require'
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

def ensure_platform_exists():
    """Ensure 'spotify' platform exists in dw.dim_platform table, create if missing"""
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        # Check if dim_platform table exists, create if missing
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'dw' AND table_name = 'dim_platform'
            )
        """)
        
        if not cur.fetchone()[0]:
            print("[WARNING] Table 'dw.dim_platform' does not exist. Creating it...")
            
            cur.execute("""
            CREATE TABLE IF NOT EXISTS dw.dim_platform (
                platform_sk SERIAL PRIMARY KEY,
                platform_code VARCHAR(50) UNIQUE NOT NULL,
                platform_name VARCHAR(100) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.commit()
        
        # Check if 'spotify' platform exists, insert if missing
        cur.execute("SELECT platform_sk FROM dw.dim_platform WHERE platform_code = 'spotify'")
        result = cur.fetchone()
        
        if not result:
            print("[WARNING] Platform 'spotify' not found. Creating it automatically...")
            
            cur.execute("""
            INSERT INTO dw.dim_platform (platform_code, platform_name)
            VALUES ('spotify', 'Spotify')
            ON CONFLICT (platform_code) DO NOTHING
            RETURNING platform_sk
            """)
            
            result = cur.fetchone()
            conn.commit()
            print(f"[SUCCESS] Platform 'spotify' created with platform_sk={result[0]}")
        else:
            print(f"[SUCCESS] Platform 'spotify' found with platform_sk={result[0]}")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"[ERROR] Error ensuring platform exists: {e}")
        return False

# ================= EXTRACT =================
if len(sys.argv) < 2:
    print("[ERROR] Usage: python spotify_etl.py <csv_file_path>", file=sys.stderr)
    sys.exit(1)

csv_file = sys.argv[1]

print("Reading Spotify CSV file...")
try:
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} records from Spotify data")
except Exception as e:
    print(f"[ERROR] Failed to read CSV: {e}", file=sys.stderr)
    sys.exit(1)

# ================= TRANSFORM =================
print("Cleaning and normalizing data...")
df = clean_dataframe(df)
print("Data cleaned successfully")

is_song_data = 'song' in df.columns and 'release_date' in df.columns
is_stats_data = 'date' in df.columns and 'listeners' in df.columns and 'followers' in df.columns

# ================= LOAD =================
print("Connecting to Neon PostgreSQL...")

if not ensure_platform_exists():
    print("[ERROR] Cannot proceed with data insertion. Please fix the platform configuration first.")
    sys.exit(1)

try:
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    
    # ---- ENABLE UUID EXTENSION ----
    cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    
    # Check if spotify_songs table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'spotify_songs'
        )
    """)
    
    if not cur.fetchone()[0]:
        print("Creating spotify_songs table...")
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
    
    # Check if spotify_stats table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'spotify_stats'
        )
    """)
    
    if not cur.fetchone()[0]:
        print("Creating spotify_stats table...")
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
    
    # ---- LOAD SONG DATA ----
    songs_data = []
    if is_song_data:
        print("Processing song-level data...")
        songs_data = [
            (
                row.get('song'),
                row.get('listeners'),
                row.get('streams'),
                row.get('saves'),
                row.get('release_date')
            )
            for _, row in df.iterrows()
            if row.get('song') and str(row.get('song')).strip() not in ['', 'nan']
        ]
        
        if songs_data:
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
            print(f"Songs loaded/updated ({len(songs_data)} rows)")
    
    # ---- ENSURE PLACEHOLDER SONG EXISTS ----
    cur.execute("""
        INSERT INTO spotify_songs (song, listeners, streams, saves, release_date)
        VALUES ('All Songs', 0, 0, 0, NULL)
        ON CONFLICT (song) DO NOTHING;
    """)
    conn.commit()
    
    # ---- LOAD STATS DATA ----
    stats_data = []
    if is_stats_data:
        print("Processing time-series stats data...")
        stats_data = [
            (
                'All Songs',
                row.get('date'),
                row.get('listeners'),
                row.get('streams'),
                row.get('followers')
            )
            for _, row in df.iterrows()
            if row.get('date') and str(row.get('date')).strip() not in ['NaT', 'nan', '']
        ]
        
        if stats_data:
            execute_batch(cur, """
                INSERT INTO spotify_stats (song, date, listeners, streams, followers)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (song, date) DO UPDATE
                SET listeners = EXCLUDED.listeners,
                    streams = EXCLUDED.streams,
                    followers = EXCLUDED.followers;
            """, stats_data, page_size=100)
            conn.commit()
            print(f"Time-series stats loaded/updated ({len(stats_data)} rows)")
    
    cur.close()
    conn.close()
    
    total_records = len(songs_data) + len(stats_data)
    print("\nSpotify ETL completed successfully!")
    if songs_data:
        print(f"{len(songs_data)} songs processed")
    if stats_data:
        print(f"{len(stats_data)} time-series records processed")
    print("Future CSV uploads will auto-update existing records.")
    print(f"RECORDS: {total_records}")
    
except Exception as e:
    print(f"[ERROR] Database error: {e}", file=sys.stderr)
    sys.exit(1)
