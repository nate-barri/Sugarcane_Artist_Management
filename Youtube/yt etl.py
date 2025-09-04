import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# --- Database connection parameters ---
db_params = {
    'dbname': 'test1',
    'user': 'postgres',
    'password': 'admin',
    'host': 'localhost',
    'port': '5432'
}

# --- Step 1: Extract ---
csv_file = "Table data.csv"
df = pd.read_csv(csv_file)

# --- Step 2: Transform ---

# Drop the summary row and rows missing essential columns
df = df.iloc[1:]  # remove the "Total" row
df = df.dropna(subset=['Content', 'Video title'])

# Parse publish time and drop invalid dates
df['Video publish time'] = pd.to_datetime(df['Video publish time'], errors='coerce')
df = df.dropna(subset=['Video publish time'])

# Convert and clean numeric fields
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce').fillna(0).astype(int)
df['Views'] = pd.to_numeric(df['Views'], errors='coerce').fillna(0).astype(int)
df['Watch time (hours)'] = pd.to_numeric(df['Watch time (hours)'], errors='coerce').fillna(0)
df['Unique viewers'] = pd.to_numeric(df['Unique viewers'], errors='coerce').fillna(0).astype(int)

# Optional: clean average view duration as text (or convert to seconds if desired)
df['Average view duration'] = df['Average view duration'].fillna('00:00')

# --- Step 3: Load into PostgreSQL ---

# Define SQL schema
create_table_query = """
CREATE TABLE IF NOT EXISTS youtube_videos (
    content_id TEXT PRIMARY KEY,
    video_title TEXT,
    publish_time TIMESTAMP,
    duration_seconds INTEGER,
    views BIGINT,
    watch_time_hours DOUBLE PRECISION,
    average_view_duration TEXT,
    unique_viewers BIGINT
);
"""

# Connect and insert into database
with psycopg2.connect(**db_params) as conn:
    with conn.cursor() as cur:
        # Create table if not exists
        cur.execute(create_table_query)

        # Convert DataFrame to list of tuples
        rows = list(df[['Content', 'Video title', 'Video publish time', 'Duration', 'Views',
                        'Watch time (hours)', 'Average view duration', 'Unique viewers']].itertuples(index=False, name=None))

        # Bulk insert query with conflict handling
        insert_query = """
            INSERT INTO youtube_videos (
                content_id, video_title, publish_time, duration_seconds,
                views, watch_time_hours, average_view_duration, unique_viewers
            ) VALUES %s
            ON CONFLICT (content_id) DO UPDATE SET
                video_title = EXCLUDED.video_title,
                publish_time = EXCLUDED.publish_time,
                duration_seconds = EXCLUDED.duration_seconds,
                views = EXCLUDED.views,
                watch_time_hours = EXCLUDED.watch_time_hours,
                average_view_duration = EXCLUDED.average_view_duration,
                unique_viewers = EXCLUDED.unique_viewers;
        """
        execute_values(cur, insert_query, rows)

print("✅ ETL process completed successfully.")
