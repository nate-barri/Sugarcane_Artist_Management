import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import re
import numpy as np

# --- DB connection parameters ---
db_params = {
    'dbname': 'test1',
    'user': 'postgres',
    'password': 'admin',
    'host': 'localhost',
    'port': '5432'
}

# --- Function to remove emojis ---
def remove_emoji(text):
    if not isinstance(text, str):
        return text
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002500-\U00002BEF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# --- Extract ---
df = pd.read_csv("YOUTUBE - SUGARCANE CONTENT DATA.csv")

# --- Transform ---
# Convert publish time to datetime
df["Video publish time"] = pd.to_datetime(df["Video publish time"], errors="coerce")

# Extract date parts
df["Publish_Day"] = df["Video publish time"].dt.day
df["Publish_Month"] = df["Video publish time"].dt.month
df["Publish_Year"] = df["Video publish time"].dt.year

# Drop redundant video_publish_time
df.drop(columns=["Video publish time"], inplace=True)

# Parse Average view duration into hours/minutes/seconds
def parse_duration(duration_str):
    if not isinstance(duration_str, str):
        return 0, 0, 0
    try:
        parts = [int(p) for p in duration_str.split(':')]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            return 0, parts[0], parts[1]
        elif len(parts) == 1:
            return 0, 0, parts[0]
        else:
            return 0, 0, 0
    except:
        return 0, 0, 0

df["Avg_Dur_Hours"], df["Avg_Dur_Minutes"], df["Avg_Dur_Seconds"] = zip(*df["Average view duration"].map(parse_duration))

# Drop redundant Average view duration
df.drop(columns=["Average view duration"], inplace=True)

# Remove emojis from all string columns
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].apply(remove_emoji)

# Replace NaT and NaN with None for SQL compatibility
df = df.replace({np.nan: None, pd.NaT: None})

# --- Load ---
conn = psycopg2.connect(**db_params)
cursor = conn.cursor()

# Create table without redundant columns
create_table_query = """
CREATE TABLE IF NOT EXISTS yt_video_etl (
    Content TEXT,
    Video_title TEXT,
    Publish_Day INT,
    Publish_Month INT,
    Publish_Year INT,
    Duration TEXT,
    Impressions BIGINT,
    Impressions_ctr FLOAT,
    Avg_views_per_viewer FLOAT,
    New_viewers BIGINT,
    Subscribers_gained BIGINT,
    Subscribers_lost BIGINT,
    Likes BIGINT,
    Shares BIGINT,
    Comments_added BIGINT,
    Views BIGINT,
    Watch_time_hours FLOAT,
    Avg_Dur_Hours INT,
    Avg_Dur_Minutes INT,
    Avg_Dur_Seconds INT
);
"""
cursor.execute(create_table_query)
conn.commit()

# Prepare insert query
insert_query = """
INSERT INTO yt_video_etl (
    Content, Video_title,
    Publish_Day, Publish_Month, Publish_Year,
    Duration, Impressions, Impressions_ctr,
    Avg_views_per_viewer, New_viewers,
    Subscribers_gained, Subscribers_lost, Likes, Shares,
    Comments_added, Views, Watch_time_hours,
    Avg_Dur_Hours, Avg_Dur_Minutes, Avg_Dur_Seconds
) VALUES %s
"""

# Prepare tuples for insert
data_tuples = [
    (
        row["Content"],
        row["Video title"],
        row["Publish_Day"],
        row["Publish_Month"],
        row["Publish_Year"],
        row["Duration"],
        row["Impressions"],
        row["Impressions click-through rate (%)"],
        row["Average views per viewer"],
        row["New viewers"],
        row["Subscribers gained"],
        row["Subscribers lost"],
        row["Likes"],
        row["Shares"],
        row["Comments added"],
        row["Views"],
        row["Watch time (hours)"],
        row["Avg_Dur_Hours"],
        row["Avg_Dur_Minutes"],
        row["Avg_Dur_Seconds"]
    )
    for _, row in df.iterrows()
]

execute_values(cursor, insert_query, data_tuples)
conn.commit()

cursor.close()
conn.close()

print("ETL process completed: removed redundant publish_time and average_view_duration columns.")
