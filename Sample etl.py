import psycopg2
import pandas as pd
import emoji
from datetime import datetime

# PostgreSQL Connection Config
POSTGRES_CONFIG = {
    "host": "localhost",
    "database": "capstone",
    "user": "postgres",
    "password": "admin",
    "port": "5432"
}

# Function to clean emojis from text
def remove_emojis(text):
    return emoji.replace_emoji(text, replace="") if isinstance(text, str) else text

# Function to normalize the publish time format
def normalize_publish_time(timestamp):
    try:
        return datetime.strptime(timestamp, "%m/%d/%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None  # Return None if the format is invalid

# Function to extract data
def extract_data(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.lower()

    # Mapping CSV column names to expected column names
    column_mapping = {
        "post id": "post_id",
        "page id": "page_id",
        "page name": "page_name",
        "title": "title",
        "description": "description",
        "duration (sec)": "duration_sec",
        "publish time": "publish_time",
        "caption type": "caption_type",
        "permalink": "permalink",
        "is crosspost": "is_crosspost",
        "is share": "is_share",
        "post type": "post_type",
        "languages": "languages",
        "custom labels": "custom_labels",
        "funded content status": "funded_content_status",
        "data comment": "data_comment",
        "date": "date",
        "impressions": "impressions",
        "reach": "reach",
        "reactions, comments and shares": "reactions_comments_shares",
        "reactions": "reactions",
        "comments": "comments",
        "shares": "shares",
        "total clicks": "total_clicks",
        "other clicks": "other_clicks",
        "matched audience targeting consumption (photo click)": "matched_audience_targeting",
        "link clicks": "link_clicks",
        "views": "views",
        "reels_plays:count": "reels_plays_count",
        "seconds viewed": "seconds_viewed",
        "average seconds viewed": "avg_seconds_viewed",
        "estimated earnings (usd)": "estimated_earnings"
    }

    df.rename(columns=column_mapping, inplace=True)

    # Apply transformations
    df["publish_time"] = df["publish_time"].apply(normalize_publish_time)
    df["title"] = df["title"].apply(remove_emojis)
    df["description"] = df["description"].apply(remove_emojis)
    df["is_crosspost"] = df["is_crosspost"].astype(bool, errors='ignore')
    df["is_share"] = df["is_share"].astype(bool, errors='ignore')
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    return df

# Function to load data into PostgreSQL
def load_data_to_postgres(df, table_name):
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()

        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            post_id SERIAL PRIMARY KEY,
            page_id VARCHAR(50),
            page_name VARCHAR(550),
            title VARCHAR(550),
            description VARCHAR(550),
            duration_sec INT,
            publish_time TIMESTAMP,
            caption_type VARCHAR(550),
            permalink VARCHAR(550),
            is_crosspost BOOLEAN,
            is_share BOOLEAN,
            post_type VARCHAR(550),
            languages VARCHAR(550),
            custom_labels VARCHAR(550),
            funded_content_status VARCHAR(550),
            data_comment VARCHAR(550),
            date DATE,
            impressions INT,
            reach INT,
            reactions_comments_shares INT,
            reactions INT,
            comments INT,
            shares INT,
            total_clicks INT,
            other_clicks INT,
            matched_audience_targeting INT,
            link_clicks INT,
            views INT,
            reels_plays_count INT,
            seconds_viewed INT,
            avg_seconds_viewed FLOAT,
            estimated_earnings FLOAT
        );
        """
        cursor.execute(create_table_query)
        conn.commit()

        insert_query = f"""
        INSERT INTO {table_name} (
            page_id, page_name, title, description, duration_sec, publish_time, caption_type,
            permalink, is_crosspost, is_share, post_type, languages, custom_labels, 
            funded_content_status, data_comment, date, impressions, reach, 
            reactions_comments_shares, reactions, comments, shares, total_clicks, 
            other_clicks, matched_audience_targeting, link_clicks, views, 
            reels_plays_count, seconds_viewed, avg_seconds_viewed, estimated_earnings
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """

        for _, row in df.iterrows():
            row_tuple = tuple(None if pd.isna(x) else x for x in row)
            cursor.execute(insert_query, row_tuple)

        conn.commit()
        cursor.close()
        conn.close()
        print(" Data successfully inserted into PostgreSQL!")

    except Exception as e:
        print(f" Error: {e}")

# Run ETL
def run_etl(csv_file, table_name):
    print(" Extracting data from CSV...")
    try:
        df = extract_data(csv_file)
        print(" Data extracted successfully! Preview:")
        print(df.info())
        print(df.head())
    except Exception as e:
        print(f" Failed to read CSV file: {e}")
        return

    if df.empty:
        print(" CSV file is empty or incorrectly formatted!")
        return

    print(" Loading data into PostgreSQL...")
    load_data_to_postgres(df, table_name)

# Example Usage
csv_file = "Jan-27-2025_Feb-23-2025_541651602373282 (1).csv"
table_name = "social_media_metrics"
run_etl(csv_file, table_name)
