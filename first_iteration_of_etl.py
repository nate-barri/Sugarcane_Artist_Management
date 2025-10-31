import psycopg2
import csv
import emoji
from datetime import datetime

# Database connection parameters
DB_HOST = 'localhost'
DB_NAME = 'test1'
DB_USER = 'postgres'
DB_PASSWORD = 'admin'
DB_PORT = '5432'

# Function to connect to PostgreSQL
def connect_to_db():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

# Function to remove emojis from text
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='') if text else text

# Function to clean and convert data
def clean_data(row):
    try:
        post_id = int(float(row[0]))  # Convert float-like string to int
        page_id = int(float(row[1]))  # Convert scientific notation to int
        page_name = remove_emojis(row[2])
        title = remove_emojis(row[3])
        description = remove_emojis(row[4])
        duration = int(row[5]) if row[5] else None

        # Normalize publish_time
        if row[6]:
            dt = datetime.strptime(row[6], "%m/%d/%Y %H:%M")
            publish_year = dt.year
            publish_month = dt.month
            publish_day = dt.day
            publish_time = dt.time()
        else:
            publish_year = publish_month = publish_day = None
            publish_time = None

        caption_type = remove_emojis(row[7])
        permalink = row[8]
        is_crosspost = bool(int(row[9])) if row[9] else False
        is_share = bool(int(row[10])) if row[10] else False
        post_type = remove_emojis(row[11])
        languages = remove_emojis(row[12]) if row[12] else None
        custom_labels = remove_emojis(row[13]) if row[13] else None
        funded_content_status = remove_emojis(row[14]) if row[14] else None
        data_comment = remove_emojis(row[15]) if row[15] else None

        # Ensure 'date' is stored as a string
        date = row[16].strip() if row[16].strip().lower() not in ["lifetime", "n/a", "", " "] else None

        impressions = int(row[17]) if row[17] else 0
        reach = int(row[18]) if row[18] else 0
        reactions_comments_shares = int(row[19]) if row[19] else 0
        reactions = int(row[20]) if row[20] else 0
        comments = int(row[21]) if row[21] else 0
        shares = int(row[22]) if row[22] else 0
        total_clicks = int(row[23]) if row[23] else 0
        other_clicks = int(float(row[24])) if row[24] else 0
        matched_audience_targeting_consumption = int(float(row[25])) if row[25] else 0
        link_clicks = int(float(row[26])) if row[26] else 0
        views = int(float(row[27])) if row[27] else 0
        reels_plays_count = int(float(row[28])) if row[28] else 0
        seconds_viewed = int(float(row[29])) if row[29] else 0
        avg_seconds_viewed = float(row[30]) if row[30] else 0.0
        estimated_earnings = float(row[31]) if row[31] else 0.0

        return (
            post_id, page_id, page_name, title, description, duration,
            publish_year, publish_month, publish_day, publish_time, 
            caption_type, permalink, is_crosspost, is_share, post_type, 
            languages, custom_labels, funded_content_status, data_comment, 
            date, impressions, reach, reactions_comments_shares, reactions, 
            comments, shares, total_clicks, other_clicks, 
            matched_audience_targeting_consumption, link_clicks, views, 
            reels_plays_count, seconds_viewed, avg_seconds_viewed, 
            estimated_earnings
        )
    except (ValueError, IndexError) as e:
        print(f"Skipping row {row} due to error: {e}")
        return None


# Main function to ingest data into the cleaned table
def ingest_data(file_path):
    conn = connect_to_db()
    cur = conn.cursor()
    
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        
        for row in reader:
            cleaned_row = clean_data(row)
            if cleaned_row:
                try:
                    cur.execute("""
                        INSERT INTO analytics_cleaned (
                            post_id, page_id, page_name, title, description, duration, 
                            publish_year, publish_month, publish_day, publish_time, 
                            caption_type, permalink, is_crosspost, is_share, post_type, 
                            languages, custom_labels, funded_content_status, data_comment, 
                            date, impressions, reach, reactions_comments_shares, reactions, 
                            comments, shares, total_clicks, other_clicks, 
                            matched_audience_targeting_consumption, link_clicks, views, 
                            reels_plays_count, seconds_viewed, avg_seconds_viewed, 
                            estimated_earnings
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, cleaned_row)

                except psycopg2.Error as e:
                    print(f"Database error for row {row}: {e}")
                    conn.rollback()  # Rollback transaction on error

    conn.commit()
    cur.close()
    conn.close()
    print("Cleaned data ingestion complete.")

if __name__ == "__main__":
    ingest_data("Jan-01-2025_Mar-18-2025_662977099719234.csv"),
    ingest_data("Oct-01-2024_Dec-31-2024_1396476811341844.csv")
