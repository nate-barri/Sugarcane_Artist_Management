import pandas as pd
import psycopg2
from datetime import datetime
import os

# ================= DB CONNECTION =================
db_params = {
    "dbname":   "neondb",
    "user":     "neondb_owner",
    "password": "npg_dGzvq4CJPRx7",
    "host":     "ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech",
    "port":     "5432",
    "sslmode":  "require",
}

def execute_query(conn, query):
    return pd.read_sql_query(query, conn)

def safe_divide(a, b):
    return a / b if b != 0 else 0

# ================= QUERY FUNCTIONS =================
def get_overview_metrics(conn):
    query = """
    SELECT 
        COUNT(*) as total_videos,
        SUM(views) as total_views,
        SUM(likes) as total_likes,
        SUM(shares) as total_shares,
        SUM(comments_added) as total_comments,
        SUM(saves) as total_saves,
        AVG(views) as avg_views,
        AVG(likes) as avg_likes,
        AVG(shares) as avg_shares,
        AVG(comments_added) as avg_comments,
        AVG(saves) as avg_saves,
        AVG(duration_sec) as avg_duration_seconds,
        COUNT(DISTINCT CONCAT(publish_year, '-', publish_month)) as total_months
    FROM public.tt_video_etl
    WHERE views IS NOT NULL;
    """
    df = execute_query(conn, query)

    total_views    = df["total_views"].iloc[0] or 0
    total_likes    = df["total_likes"].iloc[0] or 0
    total_shares   = df["total_shares"].iloc[0] or 0
    total_comments = df["total_comments"].iloc[0] or 0
    total_saves    = df["total_saves"].iloc[0] or 0
    total_videos   = df["total_videos"].iloc[0] or 0
    total_months   = df["total_months"].iloc[0] or 1

    df["engagement_rate"] = safe_divide(
        (total_likes + total_shares + total_comments + total_saves), total_views) * 100
    df["like_rate"]    = safe_divide(total_likes, total_views) * 100
    df["share_rate"]   = safe_divide(total_shares, total_views) * 100
    df["comment_rate"] = safe_divide(total_comments, total_views) * 100
    df["save_rate"]    = safe_divide(total_saves, total_views) * 100
    df["avg_videos_per_month"] = safe_divide(total_videos, total_months)

    return df

def get_top_performing_videos(conn, limit=50):
    query_views = f"""
    SELECT video_id, title, views, likes, shares, comments_added, saves, publish_time, url
    FROM public.tt_video_etl
    WHERE views IS NOT NULL
    ORDER BY views DESC
    LIMIT {limit};
    """
    query_engagement = f"""
    SELECT video_id, title, views, likes, shares, comments_added, saves,
           (COALESCE(likes,0) + COALESCE(shares,0) + COALESCE(comments_added,0) + COALESCE(saves,0)) as total_engagement,
           publish_time, url
    FROM public.tt_video_etl
    WHERE views IS NOT NULL
    ORDER BY total_engagement DESC
    LIMIT {limit};
    """
    query_engagement_rate = f"""
    SELECT video_id, title, views, likes, shares, comments_added, saves,
           CASE WHEN views > 0 THEN 
               ((COALESCE(likes,0) + COALESCE(shares,0) + COALESCE(comments_added,0) + COALESCE(saves,0))::FLOAT / views) * 100
           ELSE 0 END as engagement_rate,
           publish_time, url
    FROM public.tt_video_etl
    WHERE views > 0
    ORDER BY engagement_rate DESC
    LIMIT {limit};
    """
    return {
        "by_views": execute_query(conn, query_views),
        "by_engagement": execute_query(conn, query_engagement),
        "by_engagement_rate": execute_query(conn, query_engagement_rate),
    }

def get_temporal_analysis(conn):
    query_monthly = """
    SELECT publish_year, publish_month, COUNT(*) as video_count,
           SUM(views) as total_views, SUM(likes) as total_likes,
           SUM(shares) as total_shares, SUM(comments_added) as total_comments,
           AVG(views) as avg_views, AVG(likes) as avg_likes
    FROM public.tt_video_etl
    WHERE publish_year IS NOT NULL AND publish_month IS NOT NULL AND views IS NOT NULL
    GROUP BY publish_year, publish_month
    ORDER BY publish_year ASC, publish_month ASC;
    """
    query_dow = """
    SELECT EXTRACT(DOW FROM publish_time) as day_of_week,
           CASE EXTRACT(DOW FROM publish_time)
               WHEN 0 THEN 'Sunday' WHEN 1 THEN 'Monday' WHEN 2 THEN 'Tuesday'
               WHEN 3 THEN 'Wednesday' WHEN 4 THEN 'Thursday' WHEN 5 THEN 'Friday'
               WHEN 6 THEN 'Saturday'
           END as day_name,
           COUNT(*) as video_count, AVG(views) as avg_views, AVG(likes) as avg_likes,
           AVG(COALESCE(likes,0) + COALESCE(shares,0) + COALESCE(comments_added,0) + COALESCE(saves,0)) as avg_engagement
    FROM public.tt_video_etl
    WHERE publish_time IS NOT NULL AND views IS NOT NULL
    GROUP BY EXTRACT(DOW FROM publish_time)
    ORDER BY EXTRACT(DOW FROM publish_time);
    """
    query_yearly = """
    SELECT 
        publish_year,
        publish_month,
        COUNT(*) as video_count
    FROM public.tt_video_etl
    WHERE publish_year IS NOT NULL AND publish_month IS NOT NULL AND views IS NOT NULL
    GROUP BY publish_year, publish_month
    ORDER BY publish_year, publish_month;
    """
    return {
        "monthly": execute_query(conn, query_monthly),
        "day_of_week": execute_query(conn, query_dow),
        "yearly": execute_query(conn, query_yearly),
    }

def get_content_analysis(conn):
    query_post_type = """
    SELECT post_type, COUNT(*) as video_count,
           AVG(views) as avg_views, AVG(likes) as avg_likes,
           AVG(shares) as avg_shares, AVG(comments_added) as avg_comments,
           SUM(views) as total_views
    FROM public.tt_video_etl
    WHERE post_type IS NOT NULL AND views IS NOT NULL
    GROUP BY post_type
    ORDER BY total_views DESC;
    """
    query_duration = """
    WITH duration_buckets AS (
        SELECT 
            CASE 
                WHEN duration_sec <= 15 THEN '0-15s'
                WHEN duration_sec <= 30 THEN '16-30s'
                WHEN duration_sec <= 60 THEN '31-60s'
                WHEN duration_sec <= 120 THEN '61-120s'
                ELSE '120s+'
            END as duration_bucket,
            CASE 
                WHEN duration_sec <= 15 THEN 1
                WHEN duration_sec <= 30 THEN 2
                WHEN duration_sec <= 60 THEN 3
                WHEN duration_sec <= 120 THEN 4
                ELSE 5
            END as sort_order,
            views, likes, shares, comments_added, saves
        FROM public.tt_video_etl
        WHERE duration_sec IS NOT NULL AND views IS NOT NULL
    )
    SELECT duration_bucket, COUNT(*) as video_count,
           AVG(views) as avg_views, AVG(likes) as avg_likes,
           AVG(COALESCE(likes,0) + COALESCE(shares,0) + COALESCE(comments_added,0) + COALESCE(saves,0)) as avg_engagement
    FROM duration_buckets
    GROUP BY duration_bucket, sort_order
    ORDER BY sort_order;
    """
    query_sound = """
    SELECT 
        CASE 
            WHEN sound_used IS NULL OR sound_used = '' OR TRIM(sound_used) = '' THEN 'No Sound'
            ELSE TRIM(sound_used)
        END as sound_category,
        COUNT(*) as video_count, 
        SUM(views) as total_views,
        AVG(views) as avg_views, 
        AVG(likes) as avg_likes
    FROM public.tt_video_etl
    WHERE views IS NOT NULL
        AND LOWER(TRIM(sound_used)) NOT IN ('sunet original', 'please', 'apt. - rose')
    GROUP BY sound_category
    ORDER BY total_views DESC
    LIMIT 20;
    """
    return {
        "post_type": execute_query(conn, query_post_type),
        "duration": execute_query(conn, query_duration),
        "sound": execute_query(conn, query_sound),
    }

def get_engagement_distribution(conn):
    query = """
    SELECT 
        views,
        likes,
        shares,
        comments_added,
        saves,
        CASE WHEN views > 0 THEN (likes::FLOAT / views) * 100 ELSE 0 END AS like_rate,
        CASE WHEN views > 0 THEN (shares::FLOAT / views) * 100 ELSE 0 END AS share_rate,
        CASE WHEN views > 0 THEN (comments_added::FLOAT / views) * 100 ELSE 0 END AS comment_rate,
        CASE WHEN views > 0 THEN 
            ((COALESCE(likes,0) + COALESCE(shares,0) + COALESCE(comments_added,0) + COALESCE(saves,0))::FLOAT / views) * 100
        ELSE 0 END AS engagement_rate
    FROM public.tt_video_etl
    WHERE views > 0;
    """
    return execute_query(conn, query)

# ================= MAIN EXECUTION =================
def export_analytics_to_csv():
    print("Starting TikTok Analytics CSV Export...\n")
    
    # Define the target directory path
    base_path = r"C:\Users\Luis\Desktop\COLLEGE\CAPSTONE\Sugarcane_Artist_Management\public"
    output_dir = os.path.join(base_path, "tiktok_analytics_export")
    
    # Create directory if it doesn't exist
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ Created output directory: {output_dir}\n")
        else:
            print(f"✓ Using existing directory: {output_dir} (files will be overwritten)\n")
    except Exception as e:
        print(f"❌ Error creating directory: {str(e)}")
        print(f"   Make sure the path exists: {base_path}")
        return
    
    max_retries, retry_delay, conn = 3, 2, None

    for attempt in range(max_retries):
        try:
            print(f"Attempting database connection (attempt {attempt+1}/{max_retries})...")
            conn = psycopg2.connect(**db_params)
            print("✓ Database connection successful!\n")
            break
        except psycopg2.OperationalError as e:
            print(f"✗ Connection attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                import time
                print(f"  Retrying in {retry_delay} seconds...\n")
                time.sleep(retry_delay)
            else:
                print("\n❌ Failed to connect to database after multiple attempts.")
                return
    
    if conn is None:
        print("❌ Could not establish database connection.")
        return

    try:
        print("Fetching and exporting data...\n")
        
        # 1. Overview Metrics
        print("1. Exporting overview metrics...")
        overview = get_overview_metrics(conn)
        overview.to_csv(os.path.join(output_dir, "01_overview_metrics.csv"), index=False)
        print("   ✓ Saved: 01_overview_metrics.csv")
        
        # 2. Top Performing Videos
        print("\n2. Exporting top performing videos...")
        top_videos = get_top_performing_videos(conn, limit=50)
        top_videos["by_views"].to_csv(os.path.join(output_dir, "02_top_videos_by_views.csv"), index=False)
        top_videos["by_engagement"].to_csv(os.path.join(output_dir, "03_top_videos_by_engagement.csv"), index=False)
        top_videos["by_engagement_rate"].to_csv(os.path.join(output_dir, "04_top_videos_by_engagement_rate.csv"), index=False)
        print("   ✓ Saved: 02_top_videos_by_views.csv")
        print("   ✓ Saved: 03_top_videos_by_engagement.csv")
        print("   ✓ Saved: 04_top_videos_by_engagement_rate.csv")
        
        # 3. Temporal Analysis
        print("\n3. Exporting temporal analysis...")
        temporal = get_temporal_analysis(conn)
        temporal["monthly"].to_csv(os.path.join(output_dir, "05_monthly_stats.csv"), index=False)
        temporal["day_of_week"].to_csv(os.path.join(output_dir, "06_day_of_week_stats.csv"), index=False)
        temporal["yearly"].to_csv(os.path.join(output_dir, "07_yearly_monthly_breakdown.csv"), index=False)
        print("   ✓ Saved: 05_monthly_stats.csv")
        print("   ✓ Saved: 06_day_of_week_stats.csv")
        print("   ✓ Saved: 07_yearly_monthly_breakdown.csv")
        
        # 4. Content Analysis
        print("\n4. Exporting content analysis...")
        content = get_content_analysis(conn)
        if not content["post_type"].empty:
            content["post_type"].to_csv(os.path.join(output_dir, "08_post_type_analysis.csv"), index=False)
            print("   ✓ Saved: 08_post_type_analysis.csv")
        if not content["duration"].empty:
            content["duration"].to_csv(os.path.join(output_dir, "09_duration_analysis.csv"), index=False)
            print("   ✓ Saved: 09_duration_analysis.csv")
        if not content["sound"].empty:
            content["sound"].to_csv(os.path.join(output_dir, "10_sound_analysis.csv"), index=False)
            print("   ✓ Saved: 10_sound_analysis.csv")
        
        # 5. Engagement Distribution
        print("\n5. Exporting engagement distribution...")
        engagement_dist = get_engagement_distribution(conn)
        engagement_dist.to_csv(os.path.join(output_dir, "11_engagement_distribution.csv"), index=False)
        print("   ✓ Saved: 11_engagement_distribution.csv")
        
        # 6. Export full dataset
        print("\n6. Exporting full dataset...")
        full_query = """
        SELECT * FROM public.tt_video_etl
        WHERE views IS NOT NULL
        ORDER BY publish_time DESC;
        """
        full_data = execute_query(conn, full_query)
        full_data.to_csv(os.path.join(output_dir, "12_full_dataset.csv"), index=False)
        print("   ✓ Saved: 12_full_dataset.csv")
        
        # Create a summary file with last export time
        print("\n7. Creating export summary...")
        summary = {
            "Last Export Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Total Files": [13],
            "Total Videos": [overview["total_videos"].iloc[0]],
            "Total Views": [overview["total_views"].iloc[0]],
            "Total Likes": [overview["total_likes"].iloc[0]],
            "Output Directory": [output_dir]
        }
        pd.DataFrame(summary).to_csv(os.path.join(output_dir, "00_export_summary.csv"), index=False)
        print("   ✓ Saved: 00_export_summary.csv")
        
        print(f"\n{'='*60}")
        print("✅ All data exported successfully!")
        print(f"{'='*60}")
        print(f"\nLocation: {output_dir}")
        print(f"Total files: 13 CSV files")
        print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nFiles exported:")
        print("  - Overview metrics")
        print("  - Top performing videos (3 files)")
        print("  - Temporal analysis (3 files)")
        print("  - Content analysis (3 files)")
        print("  - Engagement distribution")
        print("  - Full dataset")
        print("  - Export summary")
        
    except Exception as e:
        print(f"\n❌ Error during export: {str(e)}")
        raise
    finally:
        conn.close()
        print("\n✓ Database connection closed.")

if __name__ == "__main__":
    export_analytics_to_csv()