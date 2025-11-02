
import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

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

# ================= VISUALIZATION FUNCTIONS =================
def _annotate_bars(ax, values, fmt="{:,.0f}"):
    for i, v in enumerate(values):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom")

def plot_overview_metrics(overview_df):
    """Each panel split into its own figure (no subplots)."""

# Data
    metrics = ["total_views", "total_likes", "total_shares", "total_comments", "total_saves"]
    labels = ["Views", "Likes", "Shares", "Comments", "Saves"]
    values = [overview_df[m].iloc[0] for m in metrics]

    # Create figure
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Draw bars with softer color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = ax.bar(labels, values, color=colors)

    # Title
    plt.ticklabel_format(style='plain', axis='y')
    ax.set_title("Total Engagement Metrics", fontsize=18, fontweight="bold", pad=15)

    ax.set_ylabel("Count")
    ax.set_xlabel("Engagements")

    ax.tick_params(axis="x", rotation=25)

    # Add gridlines
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate bars 
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(values) * 0.01),
                f"{val:,.0f}",
                ha="center", va="bottom",
                fontsize=11, fontweight="semibold")

    # Add total annotation
    total = sum(values)
    ax.text(0.95, 0.9, f"Total Engagements: {total:,.0f}",
            transform=ax.transAxes, fontsize=11,
            color="gray", ha="right", style="italic")

    ax.legend(bars, labels, title="Metrics", loc="center left",
            bbox_to_anchor=(1.02, 0.85), frameon=True)

    plt.tight_layout()
    plt.show()

    # --- Average metrics per video ---
    avg_metrics = ["avg_views", "avg_likes", "avg_shares", "avg_comments", "avg_saves"]
    avg_values  = [overview_df[m].iloc[0] for m in avg_metrics]
    labels = ["Views", "Likes", "Shares", "Comments", "Saves"]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    bars = ax.bar(labels, avg_values, color=colors)
    ax.set_title("Average Engagement per Video", fontweight="bold")
    ax.set_ylabel("Average Count")
    ax.set_xlabel("Engagements")
    ax.tick_params(axis="x", rotation=45)

    for bar, value in zip(bars, avg_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_values)*0.01,
                f"{value:,.0f}", ha="center", va="bottom", fontsize=9)

    ax.legend(bars, labels, title="Metrics", loc="center left",
            bbox_to_anchor=(1.02, 0.85), frameon=True)

    plt.tight_layout()
    plt.show()

    # --- Engagement rates ---
    rate_labels = ["Like Rate","Share Rate","Comment Rate","Save Rate","Overall\nEngagement"]
    rate_values = [
        overview_df["like_rate"].iloc[0],
        overview_df["share_rate"].iloc[0],
        overview_df["comment_rate"].iloc[0],
        overview_df["save_rate"].iloc[0],
        overview_df["engagement_rate"].iloc[0],
    ]
    plt.figure()
    ax = plt.gca()
    bars = ax.bar(rate_labels, rate_values, color=["#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"])
    ax.set_title("Engagement Rates (%)", fontweight="bold")
    ax.set_ylabel("Percentage")
    ax.set_xlabel("Engagements")
    ax.tick_params(axis="x", rotation=45)
    _annotate_bars(ax, rate_values, "{:.2f}%")
    
    ax.legend(bars, rate_labels, title="Metrics", loc="center left",
            bbox_to_anchor=(1.02, 0.85), frameon=True)

    plt.tight_layout()
    plt.show()

    # --- Content stats (text-only figure) ---
    plt.figure()
    ax = plt.gca()
    ax.axis("off")
    ax.set_title("Content Stats", fontweight="bold")
    ax.text(0.5, 0.7, f"Total Videos\n{int(overview_df['total_videos'].iloc[0]):,}",
            ha="center", va="center", fontsize=20, fontweight="bold")
    ax.text(0.5, 0.4, f"Avg Duration\n{overview_df['avg_duration_seconds'].iloc[0]:.1f}s",
            ha="center", va="center", fontsize=16)
    ax.text(0.5, 0.15, f"Avg Videos/Month\n{overview_df['avg_videos_per_month'].iloc[0]:.1f}",
            ha="center", va="center", fontsize=16, color="#2ca02c")
    plt.tight_layout(); plt.show()

    # --- Summary statistics (text-only figure) ---
    summary_text = f"""
SUMMARY STATISTICS

Total Videos: {int(overview_df['total_videos'].iloc[0]):,}
Total Views: {overview_df['total_views'].iloc[0]:,.0f}
Total Likes: {overview_df['total_likes'].iloc[0]:,.0f}

Avg Views/Video: {overview_df['avg_views'].iloc[0]:,.0f}
Avg Likes/Video: {overview_df['avg_likes'].iloc[0]:,.0f}
Avg Videos/Month: {overview_df['avg_videos_per_month'].iloc[0]:.1f}

Engagement Rate: {overview_df['engagement_rate'].iloc[0]:.2f}%
"""
    plt.figure()
    ax = plt.gca()
    ax.axis("off")
    ax.text(0.05, 0.5, summary_text, ha="left", va="center", fontsize=11, family="monospace")
    plt.tight_layout(); plt.show()

def plot_top_videos(top_videos):
    """Three separate figures."""
    # Top by views
    tv = top_videos["by_views"].head(10)
    plt.figure()
    ax = plt.gca()
    ax.barh(range(len(tv)), tv["views"])
    ax.set_yticks(range(len(tv)))
    ax.set_yticklabels([t[:30]+"..." if len(t)>30 else t for t in tv["title"]], fontsize=8)
    ax.set_xlabel("Views"); ax.set_title("Top 10 by Views", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout(); plt.show()

    # Top by total engagement
    te = top_videos["by_engagement"].head(10)
    plt.figure()
    ax = plt.gca()
    ax.barh(range(len(te)), te["total_engagement"], color="orange")
    ax.set_yticks(range(len(te)))
    ax.set_yticklabels([t[:30]+"..." if len(t)>30 else t for t in te["title"]], fontsize=8)
    ax.set_xlabel("Total Engagement"); ax.set_title("Top 10 by Total Engagement", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout(); plt.show()

    # Top by engagement rate
    tr = top_videos["by_engagement_rate"].head(10)
    plt.figure()
    ax = plt.gca()
    ax.barh(range(len(tr)), tr["engagement_rate"], color="green")
    ax.set_yticks(range(len(tr)))
    ax.set_yticklabels([t[:30]+"..." if len(t)>30 else t for t in tr["title"]], fontsize=8)
    ax.set_xlabel("Engagement Rate (%)"); ax.set_title("Top 10 by Engagement Rate", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout(); plt.show()

def plot_temporal_analysis(temporal):
    """Four separate figures."""
    monthly = temporal["monthly"].copy()
    monthly["date_label"] = monthly["publish_year"].astype(str) + "-" + monthly["publish_month"].astype(str).str.zfill(2)
    x_idx = range(len(monthly))

    # Total views by month
    plt.figure()
    ax = plt.gca()
    ax.plot(x_idx, monthly["total_views"], marker="o", linewidth=2)
    ax.set_title("Total Views by Month", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Total Views")
    ax.grid(True, alpha=0.3); ax.tick_params(axis="x", rotation=45)
    ax.set_xticks(x_idx); ax.set_xticklabels(monthly["date_label"])
    plt.tight_layout(); plt.show()

    # Videos posted by month
    plt.figure()
    ax = plt.gca()
    ax.bar(x_idx, monthly["video_count"], color="coral")
    ax.set_title("Videos Posted by Month", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Video Count")
    ax.grid(True, alpha=0.3, axis="y"); ax.tick_params(axis="x", rotation=45)
    ax.set_xticks(x_idx); ax.set_xticklabels(monthly["date_label"])
    plt.tight_layout(); plt.show()

    # Average views per video by month
    plt.figure()
    ax = plt.gca()
    ax.plot(x_idx, monthly["avg_views"], marker="s", color="green", linewidth=2)
    ax.set_title("Average Views per Video by Month", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Avg Views")
    ax.grid(True, alpha=0.3); ax.tick_params(axis="x", rotation=45)
    ax.set_xticks(x_idx); ax.set_xticklabels(monthly["date_label"])
    plt.tight_layout(); plt.show()

    # Average views by day of week
    dow = temporal["day_of_week"]
    plt.figure()
    ax = plt.gca()
    ax.bar(dow["day_name"], dow["avg_views"], color="purple")
    ax.set_title("Average Views by Day of Week", fontweight="bold")
    ax.set_xlabel("Day of Week"); ax.set_ylabel("Avg Views")
    ax.tick_params(axis="x", rotation=45); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.show()

def plot_videos_per_month_by_year(yearly_stats):
    """Already produced as three separate figures: grouped bars, lines, heatmap."""
    years  = sorted(yearly_stats["publish_year"].unique())
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_nums = list(range(1, 13))
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
    x = np.arange(len(months))

    # --- Grouped Bar Chart ---
    plt.figure(figsize=(14,6))
    ax = plt.gca()
    ax.set_title("Videos Posted by Month - Year Comparison (Grouped Bar Chart)", fontweight="bold")
    width = 0.8 / len(years)
    for i, year in enumerate(years):
        yd = yearly_stats[yearly_stats["publish_year"] == year]
        vals = [(yd[yd["publish_month"] == m]["video_count"].values[0] if len(yd[yd["publish_month"] == m])>0 else 0)
                for m in month_nums]
        offset = (i - len(years)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=str(int(year)), color=colors[i])
        for j, (bar, val) in enumerate(zip(bars, vals)):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f"{int(val)}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Month", fontweight="bold"); ax.set_ylabel("Number of Videos", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(months); ax.legend(title="Year", loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.show()

    # --- Line Chart ---
    plt.figure(figsize=(14,6))
    ax = plt.gca()
    ax.set_title("Videos Posted by Month - Year Comparison (Line Chart)", fontweight="bold")
    for i, year in enumerate(years):
        yd = yearly_stats[yearly_stats["publish_year"] == year]
        vals = [(yd[yd["publish_month"] == m]["video_count"].values[0] if len(yd[yd["publish_month"] == m])>0 else 0)
                for m in month_nums]
        ax.plot(months, vals, marker="o", linewidth=2.5, markersize=8, label=str(int(year)), color=colors[i])
        for j, val in enumerate(vals):
            if val > 0:
                ax.text(j, val, f"{int(val)}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Month", fontweight="bold"); ax.set_ylabel("Number of Videos", fontweight="bold")
    ax.legend(title="Year", loc="upper left"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    # --- Heatmap ---
    plt.figure(figsize=(14,6))
    ax = plt.gca()
    ax.set_title("Videos Posted by Month - Year Comparison (Heatmap)", fontweight="bold")
    heatmap_data = []
    for year in years:
        yd = yearly_stats[yearly_stats["publish_year"] == year]
        vals = [(yd[yd["publish_month"] == m]["video_count"].values[0] if len(yd[yd["publish_month"] == m])>0 else 0)
                for m in month_nums]
        heatmap_data.append(vals)
    arr = np.array(heatmap_data)
    im = ax.imshow(arr, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(months))); ax.set_yticks(np.arange(len(years)))
    ax.set_xticklabels(months); ax.set_yticklabels([str(int(y)) for y in years])
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("Number of Videos", rotation=270, labelpad=20, fontweight="bold")
    for i in range(len(years)):
        for j in range(len(months)):
            ax.text(j, i, int(arr[i, j]),
                    ha="center", va="center",
                    color=("black" if arr[i, j] < arr.max()/2 else "white"),
                    fontsize=9, fontweight="bold")
    ax.set_xlabel("Month", fontweight="bold"); ax.set_ylabel("Year", fontweight="bold")
    plt.tight_layout(); plt.show()

def plot_content_analysis(content):
    """Four separate figures (if data available)."""
    # Post type count
    if not content["post_type"].empty:
        pt = content["post_type"]
        plt.figure()
        ax = plt.gca()
        ax.bar(pt["post_type"], pt["video_count"], color="steelblue")
        ax.set_title("Video Count by Post Type", fontweight="bold")
        ax.set_xlabel("Post Type"); ax.set_ylabel("Count"); ax.tick_params(axis="x", rotation=45)
        for i, v in enumerate(pt["video_count"]):
            ax.text(i, v, str(int(v)), ha="center", va="bottom")
        plt.tight_layout(); plt.show()

        # Avg views by post type
        plt.figure()
        ax = plt.gca()
        ax.bar(pt["post_type"], pt["avg_views"], color="orange")
        ax.set_title("Average Views by Post Type", fontweight="bold")
        ax.set_xlabel("Post Type"); ax.set_ylabel("Avg Views"); ax.tick_params(axis="x", rotation=45)
        plt.tight_layout(); plt.show()

    # Duration analysis
    if not content["duration"].empty:
        dur = content["duration"]
        plt.figure()
        ax = plt.gca()
        ax.bar(dur["duration_bucket"], dur["video_count"], color="green")
        ax.set_title("Video Count by Duration", fontweight="bold")
        ax.set_xlabel("Duration Bucket"); ax.set_ylabel("Count"); ax.tick_params(axis="x", rotation=45)
        for i, v in enumerate(dur["video_count"]):
            ax.text(i, v, str(int(v)), ha="center", va="bottom")
        plt.tight_layout(); plt.show()

        # Avg engagement by duration
        plt.figure()
        ax = plt.gca()
        ax.bar(dur["duration_bucket"], dur["avg_engagement"], color="purple")
        ax.set_title("Average Engagement by Duration", fontweight="bold")
        ax.set_xlabel("Duration Bucket"); ax.set_ylabel("Avg Engagement"); ax.tick_params(axis="x", rotation=45)
        plt.tight_layout(); plt.show()

    # Sound used vs not - IMPROVED VERSION
    if not content["sound"].empty:
        sd = content["sound"]
        plt.figure(figsize=(14, 6))
        ax = plt.gca()
        
        # Sort by total views descending
        sd_sorted = sd.sort_values("total_views", ascending=True)
        
        bars = ax.barh(range(len(sd_sorted)), sd_sorted["total_views"], color="#8c564b")
        ax.set_yticks(range(len(sd_sorted)))
        ax.set_yticklabels([label[:40]+"..." if len(label)>40 else label for label in sd_sorted["sound_category"]], fontsize=9)
        ax.set_xlabel("Total Views", fontweight="bold")
        ax.set_title("Total Views by Sound Usage", fontweight="bold")
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, sd_sorted["total_views"])):
            ax.text(val, bar.get_y() + bar.get_height()/2, f"{int(val):,}", 
                   ha="left", va="center", fontsize=8, fontweight="bold")
        
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.show()

def plot_engagement_distribution(conn):
    """Each distribution in its own figure."""
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
    df = execute_query(conn, query)

    # Views
    plt.figure(figsize=(12,6))
    ax = plt.gca()
    ax.hist(df["views"], bins=50, color="skyblue", edgecolor="black")
    ax.set_title("Views Distribution", fontweight="bold"); ax.set_xlabel("Views"); ax.set_ylabel("Frequency")
    ax.axvline(df["views"].median(), color="red", linestyle="--", label=f"Median: {df['views'].median():,.0f}")
    ax.legend(); plt.tight_layout(); plt.show()

    # Likes
    plt.figure(figsize=(12,6))
    ax = plt.gca()
    ax.hist(df["likes"], bins=50, color="orange", edgecolor="black")
    ax.set_title("Likes Distribution", fontweight="bold"); ax.set_xlabel("Likes"); ax.set_ylabel("Frequency")
    ax.axvline(df["likes"].median(), color="red", linestyle="--", label=f"Median: {df['likes'].median():,.0f}")
    ax.legend(); plt.tight_layout(); plt.show()

    # Shares
    plt.figure(figsize=(12,6))
    ax = plt.gca()
    ax.hist(df["shares"], bins=50, color="green", edgecolor="black")
    ax.set_title("Shares Distribution", fontweight="bold"); ax.set_xlabel("Shares"); ax.set_ylabel("Frequency")
    ax.axvline(df["shares"].median(), color="red", linestyle="--", label=f"Median: {df['shares'].median():,.0f}")
    ax.legend(); plt.tight_layout(); plt.show()

    # Like Rate
    plt.figure(figsize=(12,6))
    ax = plt.gca()
    ax.hist(df["like_rate"], bins=50, color="purple", edgecolor="black")
    ax.set_title("Like Rate Distribution (%)", fontweight="bold"); ax.set_xlabel("Like Rate (%)"); ax.set_ylabel("Frequency")
    ax.axvline(df["like_rate"].median(), color="red", linestyle="--", label=f"Median: {df['like_rate'].median():.2f}%")
    ax.legend(); plt.tight_layout(); plt.show()

    # Engagement Rate
    plt.figure(figsize=(12,6))
    ax = plt.gca()
    ax.hist(df["engagement_rate"], bins=50, color="coral", edgecolor="black")
    ax.set_title("Engagement Rate Distribution (%)", fontweight="bold"); ax.set_xlabel("Engagement Rate (%)"); ax.set_ylabel("Frequency")
    ax.axvline(df["engagement_rate"].median(), color="red", linestyle="--", label=f"Median: {df['engagement_rate'].median():.2f}%")
    ax.legend(); plt.tight_layout(); plt.show()

    # Box plot comparison
    plt.figure(figsize=(12,6))
    ax = plt.gca()
    ax.boxplot([df["like_rate"], df["share_rate"], df["engagement_rate"]],
               labels=["Like Rate","Share Rate","Engagement Rate"])
    ax.set_title("Engagement Rates Comparison", fontweight="bold"); ax.set_ylabel("Rate (%)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.show()

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

def get_top_performing_videos(conn, limit=10):
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

# ================= MAIN EXECUTION =================
def generate_descriptive_analytics():
    print("Starting TikTok Descriptive Analytics with Visualizations...\n")
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
                import time; print(f"  Retrying in {retry_delay} seconds...\n"); time.sleep(retry_delay)
            else:
                print("\n❌ Failed to connect to database after multiple attempts.")
                print("Please check internet, server status, and credentials."); return
    if conn is None:
        print("❌ Could not establish database connection."); return

    try:
        print("Fetching data...")
        overview = get_overview_metrics(conn)
        top_videos = get_top_performing_videos(conn, limit=10)
        temporal = get_temporal_analysis(conn)
        content  = get_content_analysis(conn)

        print(f"\nAverage Videos Posted per Month: {overview['avg_videos_per_month'].iloc[0]:.2f}")

        print("\nGenerating visualizations...\n")
        print("1. Overview Metrics");                 plot_overview_metrics(overview)
        print("2. Top Performing Videos");            plot_top_videos(top_videos)
        print("3. Temporal Analysis");                plot_temporal_analysis(temporal)
        print("4. Videos per Month by Year");         plot_videos_per_month_by_year(temporal["yearly"])
        print("5. Content Analysis");                 plot_content_analysis(content)
        print("6. Engagement Distributions");         plot_engagement_distribution(conn)
        print("\nAll visualizations complete!")

    except Exception as e:
        print(f"Error: {str(e)}"); raise
    finally:
        conn.close()

if __name__ == "__main__":
    generate_descriptive_analytics()