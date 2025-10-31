import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Database connection parameters
DB_PARAMS = {
    "dbname": "neondb",
    "user": "neondb_owner",
    "password": "npg_dGzvq4CJPRx7",
    "host": "ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech",
    "port": "5432",
    "sslmode": "require",
}

def fetch_data():
    """Fetch data from PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        query = """
        SELECT 
            post_id, page_id, page_name, title, description, post_type,
            duration_sec, publish_time, year, month, day, time,
            permalink, is_crosspost, is_share, funded_content_status,
            reach, shares, comments, reactions, seconds_viewed,
            average_seconds_viewed, impressions
        FROM public.facebook_data_set
        ORDER BY publish_time
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"Successfully fetched {len(df)} rows from database")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        return None

def prepare_data(df):
    """Prepare and clean data for analysis"""
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    df['date'] = df['publish_time'].dt.date
    
    # Fill missing values with 0 for engagement metrics
    df['reactions'] = df['reactions'].fillna(0)
    df['comments'] = df['comments'].fillna(0)
    df['shares'] = df['shares'].fillna(0)
    df['reach'] = df['reach'].fillna(0)
    
    # Calculate engagement metrics
    df['total_engagement'] = df['reactions'] + df['comments'] + df['shares']
    
    # Calculate engagement rate, handling zero reach
    df['engagement_rate'] = np.where(
        df['reach'] > 0,
        df['total_engagement'] / df['reach'],
        0
    )
    
    df['completion_rate'] = np.where(
        df['duration_sec'] > 0,
        df['average_seconds_viewed'] / df['duration_sec'],
        np.nan
    )
    
    return df

def plot_cumulative_reach(df):
    """Plot cumulative reach growth over time"""
    # Sort by publish time and calculate cumulative metrics
    df_sorted = df.sort_values('publish_time').copy()
    df_sorted['cumulative_reach'] = df_sorted['reach'].cumsum()
    df_sorted['cumulative_engagement'] = df_sorted['total_engagement'].cumsum()
    
    # Calculate cumulative engagement rate
    df_sorted['cumulative_engagement_rate'] = np.where(
        df_sorted['cumulative_reach'] > 0,
        (df_sorted['cumulative_engagement'] / df_sorted['cumulative_reach']) * 100,
        0
    )
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Cumulative Reach
    ax1.plot(df_sorted['publish_time'], df_sorted['cumulative_reach'], 
             linewidth=2, color='#1f77b4', marker='o', markersize=3)
    ax1.fill_between(df_sorted['publish_time'], df_sorted['cumulative_reach'], 
                     alpha=0.3, color='#1f77b4')
    
    ax1.set_title('Cumulative Reach Growth Over Time', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative Reach', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Plot 2: Cumulative Engagement Rate
    ax2.plot(df_sorted['publish_time'], df_sorted['cumulative_engagement_rate'], 
             linewidth=2, color='#2ecc71', marker='o', markersize=3)
    ax2.fill_between(df_sorted['publish_time'], df_sorted['cumulative_engagement_rate'], 
                     alpha=0.3, color='#2ecc71')
    
    ax2.set_title('Cumulative Engagement Rate Over Time', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Cumulative Engagement Rate (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_engagement_summary_metrics(df):
    """Plot total engagement metrics and engagement rates"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate totals
    total_reactions = df['reactions'].sum()
    total_comments = df['comments'].sum()
    total_shares = df['shares'].sum()
    total_reach = df['reach'].sum()
    total_engagement = df['total_engagement'].sum()
    
    # Metrics data for first chart
    metrics = ['Reactions', 'Comments', 'Shares']
    values = [total_reactions, total_comments, total_shares]
    colors_chart1 = ['#ff7f0e', '#9467bd', '#2ca02c']
    
    # Plot 1: Total Engagement Metrics
    bars1 = ax1.bar(metrics, values, color=colors_chart1, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_title('Total Engagement Metrics', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Calculate engagement rates
    like_rate = (total_reactions / total_reach * 100) if total_reach > 0 else 0
    comment_rate = (total_comments / total_reach * 100) if total_reach > 0 else 0
    share_rate = (total_shares / total_reach * 100) if total_reach > 0 else 0
    overall_engagement_rate = (total_engagement / total_reach * 100) if total_reach > 0 else 0
    
    # Rates data for second chart
    rate_labels = ['Like Rate', 'Comment Rate', 'Share Rate', 'Overall\nEngagement']
    rate_values = [like_rate, comment_rate, share_rate, overall_engagement_rate]
    colors_chart2 = ['#ff7f0e', '#9467bd', '#2ca02c', '#8b4513']
    
    # Plot 2: Engagement Rates
    bars2 = ax2.bar(rate_labels, rate_values, color=colors_chart2, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_title('Engagement Rates (%)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Percentage', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_engagement_by_post_type(df):
    """Analyze engagement by post type"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Average engagement by post type
    engagement_by_type = df.groupby('post_type').agg({
        'total_engagement': 'mean',
        'reach': 'mean',
        'engagement_rate': 'mean',
        'post_id': 'count'
    }).reset_index()
    engagement_by_type.columns = ['post_type', 'avg_engagement', 'avg_reach', 'avg_engagement_rate', 'post_count']
    
    axes[0, 0].bar(engagement_by_type['post_type'], engagement_by_type['avg_engagement'], 
                   color='#2ecc71', alpha=0.8)
    axes[0, 0].set_title('Average Total Engagement by Post Type', fontweight='bold')
    axes[0, 0].set_xlabel('Post Type')
    axes[0, 0].set_ylabel('Avg Total Engagement')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Reach by post type
    axes[0, 1].bar(engagement_by_type['post_type'], engagement_by_type['avg_reach'], 
                   color='#3498db', alpha=0.8)
    axes[0, 1].set_title('Average Reach by Post Type', fontweight='bold')
    axes[0, 1].set_xlabel('Post Type')
    axes[0, 1].set_ylabel('Avg Reach')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Engagement rate by post type
    axes[1, 0].bar(engagement_by_type['post_type'], engagement_by_type['avg_engagement_rate'] * 100, 
                   color='#e74c3c', alpha=0.8)
    axes[1, 0].set_title('Average Engagement Rate by Post Type', fontweight='bold')
    axes[1, 0].set_xlabel('Post Type')
    axes[1, 0].set_ylabel('Avg Engagement Rate (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Post count by type
    axes[1, 1].pie(engagement_by_type['post_count'], labels=engagement_by_type['post_type'],
                   autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    axes[1, 1].set_title('Distribution of Post Types', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_temporal_patterns(df):
    """Analyze temporal posting and engagement patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Posts per month
    monthly_posts = df.groupby(['year', 'month']).size().reset_index(name='post_count')
    monthly_posts['year_month'] = pd.to_datetime(monthly_posts[['year', 'month']].assign(day=1))
    
    axes[0, 0].plot(monthly_posts['year_month'], monthly_posts['post_count'], 
                    marker='o', linewidth=2, color='#9b59b6')
    axes[0, 0].set_title('Posts Published Per Month', fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Number of Posts')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Engagement by day of week
    df['day_of_week'] = df['publish_time'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    engagement_by_day = df.groupby('day_of_week')['total_engagement'].mean().reindex(day_order)
    
    axes[0, 1].bar(range(7), engagement_by_day.values, color='#f39c12', alpha=0.8)
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(day_order, rotation=45, ha='right')
    axes[0, 1].set_title('Average Engagement by Day of Week', fontweight='bold')
    axes[0, 1].set_ylabel('Avg Total Engagement')
    
    # 3. Reach by hour (if time data is available)
    df['hour'] = df['publish_time'].dt.hour
    reach_by_hour = df.groupby('hour')['reach'].mean()
    
    axes[1, 0].plot(reach_by_hour.index, reach_by_hour.values, 
                    marker='o', linewidth=2, color='#16a085')
    axes[1, 0].set_title('Average Reach by Hour of Day', fontweight='bold')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Avg Reach')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(range(0, 24, 2))
    
    # 4. Monthly reach trend
    monthly_reach = df.groupby(['year', 'month'])['reach'].sum().reset_index()
    monthly_reach['year_month'] = pd.to_datetime(monthly_reach[['year', 'month']].assign(day=1))
    
    axes[1, 1].plot(monthly_reach['year_month'], monthly_reach['reach'], 
                    marker='o', linewidth=2, color='#c0392b')
    axes[1, 1].fill_between(monthly_reach['year_month'], monthly_reach['reach'], alpha=0.3, color='#c0392b')
    axes[1, 1].set_title('Total Monthly Reach Trend', fontweight='bold')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Total Reach')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_video_performance(df):
    """Analyze video content performance"""
    video_df = df[df['duration_sec'] > 0].copy()
    
    if len(video_df) == 0:
        print("No video data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Video duration vs engagement
    axes[0, 0].scatter(video_df['duration_sec'], video_df['total_engagement'], 
                       alpha=0.5, color='#8e44ad')
    axes[0, 0].set_title('Video Duration vs Total Engagement', fontweight='bold')
    axes[0, 0].set_xlabel('Duration (seconds)')
    axes[0, 0].set_ylabel('Total Engagement')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Completion rate distribution
    completion_rates = video_df['completion_rate'].dropna()
    axes[0, 1].hist(completion_rates * 100, bins=30, color='#27ae60', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Video Completion Rate Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Completion Rate (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(completion_rates.mean() * 100, color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {completion_rates.mean()*100:.1f}%')
    axes[0, 1].legend()
    
    # 3. Duration bins vs avg engagement
    video_df['duration_bin'] = pd.cut(video_df['duration_sec'], 
                                       bins=[0, 30, 60, 120, 300, float('inf')],
                                       labels=['0-30s', '30-60s', '1-2m', '2-5m', '5m+'])
    engagement_by_duration = video_df.groupby('duration_bin')['total_engagement'].mean()
    
    axes[1, 0].bar(range(len(engagement_by_duration)), engagement_by_duration.values, 
                   color='#d35400', alpha=0.8)
    axes[1, 0].set_xticks(range(len(engagement_by_duration)))
    axes[1, 0].set_xticklabels(engagement_by_duration.index, rotation=45)
    axes[1, 0].set_title('Average Engagement by Video Duration Range', fontweight='bold')
    axes[1, 0].set_ylabel('Avg Total Engagement')
    
    # 4. Average watch time vs completion rate
    axes[1, 1].scatter(video_df['average_seconds_viewed'], video_df['completion_rate'] * 100,
                       alpha=0.5, color='#2980b9')
    axes[1, 1].set_title('Avg Watch Time vs Completion Rate', fontweight='bold')
    axes[1, 1].set_xlabel('Avg Seconds Viewed')
    axes[1, 1].set_ylabel('Completion Rate (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def generate_summary_stats(df):
    """Generate and print summary statistics"""
    print("\n" + "="*60)
    print("FACEBOOK DATA DESCRIPTIVE ANALYTICS SUMMARY")
    print("="*60)
    
    print(f"\nDataset Overview:")
    print(f"  Total Posts: {len(df):,}")
    print(f"  Date Range: {df['publish_time'].min()} to {df['publish_time'].max()}")
    print(f"  Unique Pages: {df['page_id'].nunique()}")
    
    print(f"\nReach Metrics:")
    print(f"  Total Reach: {df['reach'].sum():,.0f}")
    print(f"  Average Reach per Post: {df['reach'].mean():,.0f}")
    print(f"  Median Reach: {df['reach'].median():,.0f}")
    print(f"  Max Reach (Single Post): {df['reach'].max():,.0f}")
    
    print(f"\nEngagement Metrics:")
    print(f"  Total Engagement: {df['total_engagement'].sum():,.0f}")
    print(f"  Average Engagement per Post: {df['total_engagement'].mean():,.0f}")
    print(f"  Average Engagement Rate: {df['engagement_rate'].mean()*100:.2f}%")
    print(f"  Total Reactions: {df['reactions'].sum():,.0f}")
    print(f"  Total Comments: {df['comments'].sum():,.0f}")
    print(f"  Total Shares: {df['shares'].sum():,.0f}")
    
    print(f"\nPost Type Distribution:")
    for post_type, count in df['post_type'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"  {post_type}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nTop 5 Performing Posts (by Reach):")
    top_posts = df.nlargest(5, 'reach')[['title', 'post_type', 'reach', 'total_engagement', 'publish_time']]
    for idx, row in top_posts.iterrows():
        print(f"  • {row['title'][:50]}...")
        print(f"    Type: {row['post_type']}, Reach: {row['reach']:,.0f}, Engagement: {row['total_engagement']:,.0f}")
    
    print("\n" + "="*60 + "\n")

def main():
    """Main execution function"""
    print("Fetching data from database...")
    df = fetch_data()
    
    if df is None or len(df) == 0:
        print("No data retrieved. Exiting.")
        return
    
    print(f"Data loaded successfully: {len(df)} rows")
    
    # Prepare data
    df = prepare_data(df)
    
    # Generate visualizations
    print("\nGenerating cumulative reach chart...")
    plot_cumulative_reach(df)
    
    print("Generating engagement summary metrics...")
    plot_engagement_summary_metrics(df)
    
    print("Generating post type engagement analysis...")
    plot_engagement_by_post_type(df)
    
    print("Generating temporal pattern analysis...")
    plot_temporal_patterns(df)
    
    print("Generating video performance analysis...")
    plot_video_performance(df)
    
    # Print summary statistics
    generate_summary_stats(df)
    
    print("\nAll visualizations displayed successfully!")

if __name__ == "__main__":
    main()