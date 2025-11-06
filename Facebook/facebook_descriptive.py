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
    
    # Calculate engagement rate and ratios
    df['engagement_rate'] = np.where(
        df['reach'] > 0,
        df['total_engagement'] / df['reach'],
        0
    )
    
    df['reactions_to_reach_ratio'] = np.where(
        df['reach'] > 0,
        df['reactions'] / df['reach'],
        0
    )
    
    df['comments_to_reach_ratio'] = np.where(
        df['reach'] > 0,
        df['comments'] / df['reach'],
        0
    )
    
    df['shares_to_reach_ratio'] = np.where(
        df['reach'] > 0,
        df['shares'] / df['reach'],
        0
    )
    
    df['shares_to_reactions_ratio'] = np.where(
        df['reactions'] > 0,
        df['shares'] / df['reactions'],
        0
    )
    
    df['completion_rate'] = np.where(
        df['duration_sec'] > 0,
        df['average_seconds_viewed'] / df['duration_sec'],
        np.nan
    )
    
    return df

def plot_cumulative_reach(df):
    """Plot cumulative reach growth over time with median and average reach"""
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
    
    # Calculate rolling average and median reach
    window_size = min(10, len(df_sorted))  # Use 10-post rolling window or less if fewer posts
    df_sorted['rolling_avg_reach'] = df_sorted['reach'].rolling(window=window_size, min_periods=1).mean()
    df_sorted['rolling_median_reach'] = df_sorted['reach'].rolling(window=window_size, min_periods=1).median()
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))
    
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
    
    # Plot 3: Rolling Average and Median Reach
    ax3.plot(df_sorted['publish_time'], df_sorted['rolling_avg_reach'], 
             linewidth=2, color='#e74c3c', marker='o', markersize=3, label='Rolling Average')
    ax3.plot(df_sorted['publish_time'], df_sorted['rolling_median_reach'], 
             linewidth=2, color='#9b59b6', marker='s', markersize=3, label='Rolling Median')
    ax3.fill_between(df_sorted['publish_time'], df_sorted['rolling_avg_reach'], 
                     alpha=0.2, color='#e74c3c')
    
    # Add overall average and median as horizontal lines
    overall_avg = df_sorted['reach'].mean()
    overall_median = df_sorted['reach'].median()
    ax3.axhline(overall_avg, color='#c0392b', linestyle='--', linewidth=2, 
                label=f'Overall Avg: {overall_avg:,.0f}')
    ax3.axhline(overall_median, color='#8e44ad', linestyle='--', linewidth=2, 
                label=f'Overall Median: {overall_median:,.0f}')
    
    ax3.set_title(f'Rolling Average & Median Reach (Window: {window_size} posts)', 
                  fontsize=16, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Reach', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax3.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_engagement_summary_metrics(df):
    """Plot total engagement metrics and engagement rates with reach statistics"""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    
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
    ax1.set_title('Total Engagement Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
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
    ax2.set_title('Engagement Rates (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Percentage', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Reach Statistics
    avg_reach = df['reach'].mean()
    median_reach = df['reach'].median()
    max_reach = df['reach'].max()
    
    reach_metrics = ['Average\nReach', 'Median\nReach', 'Max\nReach']
    reach_values = [avg_reach, median_reach, max_reach]
    colors_chart3 = ['#3498db', '#9b59b6', '#e74c3c']
    
    bars3 = ax3.bar(reach_metrics, reach_values, color=colors_chart3, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_title('Reach Statistics', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Reach', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Reach Distribution Histogram
    ax4.hist(df['reach'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax4.axvline(avg_reach, color='#e74c3c', linestyle='--', linewidth=2, 
                label=f'Average: {avg_reach:,.0f}')
    ax4.axvline(median_reach, color='#9b59b6', linestyle='--', linewidth=2, 
                label=f'Median: {median_reach:,.0f}')
    
    ax4.set_title('Reach Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Reach', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=11, loc='upper right')
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.show()

def plot_engagement_ratio_analysis(df):
    """Plot detailed engagement ratio analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    total_reach = df['reach'].sum()
    total_reactions = df['reactions'].sum()
    
    # Overall ratios
    reactions_reach = (df['reactions'].sum() / total_reach * 100) if total_reach > 0 else 0
    comments_reach = (df['comments'].sum() / total_reach * 100) if total_reach > 0 else 0
    shares_reach = (df['shares'].sum() / total_reach * 100) if total_reach > 0 else 0
    engagement_rate = (df['total_engagement'].sum() / total_reach * 100) if total_reach > 0 else 0
    virality_ratio = (df['shares'].sum() / total_reactions * 100) if total_reactions > 0 else 0
    
    # 1. Reactions to Reach Ratio by Post Type
    reactions_by_type = df.groupby('post_type').agg({
        'reactions': 'sum',
        'reach': 'sum'
    })
    reactions_by_type['ratio'] = (reactions_by_type['reactions'] / reactions_by_type['reach'] * 100)
    
    axes[0, 0].bar(reactions_by_type.index, reactions_by_type['ratio'], 
                   color='#ff7f0e', alpha=0.8, edgecolor='black')
    axes[0, 0].set_title('Reactions-to-Reach Ratio by Post Type', fontweight='bold')
    axes[0, 0].set_ylabel('Ratio (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(reactions_reach, color='red', linestyle='--', 
                       label=f'Overall: {reactions_reach:.2f}%', linewidth=2)
    axes[0, 0].legend()
    
    # 2. Comments to Reach Ratio by Post Type
    comments_by_type = df.groupby('post_type').agg({
        'comments': 'sum',
        'reach': 'sum'
    })
    comments_by_type['ratio'] = (comments_by_type['comments'] / comments_by_type['reach'] * 100)
    
    axes[0, 1].bar(comments_by_type.index, comments_by_type['ratio'], 
                   color='#9467bd', alpha=0.8, edgecolor='black')
    axes[0, 1].set_title('Comments-to-Reach Ratio by Post Type', fontweight='bold')
    axes[0, 1].set_ylabel('Ratio (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].axhline(comments_reach, color='red', linestyle='--', 
                       label=f'Overall: {comments_reach:.2f}%', linewidth=2)
    axes[0, 1].legend()
    
    # 3. Shares to Reach Ratio by Post Type
    shares_by_type = df.groupby('post_type').agg({
        'shares': 'sum',
        'reach': 'sum'
    })
    shares_by_type['ratio'] = (shares_by_type['shares'] / shares_by_type['reach'] * 100)
    
    axes[0, 2].bar(shares_by_type.index, shares_by_type['ratio'], 
                   color='#2ca02c', alpha=0.8, edgecolor='black')
    axes[0, 2].set_title('Shares-to-Reach Ratio by Post Type', fontweight='bold')
    axes[0, 2].set_ylabel('Ratio (%)')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    axes[0, 2].axhline(shares_reach, color='red', linestyle='--', 
                       label=f'Overall: {shares_reach:.2f}%', linewidth=2)
    axes[0, 2].legend()
    
    # 4. Overall Engagement Rate by Post Type
    engagement_by_type = df.groupby('post_type').agg({
        'total_engagement': 'sum',
        'reach': 'sum'
    })
    engagement_by_type['ratio'] = (engagement_by_type['total_engagement'] / engagement_by_type['reach'] * 100)
    
    axes[1, 0].bar(engagement_by_type.index, engagement_by_type['ratio'], 
                   color='#8b4513', alpha=0.8, edgecolor='black')
    axes[1, 0].set_title('Engagement Rate by Post Type', fontweight='bold')
    axes[1, 0].set_ylabel('Rate (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].axhline(engagement_rate, color='red', linestyle='--', 
                       label=f'Overall: {engagement_rate:.2f}%', linewidth=2)
    axes[1, 0].legend()
    
    # 5. Virality Ratio (Shares to Reactions) by Post Type
    virality_by_type = df.groupby('post_type').agg({
        'shares': 'sum',
        'reactions': 'sum'
    })
    virality_by_type['ratio'] = (virality_by_type['shares'] / virality_by_type['reactions'] * 100)
    
    axes[1, 1].bar(virality_by_type.index, virality_by_type['ratio'], 
                   color='#d62728', alpha=0.8, edgecolor='black')
    axes[1, 1].set_title('Virality Ratio (Shares-to-Reactions) by Post Type', fontweight='bold')
    axes[1, 1].set_ylabel('Ratio (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axhline(virality_ratio, color='red', linestyle='--', 
                       label=f'Overall: {virality_ratio:.2f}%', linewidth=2)
    axes[1, 1].legend()
    
    # 6. Summary comparison of all ratios
    summary_labels = ['Reactions/\nReach', 'Comments/\nReach', 'Shares/\nReach', 
                     'Engagement\nRate', 'Virality\n(Shares/Reactions)']
    summary_values = [reactions_reach, comments_reach, shares_reach, engagement_rate, virality_ratio]
    summary_colors = ['#ff7f0e', '#9467bd', '#2ca02c', '#8b4513', '#d62728']
    
    bars = axes[1, 2].bar(summary_labels, summary_values, color=summary_colors, 
                         alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1, 2].set_title('Overall Engagement Ratios Summary', fontweight='bold')
    axes[1, 2].set_ylabel('Ratio (%)')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_engagement_by_post_type(df):
    """Analyze engagement by post type with median and average reach"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Average engagement by post type
    engagement_by_type = df.groupby('post_type').agg({
        'total_engagement': 'mean',
        'reach': ['mean', 'median'],
        'engagement_rate': 'mean',
        'post_id': 'count'
    }).reset_index()
    engagement_by_type.columns = ['post_type', 'avg_engagement', 'avg_reach', 'median_reach', 'avg_engagement_rate', 'post_count']
    
    axes[0, 0].bar(engagement_by_type['post_type'], engagement_by_type['avg_engagement'], 
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    axes[0, 0].set_title('Average Total Engagement by Post Type', fontweight='bold')
    axes[0, 0].set_xlabel('Post Type')
    axes[0, 0].set_ylabel('Avg Total Engagement')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Average reach by post type
    axes[0, 1].bar(engagement_by_type['post_type'], engagement_by_type['avg_reach'], 
                   color='#3498db', alpha=0.8, edgecolor='black', label='Average')
    axes[0, 1].set_title('Average Reach by Post Type', fontweight='bold')
    axes[0, 1].set_xlabel('Post Type')
    axes[0, 1].set_ylabel('Avg Reach')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # 3. Median reach by post type
    axes[0, 2].bar(engagement_by_type['post_type'], engagement_by_type['median_reach'], 
                   color='#9b59b6', alpha=0.8, edgecolor='black')
    axes[0, 2].set_title('Median Reach by Post Type', fontweight='bold')
    axes[0, 2].set_xlabel('Post Type')
    axes[0, 2].set_ylabel('Median Reach')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    axes[0, 2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # 4. Engagement rate by post type
    axes[1, 0].bar(engagement_by_type['post_type'], engagement_by_type['avg_engagement_rate'] * 100, 
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    axes[1, 0].set_title('Average Engagement Rate by Post Type', fontweight='bold')
    axes[1, 0].set_xlabel('Post Type')
    axes[1, 0].set_ylabel('Avg Engagement Rate (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. Combined Average vs Median Reach Comparison
    x = np.arange(len(engagement_by_type['post_type']))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, engagement_by_type['avg_reach'], width, 
                   label='Average', color='#3498db', alpha=0.8, edgecolor='black')
    axes[1, 1].bar(x + width/2, engagement_by_type['median_reach'], width,
                   label='Median', color='#9b59b6', alpha=0.8, edgecolor='black')
    
    axes[1, 1].set_title('Average vs Median Reach by Post Type', fontweight='bold')
    axes[1, 1].set_xlabel('Post Type')
    axes[1, 1].set_ylabel('Reach')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(engagement_by_type['post_type'], rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # 6. Post count by type
    axes[1, 2].pie(engagement_by_type['post_count'], labels=engagement_by_type['post_type'],
                   autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    axes[1, 2].set_title('Distribution of Post Types', fontweight='bold')
    
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
    
    # 1. Video Duration Distribution with Engagement Heatmap
    # Create bins for duration
    video_df['duration_bin'] = pd.cut(video_df['duration_sec'], 
                                       bins=[0, 30, 60, 120, 180, 300, float('inf')],
                                       labels=['0-30s', '30-60s', '1-2m', '2-3m', '3-5m', '5m+'])
    
    duration_stats = video_df.groupby('duration_bin').agg({
        'total_engagement': ['mean', 'count'],
        'reach': 'mean',
        'completion_rate': 'mean'
    }).reset_index()
    duration_stats.columns = ['duration_bin', 'avg_engagement', 'count', 'avg_reach', 'avg_completion']
    
    # Bar chart with dual axis
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    bars = ax1.bar(duration_stats['duration_bin'], duration_stats['avg_engagement'], 
                   color='#3498db', alpha=0.7, label='Avg Engagement')
    ax1.set_xlabel('Video Duration')
    ax1.set_ylabel('Avg Total Engagement', color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_title('Video Engagement by Duration', fontweight='bold')
    
    line = ax1_twin.plot(duration_stats['duration_bin'], duration_stats['count'], 
                         color='#e74c3c', marker='o', linewidth=2, markersize=8, label='Video Count')
    ax1_twin.set_ylabel('Number of Videos', color='#e74c3c')
    ax1_twin.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9)
    
    # 2. Completion Rate Analysis
    completion_bins = video_df.groupby('duration_bin')['completion_rate'].apply(
        lambda x: x.dropna()
    ).reset_index(drop=True)
    
    # Box plot for completion rates by duration
    completion_data = [video_df[video_df['duration_bin'] == cat]['completion_rate'].dropna() * 100 
                      for cat in duration_stats['duration_bin']]
    
    bp = axes[0, 1].boxplot(completion_data, labels=duration_stats['duration_bin'], 
                           patch_artist=True, showmeans=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('#27ae60')
        patch.set_alpha(0.7)
    
    axes[0, 1].set_title('Completion Rate Distribution by Duration', fontweight='bold')
    axes[0, 1].set_xlabel('Video Duration')
    axes[0, 1].set_ylabel('Completion Rate (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Engagement vs Completion Rate Scatter (Performance Quadrant)
    scatter = axes[1, 0].scatter(video_df['completion_rate'] * 100, 
                                video_df['total_engagement'],
                                c=video_df['reach'], 
                                cmap='viridis', 
                                alpha=0.6, 
                                s=100,
                                edgecolors='black',
                                linewidth=0.5)
    
    # Add median lines to create quadrants
    median_completion = video_df['completion_rate'].median() * 100
    median_engagement = video_df['total_engagement'].median()
    
    axes[1, 0].axvline(median_completion, color='red', linestyle='--', alpha=0.5, linewidth=2)
    axes[1, 0].axhline(median_engagement, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    axes[1, 0].set_title('Video Performance Quadrant Analysis', fontweight='bold')
    axes[1, 0].set_xlabel('Completion Rate (%)')
    axes[1, 0].set_ylabel('Total Engagement')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add quadrant labels
    axes[1, 0].text(0.95, 0.95, 'High Engagement\nHigh Completion', 
                   transform=axes[1, 0].transAxes, fontsize=9, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    axes[1, 0].text(0.05, 0.05, 'Low Engagement\nLow Completion', 
                   transform=axes[1, 0].transAxes, fontsize=9,
                   verticalalignment='bottom', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    cbar = plt.colorbar(scatter, ax=axes[1, 0])
    cbar.set_label('Reach', rotation=270, labelpad=15)
    
    # 4. Top Performing Videos Table
    axes[1, 1].axis('off')
    
    # Get top 10 videos by engagement
    top_videos = video_df.nlargest(10, 'total_engagement')[
        ['title', 'duration_sec', 'total_engagement', 'reach', 'completion_rate']
    ].copy()
    
    top_videos['title'] = top_videos['title'].str[:30] + '...'
    top_videos['duration_sec'] = top_videos['duration_sec'].apply(lambda x: f"{int(x)}s")
    top_videos['completion_rate'] = (top_videos['completion_rate'] * 100).round(1).astype(str) + '%'
    top_videos['total_engagement'] = top_videos['total_engagement'].apply(lambda x: f"{int(x):,}")
    top_videos['reach'] = top_videos['reach'].apply(lambda x: f"{int(x):,}")
    
    table_data = [top_videos.columns.tolist()] + top_videos.values.tolist()
    
    table = axes[1, 1].table(cellText=table_data, cellLoc='left', loc='center',
                            colWidths=[0.35, 0.1, 0.15, 0.15, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    
    # Style header row
    for i in range(len(top_videos.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(top_videos.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    axes[1, 1].set_title('Top 10 Videos by Engagement', fontweight='bold', pad=20)
    
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
    
    # Calculate average posts per month
    months_active = df.groupby(['year', 'month']).size().count()
    avg_posts_per_month = len(df) / months_active if months_active > 0 else 0
    print(f"  Average Posts per Month: {avg_posts_per_month:.1f}")
    
    print(f"\nReach Metrics:")
    print(f"  Total Reach: {df['reach'].sum():,.0f}")
    print(f"  Average Reach per Post: {df['reach'].mean():,.0f}")
    print(f"  Median Reach per Post: {df['reach'].median():,.0f}")
    print(f"  Max Reach (Single Post): {df['reach'].max():,.0f}")
    print(f"  Min Reach (Single Post): {df['reach'].min():,.0f}")
    print(f"  Std Dev of Reach: {df['reach'].std():,.0f}")
    
    print(f"\nEngagement Metrics:")
    print(f"  Total Engagement: {df['total_engagement'].sum():,.0f}")
    print(f"  Average Engagement per Post: {df['total_engagement'].mean():,.0f}")
    print(f"  Median Engagement per Post: {df['total_engagement'].median():,.0f}")
    print(f"  Average Engagement Rate: {df['engagement_rate'].mean()*100:.2f}%")
    print(f"  Total Reactions: {df['reactions'].sum():,.0f}")
    print(f"  Total Comments: {df['comments'].sum():,.0f}")
    print(f"  Total Shares: {df['shares'].sum():,.0f}")
    
    print(f"\nPost Type Distribution:")
    for post_type, count in df['post_type'].value_counts().items():
        percentage = (count / len(df)) * 100
        avg_reach = df[df['post_type'] == post_type]['reach'].mean()
        median_reach = df[df['post_type'] == post_type]['reach'].median()
        print(f"  {post_type}: {count:,} ({percentage:.1f}%)")
        print(f"    - Avg Reach: {avg_reach:,.0f}, Median Reach: {median_reach:,.0f}")
    
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
    print("\nGenerating cumulative reach chart with median/average...")
    plot_cumulative_reach(df)
    
    print("Generating engagement summary metrics with reach statistics...")
    plot_engagement_summary_metrics(df)
    
    print("Generating engagement ratio analysis...")
    plot_engagement_ratio_analysis(df)
    
    print("Generating post type engagement analysis with median/average reach...")
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