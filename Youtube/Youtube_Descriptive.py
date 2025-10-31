import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Database connection
db_params = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host': 'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port': '5432',
    'sslmode': 'require'
}

class YouTubeDescriptiveAnalytics:
    def __init__(self, db_params):
        self.db_params = db_params
        self.df = None
        
    def connect_and_load_data(self):
        """Connect to database and load YouTube data"""
        try:
            conn = psycopg2.connect(**self.db_params)
            query = "SELECT * FROM yt_video_etl"
            self.df = pd.read_sql(query, conn)
            conn.close()
            print(f"Data loaded successfully: {len(self.df)} records")
            print(f"Columns: {list(self.df.columns)}")
            print(f"Data types:\n{self.df.dtypes}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_data(self):
        """Clean and prepare data for analysis"""
        if self.df is None:
            print("No data loaded. Please run connect_and_load_data() first.")
            return False
            
        # Create publish_date column
        self.df['publish_date'] = pd.to_datetime(
            self.df['publish_year'].astype(str) + '-' + 
            self.df['publish_month'].astype(str) + '-' + 
            self.df['publish_day'].astype(str)
        )
        
        # Create day of week
        self.df['day_of_week'] = self.df['publish_date'].dt.day_name()
        
        # Calculate total engagement
        self.df['total_engagement'] = self.df['likes'] + self.df['shares'] + self.df['comments_added']
        
        # Calculate engagement rates
        self.df['engagement_rate'] = self.df['total_engagement'] / self.df['views'].replace(0, np.nan)
        self.df['like_rate'] = self.df['likes'] / self.df['views'].replace(0, np.nan)
        self.df['ctr'] = self.df['impressions_ctr']
        
        # Handle missing subscriber columns (add them if they don't exist)
        if 'subscribers_gained' not in self.df.columns:
            self.df['subscribers_gained'] = 0
        if 'subscribers_lost' not in self.df.columns:
            self.df['subscribers_lost'] = 0
            
        # Calculate subscriber conversion
        self.df['net_subscribers'] = self.df['subscribers_gained'] - self.df['subscribers_lost']
        self.df['subscriber_conversion'] = self.df['net_subscribers'] / self.df['views'].replace(0, np.nan)
        
        # Convert duration from TEXT to numeric (assuming it's in seconds)
        self.df['duration_numeric'] = pd.to_numeric(self.df['duration'], errors='coerce')
        self.df['duration_minutes'] = self.df['duration_numeric'] / 60
        
        # Convert numeric columns that are stored as objects
        numeric_columns = ['avg_views_per_viewer', 'new_viewers', 'subscribers_gained', 'subscribers_lost']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        # Create year_month for visualizations
        self.df['year_month'] = self.df['publish_date'].dt.to_period('M')
        
        # Content type classification
        def classify_content_type(title):
            title_lower = title.lower()
            if 'official music video' in title_lower:
                return 'Official Music Video'
            elif 'official lyric' in title_lower or 'lyric video' in title_lower or 'lyric visualizer' in title_lower:
                return 'Lyric Video'
            elif 'instrumental' in title_lower or 'karaoke' in title_lower:
                return 'Instrumental/Karaoke'
            elif 'official audio' in title_lower:
                return 'Official Audio'
            elif 'playthrough' in title_lower or 'chords' in title_lower or 'tabs' in title_lower:
                return 'Tutorial/Playthrough'
            elif 'live' in title_lower:
                return 'Live Performance'
            elif 'bts' in title_lower:
                return 'Behind The Scenes'
            else:
                return 'Other'
        
        self.df['content_type'] = self.df['video_title'].apply(classify_content_type)
        
        print("Data preparation completed successfully")
        return True
    
    def generate_overview_stats(self):
        """Generate overview statistics"""
        if self.df is None:
            return None
            
        overview = {
            'total_videos': len(self.df),
            'total_views': self.df['views'].sum(),
            'total_engagement': self.df['total_engagement'].sum(),
            'avg_views_per_video': self.df['views'].mean(),
            'median_views': self.df['views'].median(),
            'total_watch_time_hours': self.df['watch_time_hours'].sum(),
            'avg_engagement_rate': self.df['engagement_rate'].mean() * 100,
            'date_range': f"{self.df['publish_date'].min().strftime('%Y-%m-%d')} to {self.df['publish_date'].max().strftime('%Y-%m-%d')}"
        }
        
        # Add subscriber stats only if columns exist
        if 'subscribers_gained' in self.df.columns:
            overview.update({
                'total_subscribers_gained': self.df['subscribers_gained'].sum(),
                'total_subscribers_lost': self.df['subscribers_lost'].sum(),
                'net_subscribers': self.df['net_subscribers'].sum()
            })
        
        return overview
    
    def content_performance_by_category(self, top_n=10):
        """Analyze performance by content category"""
        agg_dict = {
            'views': ['count', 'sum', 'mean', 'median'],
            'total_engagement': ['sum', 'mean', 'median'],
            'engagement_rate': ['mean', 'median'],
            'watch_time_hours': ['sum', 'mean']
        }
        
        # Add subscriber stats only if columns exist
        if 'subscribers_gained' in self.df.columns:
            agg_dict['subscribers_gained'] = ['sum', 'mean']
            
        category_stats = self.df.groupby('category').agg(agg_dict).round(2)
        
        # Flatten column names
        category_stats.columns = ['_'.join(col) for col in category_stats.columns]
        category_stats = category_stats.sort_values('views_sum', ascending=False).head(top_n)
        
        return category_stats
    
    def content_type_analysis(self):
        """Analyze performance by content type"""
        agg_dict = {
            'views': ['count', 'sum', 'mean', 'median'],
            'total_engagement': ['sum', 'mean'],
            'engagement_rate': ['mean'],
            'ctr': ['mean']
        }
        
        # Add subscriber stats only if columns exist
        if 'subscribers_gained' in self.df.columns:
            agg_dict['subscribers_gained'] = ['sum', 'mean']
            
        type_stats = self.df.groupby('content_type').agg(agg_dict).round(4)
        
        type_stats.columns = ['_'.join(col) for col in type_stats.columns]
        type_stats = type_stats.sort_values('views_sum', ascending=False)
        
        return type_stats
    
    def top_performing_videos(self, metric='views', top_n=10):
        """Get top performing videos by specified metric"""
        columns = ['video_title', 'category', 'content_type', 'views', 'total_engagement', 
                  'engagement_rate', 'publish_date']
        
        # Add subscriber column if it exists
        if 'subscribers_gained' in self.df.columns:
            columns.insert(-1, 'subscribers_gained')
            
        top_videos = self.df.nlargest(top_n, metric)[columns]
        return top_videos
    
    def duration_analysis(self):
        """Analyze performance by video duration"""
        # Create duration bins
        self.df['duration_bin'] = pd.cut(self.df['duration_minutes'], 
                                        bins=[0, 2, 3, 4, 5, 6, float('inf')], 
                                        labels=['<2min', '2-3min', '3-4min', '4-5min', '5-6min', '>6min'])
        
        agg_dict = {
            'views': ['count', 'mean', 'median'],
            'total_engagement': ['mean', 'median'],
            'engagement_rate': ['mean', 'median'],
            'ctr': ['mean']
        }
        
        # Add subscriber stats only if columns exist
        if 'subscribers_gained' in self.df.columns:
            agg_dict['subscribers_gained'] = ['mean', 'median']
            
        duration_stats = self.df.groupby('duration_bin').agg(agg_dict).round(4)
        
        duration_stats.columns = ['_'.join(col) for col in duration_stats.columns]
        
        return duration_stats
    
    def posting_day_analysis(self):
        """Analyze performance by posting day (with caveats)"""
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        agg_dict = {
            'views': ['count', 'mean', 'median'],
            'total_engagement': ['mean', 'median'],
            'engagement_rate': ['mean', 'median']
        }
        
        # Add subscriber stats only if columns exist
        if 'subscribers_gained' in self.df.columns:
            agg_dict['subscribers_gained'] = ['mean', 'median']
            
        day_stats = self.df.groupby('day_of_week').agg(agg_dict).round(2)
        
        day_stats.columns = ['_'.join(col) for col in day_stats.columns]
        day_stats = day_stats.reindex(day_order)
        
        return day_stats
    
    def engagement_quality_analysis(self):
        """Analyze engagement quality metrics"""
        # Calculate percentiles for context
        engagement_percentiles = self.df['engagement_rate'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])
        
        # High engagement videos (top 10%)
        high_engagement = self.df[self.df['engagement_rate'] >= self.df['engagement_rate'].quantile(0.9)]
        
        quality_stats = {
            'engagement_percentiles': engagement_percentiles,
            'high_engagement_videos': len(high_engagement),
            'avg_like_rate': self.df['like_rate'].mean(),
            'avg_ctr': self.df['ctr'].mean(),
            'top_categories_high_engagement': high_engagement['category'].value_counts().head(),
            'top_content_types_high_engagement': high_engagement['content_type'].value_counts().head()
        }
        
        return quality_stats
    
    def monthly_performance_trends(self):
        """Analyze performance trends over time"""
        self.df['year_month'] = self.df['publish_date'].dt.to_period('M')
        
        agg_dict = {
            'views': ['count', 'sum', 'mean'],
            'total_engagement': ['sum', 'mean'],
            'watch_time_hours': ['sum']
        }
        
        # Add subscriber stats only if columns exist
        if 'subscribers_gained' in self.df.columns:
            agg_dict['subscribers_gained'] = ['sum', 'mean']
            
        monthly_stats = self.df.groupby('year_month').agg(agg_dict).round(2)
        
        monthly_stats.columns = ['_'.join(col) for col in monthly_stats.columns]
        monthly_stats.index = monthly_stats.index.astype(str)
        
        return monthly_stats
    
    def create_visualizations(self):
        """Create key visualizations as individual graphs"""
        if self.df is None:
            print("No data available for visualization")
            return
        
        print("Creating individual visualization charts...")
        
        # 1. Top Categories by Views
        self.plot_top_categories()
        
        # 2. Content Type Performance
        self.plot_content_type_performance()
        
        # 3. Engagement Rate Distribution
        self.plot_engagement_distribution()
        
        # 4. Views vs Duration
        self.plot_views_vs_duration()
        
        # 5. Top Videos Performance
        self.plot_top_videos()
        
        # 6. Posting Day Analysis
        self.plot_posting_day_analysis()
        
        # 7. Duration vs Engagement
        self.plot_duration_engagement()
        
        # 8. Monthly Trends
        self.plot_monthly_trends()
        
        # 9. Content Type Distribution
        self.plot_content_distribution()
        
        # 10. Category Performance Matrix
        self.plot_category_matrix()
    
    def plot_top_categories(self):
        """Plot top categories by total views"""
        plt.figure(figsize=(12, 8))
        top_categories = self.df.groupby('category')['views'].sum().sort_values(ascending=False).head(10)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_categories)))
        bars = plt.bar(range(len(top_categories)), top_categories.values, color=colors)
        
        plt.title('Top 10 Categories by Total Views', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(range(len(top_categories)), top_categories.index, rotation=45, ha='right')
        plt.ylabel('Total Views', fontsize=12)
        plt.xlabel('Category', fontsize=12)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_content_type_performance(self):
        """Plot content type performance as two separate charts (no subplots)."""

        # --- Figure 1: Average views by content type ---
        plt.figure(figsize=(12, 6))
        content_type_views = self.df.groupby('content_type')['views'].mean().sort_values(ascending=False)
        colors1 = plt.cm.Set3(range(len(content_type_views)))
        bars1 = plt.bar(range(len(content_type_views)), content_type_views.values, color=colors1)
        plt.title('Average Views by Content Type', fontsize=14, fontweight='bold')
        plt.xticks(range(len(content_type_views)), content_type_views.index, rotation=45, ha='right')
        plt.ylabel('Average Views')
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K')
        )
        for bar in bars1:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., h + (h*0.01 if h else 1),
                     f'{h/1e6:.1f}M' if h >= 1e6 else f'{h/1e3:.0f}K',
                     ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.show()

        # --- Figure 2: Average engagement rate by content type ---
        plt.figure(figsize=(12, 6))
        content_type_engagement = self.df.groupby('content_type')['engagement_rate'].mean().sort_values(ascending=False)
        colors2 = plt.cm.plasma(np.linspace(0, 1, len(content_type_engagement)))
        bars2 = plt.bar(range(len(content_type_engagement)), content_type_engagement.values * 100, color=colors2)
        plt.title('Average Engagement Rate by Content Type', fontsize=14, fontweight='bold')
        plt.xticks(range(len(content_type_engagement)), content_type_engagement.index, rotation=45, ha='right')
        plt.ylabel('Engagement Rate (%)')
        for bar in bars2:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., h + 0.1, f'{h:.2f}%', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.show()
    
    def plot_engagement_distribution(self):
        """Plot engagement rate distribution"""
        plt.figure(figsize=(12, 6))
        
        valid_engagement = self.df['engagement_rate'].dropna() * 100
        
        plt.hist(valid_engagement, bins=25, alpha=0.7, color='lightgreen', edgecolor='black', density=True)
        plt.title('Engagement Rate Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Engagement Rate (%)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        
        # Add statistics lines
        mean_eng = valid_engagement.mean()
        median_eng = valid_engagement.median()
        
        plt.axvline(mean_eng, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_eng:.2f}%')
        plt.axvline(median_eng, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_eng:.2f}%')
        
        # Add percentile lines
        p75 = valid_engagement.quantile(0.75)
        p90 = valid_engagement.quantile(0.90)
        plt.axvline(p75, color='orange', linestyle=':', linewidth=2, label=f'75th percentile: {p75:.2f}%')
        plt.axvline(p90, color='purple', linestyle=':', linewidth=2, label=f'90th percentile: {p90:.2f}%')
        
        plt.legend(fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_views_vs_duration(self):
        """Plot views vs duration scatter plot"""
        plt.figure(figsize=(12, 8))
        
        valid_duration = self.df['duration_minutes'].dropna()
        valid_views = self.df.loc[valid_duration.index, 'views']
        valid_engagement = self.df.loc[valid_duration.index, 'engagement_rate'] * 100
        
        # Create scatter plot with engagement rate as color
        scatter = plt.scatter(valid_duration, valid_views, c=valid_engagement, 
                             alpha=0.7, s=60, cmap='viridis', edgecolors='black', linewidth=0.5)
        
        plt.title('Views vs Video Duration (colored by Engagement Rate)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Duration (minutes)', fontsize=12)
        plt.ylabel('Views', fontsize=12)
        plt.yscale('log')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Engagement Rate (%)', fontsize=11)
        
        # Format y-axis
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K')
        )
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_top_videos(self):
        """Plot top performing videos"""
        plt.figure(figsize=(14, 10))
        
        top_videos = self.df.nlargest(15, 'views')
        video_labels = [title[:50] + '...' if len(title) > 50 else title for title in top_videos['video_title']]
        
        # Create color map based on category
        unique_cats = top_videos['category'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_cats)))
        color_map = dict(zip(unique_cats, colors))
        bar_colors = [color_map[cat] for cat in top_videos['category']]
        
        bars = plt.barh(range(len(video_labels)), top_videos['views']/1e6, color=bar_colors)
        
        plt.title('Top 15 Videos by Views', fontsize=16, fontweight='bold', pad=20)
        plt.yticks(range(len(video_labels)), video_labels, fontsize=9)
        plt.xlabel('Views (Millions)', fontsize=12)
        
        # Add value labels
        for i, (bar, views) in enumerate(zip(bars, top_videos['views'])):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{views/1e6:.1f}M', ha='left', va='center', fontsize=9)
        
        # Create legend for categories
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[cat], label=cat) 
                          for cat in unique_cats]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_posting_day_analysis(self):
        """Posting day analysis as two separate charts."""
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        stats = self.df.groupby('day_of_week').agg({
            'views': ['count', 'mean', 'median'],
            'total_engagement': ['mean', 'median']
        })
        stats.columns = ['_'.join(col) for col in stats.columns]
        stats = stats.reindex(day_order, fill_value=0)
        x = range(len(day_order))

        # --- Figure 1: Mean vs Median Engagement ---
        plt.figure(figsize=(12, 6))
        width = 0.4
        plt.bar([i - width/2 for i in x], stats['total_engagement_mean'], width, label='Mean Engagement', color='lightblue', alpha=0.85)
        plt.bar([i + width/2 for i in x], stats['total_engagement_median'], width, label='Median Engagement', color='steelblue', alpha=0.85)
        plt.title('POSTING DAY PERFORMANCE (Cumulative Performance, NOT Audience Activity)', fontsize=13, fontweight='bold')
        plt.xticks(x, day_order)
        plt.ylabel('Total Engagement')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

        # --- Figure 2: Video Count (bars) + Avg Views (line) ---
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()

        bars = ax.bar(x, stats['views_count'], width=0.6, color='orange', alpha=0.85, label='Video Count')
        line, = ax2.plot(x, stats['views_mean']/1e3, 'ro-', linewidth=2, markersize=6, label='Avg Views (K)')

        ax.set_title('Posting Day: Volume vs Average Views')
        ax.set_xticks(x)
        ax.set_xticklabels(day_order)
        ax.set_ylabel('Number of Videos', color='orange')
        ax2.set_ylabel('Average Views (thousands)', color='red')

        ax.legend([bars, line], ['Video Count', 'Avg Views (K)'], loc='upper left')
        plt.tight_layout()
        plt.show()
    
    def plot_duration_engagement(self):
        """Duration vs engagement as two separate charts."""
        self.df['duration_bin'] = pd.cut(
            self.df['duration_minutes'],
            bins=[0, 2, 3, 4, 5, 6, float('inf')],
            labels=['<2min','2-3min','3-4min','4-5min','5-6min','>6min']
        )
        d = self.df.groupby('duration_bin').agg({
            'views': ['count', 'mean'],
            'engagement_rate': 'mean',
            'total_engagement': 'mean'
        })
        d.columns = ['_'.join(col) for col in d.columns]
        d = d.dropna()
        x = range(len(d))

        # --- Figure 1: Video count (bars) + Avg Views (line) ---
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()

        bars = ax.bar(x, d['views_count'], color='lightcoral', alpha=0.85)
        line, = ax2.plot(x, d['views_mean']/1e3, 'bo-', linewidth=2, markersize=6)

        ax.set_title('Video Count vs Average Views by Duration', fontsize=14, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(d.index)
        ax.set_ylabel('Number of Videos', color='red')
        ax2.set_ylabel('Average Views (thousands)', color='blue')
        ax.legend([bars, line], ['Video Count', 'Avg Views (K)'], loc='upper left')
        plt.tight_layout()
        plt.show()

        # --- Figure 2: Avg Engagement Rate by Duration ---
        plt.figure(figsize=(12, 6))
        bars2 = plt.bar(x, d['engagement_rate_mean'] * 100, color=plt.cm.viridis(np.linspace(0, 1, len(d))), alpha=0.9)
        plt.title('Average Engagement Rate by Duration', fontsize=14, fontweight='bold')
        plt.xticks(x, d.index)
        plt.ylabel('Engagement Rate (%)')
        for b in bars2:
            h = b.get_height()
            plt.text(b.get_x() + b.get_width()/2., h + 0.05, f'{h:.2f}%', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_trends(self):
        """Monthly performance trends as two separate charts."""
        monthly_stats = self.df.groupby('year_month').agg({
            'views': ['count', 'sum', 'mean'],
            'total_engagement': ['sum', 'mean'],
            'engagement_rate': 'mean'
        }).round(2)
        
        monthly_stats.columns = ['_'.join(col) for col in monthly_stats.columns]
        monthly_stats.index = monthly_stats.index.astype(str)

        # --- Figure 1: Upload volume vs total views ---
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        l1, = ax.plot(monthly_stats.index, monthly_stats['views_count'], marker='o', linewidth=2, label='Videos Uploaded')
        l2, = ax2.plot(monthly_stats.index, monthly_stats['views_sum']/1e6, marker='s', linewidth=2, label='Total Views (M)', color='red')
        ax.set_title('Monthly Upload Volume vs Total Views', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Videos'); ax2.set_ylabel('Total Views (Millions)', color='red')
        ax.tick_params(axis='x', rotation=45)
        ax.legend([l1, l2], ['Videos Uploaded', 'Total Views (M)'], loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # --- Figure 2: Avg views vs Avg engagement rate ---
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        l3, = ax.plot(monthly_stats.index, monthly_stats['views_mean']/1e3, marker='o', linewidth=2, label='Avg Views (K)', color='green')
        l4, = ax2.plot(monthly_stats.index, monthly_stats['engagement_rate_mean']*100, marker='^', linewidth=2, label='Avg Engagement Rate (%)', color='purple')
        ax.set_title('Monthly Average Performance Quality', fontsize=16, fontweight='bold')
        ax.set_ylabel('Average Views (thousands)', color='green')
        ax2.set_ylabel('Engagement Rate (%)', color='purple')
        ax.tick_params(axis='x', rotation=45)
        ax.legend([l3, l4], ['Avg Views (K)', 'Avg Engagement Rate (%)'], loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_content_distribution(self):
        """Plot content type distribution pie chart"""
        plt.figure(figsize=(12, 8))
        
        content_dist = self.df['content_type'].value_counts()
        colors_pie = plt.cm.Set3(range(len(content_dist)))
        
        wedges, texts, autotexts = plt.pie(content_dist.values, labels=content_dist.index, 
                                          autopct='%1.1f%%', colors=colors_pie, startangle=90)
        
        plt.title('Content Type Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # Make text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
            
        for text in texts:
            text.set_fontsize(11)
        
        # Add count information
        legend_labels = [f'{label}: {count} videos' for label, count in content_dist.items()]
        plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def plot_category_matrix(self):
        """Plot category performance matrix"""
        plt.figure(figsize=(14, 10))
        
        # Get top categories performance metrics
        cat_stats = self.df.groupby('category').agg({
            'views': ['count', 'sum', 'mean'],
            'total_engagement': 'mean',
            'engagement_rate': 'mean',
            'watch_time_hours': 'mean'
        }).round(2)
        
        cat_stats.columns = ['_'.join(col) for col in cat_stats.columns]
        cat_stats = cat_stats.sort_values('views_sum', ascending=False).head(10)
        
        # Create bubble chart: x=avg_views, y=engagement_rate, size=total_views
        x = cat_stats['views_mean'] / 1e3  # in thousands
        y = cat_stats['engagement_rate_mean'] * 100  # in percentage
        sizes = cat_stats['views_sum'] / 1e6 * 100  # scale for bubble size
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(cat_stats)))
        
        scatter = plt.scatter(x, y, s=sizes, c=colors, alpha=0.7, edgecolors='black', linewidth=1)
        
        plt.title('Category Performance Matrix\n(Bubble size = Total Views)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Average Views (thousands)', fontsize=12)
        plt.ylabel('Average Engagement Rate (%)', fontsize=12)
        
        # Add category labels
        for i, cat in enumerate(cat_stats.index):
            plt.annotate(cat, (x.iloc[i], y.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, fontweight='bold')
        
        # Add grid and average lines
        plt.grid(True, alpha=0.3)
        plt.axhline(y.mean(), color='red', linestyle='--', alpha=0.7, label=f'Avg Engagement: {y.mean():.2f}%')
        plt.axvline(x.mean(), color='blue', linestyle='--', alpha=0.7, label=f'Avg Views: {x.mean():.0f}K')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def create_additional_charts(self):
        """This method is now integrated into individual chart functions"""
        pass
    
    def create_trends_chart(self):
        """Create time-based trends chart (kept for compatibility, already covered in plot_monthly_trends)."""
        # Same logic as plot_monthly_trends but kept as one call site if you still use it elsewhere.
        monthly_stats = self.df.groupby('year_month').agg({
            'views': ['count', 'sum', 'mean'],
            'total_engagement': ['sum', 'mean'],
            'engagement_rate': 'mean'
        }).round(2)
        monthly_stats.columns = ['_'.join(col) for col in monthly_stats.columns]
        monthly_stats.index = monthly_stats.index.astype(str)

        # Figure 1
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        l1, = ax.plot(monthly_stats.index, monthly_stats['views_count'], marker='o', linewidth=2, label='Videos Uploaded')
        l2, = ax2.plot(monthly_stats.index, monthly_stats['views_sum']/1e6, marker='s', linewidth=2, label='Total Views (M)', color='red')
        ax.set_title('Monthly Upload Volume vs Total Views')
        ax.set_ylabel('Number of Videos'); ax2.set_ylabel('Total Views (Millions)', color='red')
        ax.tick_params(axis='x', rotation=45)
        ax.legend([l1, l2], ['Videos Uploaded', 'Total Views (M)'], loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Figure 2
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        l3, = ax.plot(monthly_stats.index, monthly_stats['views_mean']/1e3, marker='o', linewidth=2, label='Avg Views (K)', color='green')
        l4, = ax2.plot(monthly_stats.index, monthly_stats['engagement_rate_mean']*100, marker='^', linewidth=2, label='Avg Engagement Rate (%)', color='purple')
        ax.set_title('Monthly Average Performance Quality')
        ax.set_ylabel('Average Views (thousands)', color='green')
        ax2.set_ylabel('Engagement Rate (%)', color='purple')
        ax.tick_params(axis='x', rotation=45)
        ax.legend([l3, l4], ['Avg Views (K)', 'Avg Engagement Rate (%)'], loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_full_report(self):
        """Generate comprehensive descriptive analytics report"""
        print("=== YOUTUBE CHANNEL DESCRIPTIVE ANALYTICS REPORT ===\n")
        
        # Overview Statistics
        overview = self.generate_overview_stats()
        print("📊 CHANNEL OVERVIEW")
        print("=" * 50)
        print(f"Total Videos: {overview['total_videos']:,}")
        print(f"Total Views: {overview['total_views']:,}")
        print(f"Total Engagement: {overview['total_engagement']:,}")
        print(f"Average Views per Video: {overview['avg_views_per_video']:,.0f}")
        print(f"Median Views: {overview['median_views']:,.0f}")
        print(f"Total Watch Time: {overview['total_watch_time_hours']:,.0f} hours")
        print(f"Average Engagement Rate: {overview['avg_engagement_rate']:.2f}%")
        print(f"Data Range: {overview['date_range']}")
        
        # Add subscriber info if available
        if 'net_subscribers' in overview:
            print(f"Net Subscribers Gained: {overview['net_subscribers']:,}")
        print()
        
        # Top Categories
        print("🎵 TOP PERFORMING CATEGORIES")
        print("=" * 50)
        category_stats = self.content_performance_by_category()
        print(category_stats[['views_count', 'views_sum', 'views_mean', 'total_engagement_mean']])
        print()
        
        # Content Type Analysis
        print("📹 CONTENT TYPE PERFORMANCE")
        print("=" * 50)
        type_stats = self.content_type_analysis()
        print(type_stats[['views_count', 'views_mean', 'engagement_rate_mean', 'ctr_mean']])
        print()
        
        # Top Videos
        print("🏆 TOP 10 VIDEOS BY VIEWS")
        print("=" * 50)
        top_videos = self.top_performing_videos()
        print(top_videos[['video_title', 'category', 'views', 'total_engagement']].to_string(index=False))
        print()
        
        # Duration Analysis
        print("⏱️ DURATION ANALYSIS")
        print("=" * 50)
        duration_stats = self.duration_analysis()
        print(duration_stats[['views_count', 'views_mean', 'engagement_rate_mean']])
        print()
        
        # Posting Day Analysis (with disclaimer)
        print("📅 POSTING DAY ANALYSIS")
        print("=" * 50)
        print("⚠️  IMPORTANT: This shows cumulative performance by posting day, not when audiences are most active")
        day_stats = self.posting_day_analysis()
        print(day_stats[['views_count', 'total_engagement_mean', 'total_engagement_median']])
        print()
        
        # Engagement Quality
        print("💡 ENGAGEMENT QUALITY INSIGHTS")
        print("=" * 50)
        quality_stats = self.engagement_quality_analysis()
        print(f"Average Like Rate: {quality_stats['avg_like_rate']:.4f}")
        print(f"Average CTR: {quality_stats['avg_ctr']:.4f}")
        print(f"High Engagement Videos (top 10%): {quality_stats['high_engagement_videos']}")
        print("\nTop Categories for High Engagement:")
        print(quality_stats['top_categories_high_engagement'])
        print("\nTop Content Types for High Engagement:")
        print(quality_stats['top_content_types_high_engagement'])

# Usage Example
def main():
    # Initialize analytics class
    analytics = YouTubeDescriptiveAnalytics(db_params)
    
    # Load and prepare data
    if analytics.connect_and_load_data():
        analytics.prepare_data()
        
        # Generate full report
        analytics.generate_full_report()
        
        # Create visualizations
        analytics.create_visualizations()
        
        # You can also access individual analysis methods as needed.

if __name__ == "__main__":
    main()
