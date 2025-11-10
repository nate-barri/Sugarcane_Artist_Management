import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Database connection parameters
db_params = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host': 'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port': '5432',
    'sslmode': 'require'
}

class CrossPlatformAnalytics:
    def __init__(self, db_params):
        self.db_params = db_params
        self.conn = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_params)
            print("✓ Database connection established")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def fetch_unified_data(self):
        """
        Fetch all data from unified table - MUCH SIMPLER!
        This replaces the 3 separate fetch methods
        """
        query = """
        SELECT 
            platform,
            COUNT(*) as total_posts,
            SUM(views_reach) as total_views,
            SUM(likes_reactions) as total_likes,
            SUM(shares) as total_shares,
            SUM(comments) as total_comments,
            SUM(saves) as total_saves,
            SUM(impressions) as total_impressions,
            SUM(total_engagement) as total_engagement,
            AVG(total_engagement) as avg_engagement_per_post,
            ROUND((SUM(total_engagement)::numeric / NULLIF(SUM(views_reach), 0) * 100), 2) as engagement_rate
        FROM unified_social_analytics
        GROUP BY platform
        ORDER BY total_engagement DESC;
        """
        try:
            df = pd.read_sql(query, self.conn)
            df = df.fillna(0)
            print("✓ Unified data fetched")
            return df
        except Exception as e:
            print(f"✗ Query failed: {e}")
            print("   Make sure you've created the unified_social_analytics view first!")
            return pd.DataFrame()
    
    def fetch_top_posts(self, limit=10):
        """Fetch top performing posts across all platforms"""
        query = f"""
        SELECT 
            platform,
            post_id,
            post_date,
            views_reach,
            total_engagement,
            ROUND((total_engagement::numeric / NULLIF(views_reach, 0) * 100), 2) as engagement_rate
        FROM unified_social_analytics
        ORDER BY total_engagement DESC
        LIMIT {limit};
        """
        try:
            df = pd.read_sql(query, self.conn)
            print("✓ Top posts fetched")
            return df
        except Exception as e:
            print(f"✗ Query failed: {e}")
            return pd.DataFrame()
    
    def fetch_time_series_data(self, days=30):
        """Fetch engagement trends over time"""
        query = f"""
        SELECT 
            platform,
            post_date::date as date,
            COUNT(*) as posts_count,
            SUM(total_engagement) as daily_engagement,
            SUM(views_reach) as daily_views
        FROM unified_social_analytics
        WHERE post_date IS NOT NULL 
            AND post_date >= CURRENT_DATE - INTERVAL '{days} days'
        GROUP BY platform, post_date::date
        ORDER BY date DESC;
        """
        try:
            df = pd.read_sql(query, self.conn)
            print("✓ Time series data fetched")
            return df
        except Exception as e:
            print(f"✗ Query failed: {e}")
            return pd.DataFrame()
    
    def fetch_engagement_breakdown(self):
        """Fetch detailed engagement breakdown by type"""
        query = """
        SELECT 
            platform,
            SUM(likes_reactions) as total_likes,
            SUM(shares) as total_shares,
            SUM(comments) as total_comments,
            SUM(saves) as total_saves
        FROM unified_social_analytics
        GROUP BY platform;
        """
        try:
            df = pd.read_sql(query, self.conn)
            print("✓ Engagement breakdown fetched")
            return df
        except Exception as e:
            print(f"✗ Query failed: {e}")
            return pd.DataFrame()
    
    def create_visualizations(self, engagement_df):
        """Create comprehensive visualizations"""
        
        sns.set_style("whitegrid")
        
        # Pastel color palette
        color_map = {
            'Facebook': '#FFB347',  # Pastel orange
            'TikTok': '#FF69B4',    # Pastel pink
            'YouTube': '#FF6B6B'    # Pastel red
        }
        colors = [color_map.get(platform, '#95B8D1') for platform in engagement_df['platform']]
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main Pie Chart - Engagement Distribution (large, spans 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        wedges, texts, autotexts = ax1.pie(
            engagement_df['total_engagement'], 
            labels=engagement_df['platform'],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 14, 'weight': 'bold'},
            explode=[0.05] * len(engagement_df)
        )
        ax1.set_title('Engagement Distribution Across Platforms', 
                     fontsize=18, weight='bold', pad=20)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
        
        # 2. Total Engagement Bar Chart
        ax2 = fig.add_subplot(gs[0, 2])
        bars = ax2.bar(engagement_df['platform'], engagement_df['total_engagement'], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax2.set_title('Total Engagement by Platform', fontsize=14, weight='bold')
        ax2.set_ylabel('Total Engagement', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 3. Total Views/Reach Comparison
        ax3 = fig.add_subplot(gs[1, 2])
        bars = ax3.bar(engagement_df['platform'], engagement_df['total_views'], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax3.set_title('Total Reach/Views by Platform', fontsize=14, weight='bold')
        ax3.set_ylabel('Reach/Views', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            if height >= 1000000:
                label = f'{int(height/1000000):.1f}M'
            elif height >= 1000:
                label = f'{int(height/1000):.0f}K'
            else:
                label = f'{int(height)}'
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 4. Engagement Rate
        ax4 = fig.add_subplot(gs[2, 0])
        bars = ax4.barh(engagement_df['platform'], engagement_df['engagement_rate'], 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax4.set_title('Engagement Rate (%)', fontsize=14, weight='bold')
        ax4.set_xlabel('Engagement Rate (%)', fontsize=12)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}%',
                    ha='left', va='center', fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # 5. Posts per Platform
        ax5 = fig.add_subplot(gs[2, 1])
        bars = ax5.bar(engagement_df['platform'], engagement_df['total_posts'], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax5.set_title('Total Posts by Platform', fontsize=14, weight='bold')
        ax5.set_ylabel('Number of Posts', fontsize=12)
        ax5.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 6. Engagement per Post
        ax6 = fig.add_subplot(gs[2, 2])
        bars = ax6.bar(engagement_df['platform'], engagement_df['avg_engagement_per_post'], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax6.set_title('Avg Engagement per Post', fontsize=14, weight='bold')
        ax6.set_ylabel('Engagement per Post', fontsize=12)
        ax6.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10, weight='bold')
        
        plt.suptitle('Cross-Platform Social Media Analytics Dashboard', 
                    fontsize=22, weight='bold', y=0.98)
        
        plt.tight_layout()
        print("✓ Main dashboard created")
        plt.show()
    
    def create_engagement_breakdown_chart(self, breakdown_df):
        """Create stacked bar chart for engagement breakdown"""
        if breakdown_df.empty:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data for stacked bar chart
        platforms = breakdown_df['platform']
        likes = breakdown_df['total_likes']
        shares = breakdown_df['total_shares']
        comments = breakdown_df['total_comments']
        saves = breakdown_df['total_saves']
        
        x = np.arange(len(platforms))
        width = 0.6
        
        # Create stacked bars
        p1 = ax.bar(x, likes, width, label='Likes/Reactions', color='#FFD700')
        p2 = ax.bar(x, shares, width, bottom=likes, label='Shares', color='#87CEEB')
        p3 = ax.bar(x, comments, width, bottom=likes+shares, label='Comments', color='#98FB98')
        p4 = ax.bar(x, saves, width, bottom=likes+shares+comments, label='Saves', color='#DDA0DD')
        
        ax.set_xlabel('Platform', fontsize=12, weight='bold')
        ax.set_ylabel('Total Engagement', fontsize=12, weight='bold')
        ax.set_title('Engagement Breakdown by Type', fontsize=16, weight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(platforms)
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        print("✓ Engagement breakdown chart created")
        plt.show()
    
    def create_time_series_chart(self, time_df):
        """Create time series engagement chart"""
        if time_df.empty:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Convert date column to datetime
        time_df['date'] = pd.to_datetime(time_df['date'])
        
        # Plot 1: Daily Engagement by Platform
        for platform in time_df['platform'].unique():
            platform_data = time_df[time_df['platform'] == platform]
            ax1.plot(platform_data['date'], platform_data['daily_engagement'], 
                    marker='o', label=platform, linewidth=2)
        
        ax1.set_xlabel('Date', fontsize=12, weight='bold')
        ax1.set_ylabel('Daily Engagement', fontsize=12, weight='bold')
        ax1.set_title('Daily Engagement Trends by Platform', fontsize=14, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Daily Posts Count
        for platform in time_df['platform'].unique():
            platform_data = time_df[time_df['platform'] == platform]
            ax2.plot(platform_data['date'], platform_data['posts_count'], 
                    marker='s', label=platform, linewidth=2)
        
        ax2.set_xlabel('Date', fontsize=12, weight='bold')
        ax2.set_ylabel('Posts Count', fontsize=12, weight='bold')
        ax2.set_title('Daily Posts Count by Platform', fontsize=14, weight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        print("✓ Time series chart created")
        plt.show()
    
    def generate_report(self, engagement_df, top_posts_df=None):
        """Generate descriptive statistics report"""
        print("\n" + "="*80)
        print("CROSS-PLATFORM DESCRIPTIVE ANALYTICS REPORT")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n📊 ENGAGEMENT SUMMARY")
        print("-" * 80)
        for _, row in engagement_df.iterrows():
            print(f"\n{row['platform']}:")
            print(f"  • Total Posts: {int(row['total_posts']):,}")
            print(f"  • Total Engagement: {int(row['total_engagement']):,}")
            print(f"  • Total Reach/Views: {int(row['total_views']):,}")
            print(f"  • Total Likes/Reactions: {int(row['total_likes']):,}")
            print(f"  • Total Shares: {int(row['total_shares']):,}")
            print(f"  • Total Comments: {int(row['total_comments']):,}")
            if row['total_saves'] > 0:
                print(f"  • Total Saves: {int(row['total_saves']):,}")
            print(f"  • Engagement Rate: {row['engagement_rate']:.2f}%")
            print(f"  • Avg Engagement per Post: {int(row['avg_engagement_per_post']):,}")
        
        print("\n" + "="*80)
        total_engagement = engagement_df['total_engagement'].sum()
        total_views = engagement_df['total_views'].sum()
        total_posts = engagement_df['total_posts'].sum()
        overall_rate = (total_engagement / total_views * 100) if total_views > 0 else 0
        
        print(f"TOTAL CROSS-PLATFORM METRICS:")
        print(f"  • Total Posts: {int(total_posts):,}")
        print(f"  • Total Engagement: {int(total_engagement):,}")
        print(f"  • Total Views: {int(total_views):,}")
        print(f"  • Overall Engagement Rate: {overall_rate:.2f}%")
        print("="*80)
        
        # Calculate percentages
        print("\n📈 PLATFORM ENGAGEMENT DISTRIBUTION")
        print("-" * 80)
        for _, row in engagement_df.iterrows():
            percentage = (row['total_engagement'] / total_engagement) * 100
            print(f"{row['platform']}: {percentage:.1f}% of total engagement")
        
        # Top posts
        if top_posts_df is not None and not top_posts_df.empty:
            print("\n🏆 TOP 10 POSTS ACROSS ALL PLATFORMS")
            print("-" * 80)
            for idx, row in top_posts_df.iterrows():
                print(f"{idx+1}. {row['platform']} - ID: {row['post_id']}")
                print(f"   Date: {row['post_date']}")
                print(f"   Views: {int(row['views_reach']):,} | Engagement: {int(row['total_engagement']):,} | Rate: {row['engagement_rate']:.2f}%")
                print()
        
        print("="*80 + "\n")
    
    def run_analysis(self, include_time_series=True):
        """Run complete analysis pipeline"""
        if not self.connect():
            return
        
        try:
            # Fetch unified data - MUCH SIMPLER NOW!
            print("\n🔄 Fetching data from unified table...")
            engagement_df = self.fetch_unified_data()
            
            if engagement_df.empty:
                print("✗ No data to analyze")
                print("   Make sure you've created the unified_social_analytics view!")
                return
            
            # Fetch additional data
            print("\n🔄 Fetching additional analytics...")
            top_posts_df = self.fetch_top_posts(10)
            breakdown_df = self.fetch_engagement_breakdown()
            
            if include_time_series:
                time_df = self.fetch_time_series_data(30)
            
            # Generate report
            self.generate_report(engagement_df, top_posts_df)
            
            # Create visualizations
            print("\n🎨 Creating visualizations...")
            self.create_visualizations(engagement_df)
            
            if not breakdown_df.empty:
                self.create_engagement_breakdown_chart(breakdown_df)
            
            if include_time_series and not time_df.empty:
                self.create_time_series_chart(time_df)
            
            print("\n✓ Analysis complete!")
            
        except Exception as e:
            print(f"\n✗ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.conn:
                self.conn.close()
                print("✓ Database connection closed")

# Run the analysis
if __name__ == "__main__":
    print("="*80)
    print("CROSS-PLATFORM SOCIAL MEDIA ANALYTICS")
    print("="*80)
    
    analytics = CrossPlatformAnalytics(db_params)
    analytics.run_analysis(include_time_series=True)