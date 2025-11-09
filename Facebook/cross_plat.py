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
    
    def fetch_facebook_data(self):
        """Fetch Facebook data with NaN handling"""
        query = """
        SELECT 
            'Facebook' as platform,
            COUNT(*) as total_posts,
            SUM(CASE WHEN reach = 'NaN'::numeric THEN 0 ELSE COALESCE(reach, 0) END) as total_reach,
            SUM(CASE WHEN shares = 'NaN'::numeric THEN 0 ELSE COALESCE(shares, 0) END) as total_shares,
            SUM(CASE WHEN comments = 'NaN'::numeric THEN 0 ELSE COALESCE(comments, 0) END) as total_comments,
            SUM(CASE WHEN reactions = 'NaN'::numeric THEN 0 ELSE COALESCE(reactions, 0) END) as total_reactions,
            SUM(CASE WHEN impressions = 'NaN'::numeric THEN 0 ELSE COALESCE(impressions, 0) END) as total_impressions
        FROM facebook_data_set;
        """
        try:
            df = pd.read_sql(query, self.conn)
            df = df.fillna(0)
            print("✓ Facebook data fetched")
            return df
        except Exception as e:
            print(f"✗ Facebook query failed: {e}")
            return pd.DataFrame()
    
    def fetch_tiktok_data(self):
        """Fetch TikTok data"""
        query = """
        SELECT 
            'TikTok' as platform,
            COUNT(*) as total_posts,
            COALESCE(SUM(views), 0) as total_views,
            COALESCE(SUM(likes), 0) as total_likes,
            COALESCE(SUM(shares), 0) as total_shares,
            COALESCE(SUM(comments_added), 0) as total_comments,
            COALESCE(SUM(saves), 0) as total_saves
        FROM tt_video_etl;
        """
        try:
            df = pd.read_sql(query, self.conn)
            df = df.fillna(0)
            print("✓ TikTok data fetched")
            return df
        except Exception as e:
            print(f"✗ TikTok query failed: {e}")
            return pd.DataFrame()
    
    def fetch_youtube_data(self):
        """Fetch YouTube data"""
        query = """
        SELECT 
            'YouTube' as platform,
            COUNT(*) as total_posts,
            COALESCE(SUM(views), 0) as total_views,
            COALESCE(SUM(likes), 0) as total_likes,
            COALESCE(SUM(shares), 0) as total_shares,
            COALESCE(SUM(comments_added), 0) as total_comments,
            COALESCE(SUM(impressions), 0) as total_impressions
        FROM yt_video_etl;
        """
        try:
            df = pd.read_sql(query, self.conn)
            df = df.fillna(0)
            print("✓ YouTube data fetched")
            return df
        except Exception as e:
            print(f"✗ YouTube query failed: {e}")
            return pd.DataFrame()
    
    def calculate_engagement_metrics(self, fb_df, tt_df, yt_df):
        """Calculate engagement metrics for all platforms"""
        metrics = []
        
        # Facebook engagement
        if not fb_df.empty:
            fb_shares = fb_df['total_shares'].values[0]
            fb_comments = fb_df['total_comments'].values[0]
            fb_reactions = fb_df['total_reactions'].values[0]
            fb_reach = fb_df['total_reach'].values[0]
            fb_impressions = fb_df['total_impressions'].values[0]
            
            fb_engagement = fb_shares + fb_comments + fb_reactions
            fb_views = fb_reach if fb_reach > 0 else fb_impressions
            
            metrics.append({
                'platform': 'Facebook',
                'total_engagement': fb_engagement,
                'total_posts': fb_df['total_posts'].values[0],
                'views': fb_views if fb_views > 0 else 1
            })
        
        # TikTok engagement
        if not tt_df.empty:
            tt_likes = tt_df['total_likes'].values[0]
            tt_shares = tt_df['total_shares'].values[0]
            tt_comments = tt_df['total_comments'].values[0]
            tt_saves = tt_df['total_saves'].values[0]
            tt_views = tt_df['total_views'].values[0]
            
            tt_engagement = tt_likes + tt_shares + tt_comments + tt_saves
            
            metrics.append({
                'platform': 'TikTok',
                'total_engagement': tt_engagement,
                'total_posts': tt_df['total_posts'].values[0],
                'views': tt_views if tt_views > 0 else 1
            })
        
        # YouTube engagement
        if not yt_df.empty:
            yt_likes = yt_df['total_likes'].values[0]
            yt_shares = yt_df['total_shares'].values[0]
            yt_comments = yt_df['total_comments'].values[0]
            yt_views = yt_df['total_views'].values[0]
            
            yt_engagement = yt_likes + yt_shares + yt_comments
            
            metrics.append({
                'platform': 'YouTube',
                'total_engagement': yt_engagement,
                'total_posts': yt_df['total_posts'].values[0],
                'views': yt_views if yt_views > 0 else 1
            })
        
        return pd.DataFrame(metrics)
    
    def create_visualizations(self, engagement_df):
        """Create comprehensive visualizations"""
        
        sns.set_style("whitegrid")
        
        # Pastel color palette
        pastel_orange = '#FFB347'  # Facebook
        pastel_pink = '#FF69B4'     # TikTok
        pastel_red = '#FF6B6B'      # YouTube
        
        colors = [pastel_orange, pastel_pink, pastel_red]
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main Pie Chart - Engagement Distribution (large, spans 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        wedges, texts, autotexts = ax1.pie(
            engagement_df['total_engagement'], 
            labels=engagement_df['platform'],
            colors=colors[:len(engagement_df)],
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
                      color=colors[:len(engagement_df)], alpha=0.8, edgecolor='black', linewidth=2)
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
        bars = ax3.bar(engagement_df['platform'], engagement_df['views'], 
                      color=colors[:len(engagement_df)], alpha=0.8, edgecolor='black', linewidth=2)
        ax3.set_title('Total Reach/Views by Platform', fontsize=14, weight='bold')
        ax3.set_ylabel('Reach/Views', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height/1000000):.1f}M' if height >= 1000000 else f'{int(height/1000):.0f}K',
                    ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 4. Engagement Rate
        ax4 = fig.add_subplot(gs[2, 0])
        engagement_rate = (engagement_df['total_engagement'] / engagement_df['views'] * 100)
        bars = ax4.barh(engagement_df['platform'], engagement_rate, 
                       color=colors[:len(engagement_df)], alpha=0.8, edgecolor='black', linewidth=2)
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
                      color=colors[:len(engagement_df)], alpha=0.8, edgecolor='black', linewidth=2)
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
        eng_per_post = engagement_df['total_engagement'] / engagement_df['total_posts']
        bars = ax6.bar(engagement_df['platform'], eng_per_post, 
                      color=colors[:len(engagement_df)], alpha=0.8, edgecolor='black', linewidth=2)
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
        
        # Create additional analysis charts
        self.create_additional_analysis(engagement_df)
    
    def generate_report(self, engagement_df):
        """Generate descriptive statistics report"""
        print("\n" + "="*80)
        print("CROSS-PLATFORM DESCRIPTIVE ANALYTICS REPORT")
        print("="*80)
        
        print("\n📊 ENGAGEMENT SUMMARY")
        print("-" * 80)
        for _, row in engagement_df.iterrows():
            eng_rate = (row['total_engagement'] / row['views'] * 100)
            eng_per_post = row['total_engagement'] / row['total_posts']
            print(f"\n{row['platform']}:")
            print(f"  • Total Posts: {int(row['total_posts']):,}")
            print(f"  • Total Engagement: {int(row['total_engagement']):,}")
            print(f"  • Total Reach/Views: {int(row['views']):,}")
            print(f"  • Engagement Rate: {eng_rate:.2f}%")
            print(f"  • Avg Engagement per Post: {int(eng_per_post):,}")
        
        print("\n" + "="*80)
        total_engagement = engagement_df['total_engagement'].sum()
        print(f"TOTAL CROSS-PLATFORM ENGAGEMENT: {int(total_engagement):,}")
        print("="*80)
        
        # Calculate percentages
        print("\n📈 PLATFORM ENGAGEMENT DISTRIBUTION")
        print("-" * 80)
        for _, row in engagement_df.iterrows():
            percentage = (row['total_engagement'] / total_engagement) * 100
            print(f"{row['platform']}: {percentage:.1f}%")
        
        print("\n" + "="*80 + "\n")
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        if not self.connect():
            return
        
        try:
            # Fetch data from all platforms
            print("\n🔄 Fetching data from databases...")
            fb_df = self.fetch_facebook_data()
            tt_df = self.fetch_tiktok_data()
            yt_df = self.fetch_youtube_data()
            
            # Calculate engagement metrics
            print("\n📊 Calculating engagement metrics...")
            engagement_df = self.calculate_engagement_metrics(fb_df, tt_df, yt_df)
            
            if engagement_df.empty:
                print("✗ No data to analyze")
                return
            
            # Generate report
            self.generate_report(engagement_df)
            
            # Create visualizations
            print("\n🎨 Creating visualizations...")
            self.create_visualizations(engagement_df)
            
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
    analytics = CrossPlatformAnalytics(db_params)
    analytics.run_analysis()