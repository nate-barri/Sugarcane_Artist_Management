import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from sqlalchemy import create_engine
import warnings

# Suppress SQLAlchemy warnings
warnings.filterwarnings('ignore', category=UserWarning)

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
        self.engine = None
        
    def connect(self):
        """Establish database connection using SQLAlchemy"""
        try:
            # Create SQLAlchemy engine
            connection_string = (
                f"postgresql://{self.db_params['user']}:{self.db_params['password']}"
                f"@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['dbname']}"
                f"?sslmode={self.db_params['sslmode']}"
            )
            self.engine = create_engine(connection_string)
            
            # Also keep psycopg2 connection for other operations
            self.conn = psycopg2.connect(**self.db_params)
            print("✓ Database connection established")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def fetch_facebook_data(self):
        """Fetch and aggregate Facebook data with detailed debugging"""
        query = """
        SELECT 
            'Facebook' as platform,
            COUNT(*) as total_posts,
            COALESCE(SUM(reach), 0) as total_reach,  
            COALESCE(SUM(shares), 0) as total_shares,
            COALESCE(SUM(comments), 0) as total_comments,
            COALESCE(SUM(reactions), 0) as total_reactions,
            COALESCE(SUM(impressions), 0) as total_impressions,
            COALESCE(AVG(reach), 0) as avg_reach,
            COALESCE(AVG(shares), 0) as avg_shares,
            COALESCE(AVG(comments), 0) as avg_comments,
            COALESCE(AVG(reactions), 0) as avg_reactions,
            MIN(reach) as min_reach,
            MAX(reach) as max_reach,
            COUNT(CASE WHEN reach > 0 THEN 1 END) as posts_with_reach
        FROM facebook_data_set
        WHERE (reach IS NOT NULL OR shares IS NOT NULL OR comments IS NOT NULL OR reactions IS NOT NULL);
        """
        try:
            df = pd.read_sql(query, self.engine)
            print("✓ Facebook data fetched")
            print(f"  Debug - Posts with reach: {df['posts_with_reach'].iloc[0] if not df.empty else 0}")
            print(f"  Debug - Min reach: {df['min_reach'].iloc[0] if not df.empty else 0}")
            print(f"  Debug - Max reach: {df['max_reach'].iloc[0] if not df.empty else 0}")
            return df
        except Exception as e:
            print(f"✗ Facebook query failed: {e}")
            return pd.DataFrame()
    
    def fetch_tiktok_data(self):
        """Fetch and aggregate TikTok data"""
        query = """
        SELECT 
            'TikTok' as platform,
            COUNT(*) as total_posts,
            COALESCE(SUM(views), 0) as total_views,
            COALESCE(SUM(likes), 0) as total_likes,
            COALESCE(SUM(shares), 0) as total_shares,
            COALESCE(SUM(comments_added), 0) as total_comments,
            COALESCE(SUM(saves), 0) as total_saves,
            COALESCE(AVG(views), 0) as avg_views,
            COALESCE(AVG(likes), 0) as avg_likes,
            COALESCE(AVG(shares), 0) as avg_shares,
            COALESCE(AVG(comments_added), 0) as avg_comments
        FROM tt_video_etl
        WHERE views IS NOT NULL;
        """
        try:
            df = pd.read_sql(query, self.engine)
            print("✓ TikTok data fetched")
            return df
        except Exception as e:
            print(f"✗ TikTok query failed: {e}")
            return pd.DataFrame()
    
    def fetch_youtube_data(self):
        """Fetch and aggregate YouTube data"""
        query = """
        SELECT 
            'YouTube' as platform,
            COUNT(*) as total_posts,
            COALESCE(SUM(views), 0) as total_views,
            COALESCE(SUM(likes), 0) as total_likes,
            COALESCE(SUM(shares), 0) as total_shares,
            COALESCE(SUM(comments_added), 0) as total_comments,
            COALESCE(SUM(impressions), 0) as total_impressions,
            COALESCE(SUM(watch_time_hours), 0) as total_watch_time,
            COALESCE(AVG(views), 0) as avg_views,
            COALESCE(AVG(likes), 0) as avg_likes,
            COALESCE(AVG(shares), 0) as avg_shares,
            COALESCE(AVG(comments_added), 0) as avg_comments
        FROM yt_video_etl
        WHERE views IS NOT NULL;
        """
        try:
            df = pd.read_sql(query, self.engine)
            print("✓ YouTube data fetched")
            return df
        except Exception as e:
            print(f"✗ YouTube query failed: {e}")
            return pd.DataFrame()
    
    def calculate_engagement_metrics(self, fb_df, tt_df, yt_df):
        """Calculate engagement metrics for all platforms"""
        print("\n📊 Calculating engagement metrics...")
        
        metrics = {}
        
        # Facebook metrics
        if not fb_df.empty:
            fb_engagement = (
                fb_df['total_shares'].iloc[0] + 
                fb_df['total_comments'].iloc[0] + 
                fb_df['total_reactions'].iloc[0]
            )
            metrics['facebook'] = {
                'total_posts': int(fb_df['total_posts'].iloc[0]),
                'total_engagement': int(fb_engagement),
                'total_reach': int(fb_df['total_reach'].iloc[0]),
                'engagement_rate': (fb_engagement / fb_df['total_reach'].iloc[0] * 100) 
                                   if fb_df['total_reach'].iloc[0] > 0 else 0,
                'avg_engagement': fb_engagement / fb_df['total_posts'].iloc[0] 
                                 if fb_df['total_posts'].iloc[0] > 0 else 0
            }
        
        # TikTok metrics
        if not tt_df.empty:
            tt_engagement = (
                tt_df['total_likes'].iloc[0] + 
                tt_df['total_shares'].iloc[0] + 
                tt_df['total_comments'].iloc[0] +
                tt_df['total_saves'].iloc[0]
            )
            metrics['tiktok'] = {
                'total_posts': int(tt_df['total_posts'].iloc[0]),
                'total_engagement': int(tt_engagement),
                'total_reach': int(tt_df['total_views'].iloc[0]),
                'engagement_rate': (tt_engagement / tt_df['total_views'].iloc[0] * 100) 
                                   if tt_df['total_views'].iloc[0] > 0 else 0,
                'avg_engagement': tt_engagement / tt_df['total_posts'].iloc[0] 
                                 if tt_df['total_posts'].iloc[0] > 0 else 0
            }
        
        # YouTube metrics
        if not yt_df.empty:
            yt_engagement = (
                yt_df['total_likes'].iloc[0] + 
                yt_df['total_shares'].iloc[0] + 
                yt_df['total_comments'].iloc[0]
            )
            metrics['youtube'] = {
                'total_posts': int(yt_df['total_posts'].iloc[0]),
                'total_engagement': int(yt_engagement),
                'total_reach': int(yt_df['total_views'].iloc[0]),
                'engagement_rate': (yt_engagement / yt_df['total_views'].iloc[0] * 100) 
                                   if yt_df['total_views'].iloc[0] > 0 else 0,
                'avg_engagement': yt_engagement / yt_df['total_posts'].iloc[0] 
                                 if yt_df['total_posts'].iloc[0] > 0 else 0
            }
        
        return metrics
    
    def print_report(self, metrics):
        """Print formatted analytics report"""
        print("\n" + "="*80)
        print("CROSS-PLATFORM DESCRIPTIVE ANALYTICS REPORT")
        print("="*80)
        print("\n📊 ENGAGEMENT SUMMARY")
        print("-"*80)
        
        total_engagement = 0
        
        for platform, data in metrics.items():
            platform_name = platform.capitalize()
            total_engagement += data['total_engagement']
            
            print(f"\n{platform_name}:")
            print(f"  • Total Posts: {data['total_posts']:,}")
            print(f"  • Total Engagement: {data['total_engagement']:,}")
            print(f"  • Total Reach/Views: {data['total_reach']:,}")
            print(f"  • Engagement Rate: {data['engagement_rate']:.2f}%")
            print(f"  • Avg Engagement per Post: {data['avg_engagement']:,.0f}")
        
        print("\n" + "="*80)
        print(f"TOTAL CROSS-PLATFORM ENGAGEMENT: {total_engagement:,}")
        print("="*80)
        
        # Distribution
        print("\n📈 PLATFORM ENGAGEMENT DISTRIBUTION")
        print("-"*80)
        
        if total_engagement > 0:
            for platform, data in metrics.items():
                percentage = (data['total_engagement'] / total_engagement) * 100
                print(f"{platform.capitalize()}: {percentage:.1f}%")
        else:
            print("⚠️  No engagement data available")
        
        print("="*80 + "\n")
    
    def create_visualizations(self, metrics):
        """Create comprehensive visualizations"""
        print("🎨 Creating visualizations...")
        
        # Define distinct colors
        colors = {
            'facebook': '#2E86AB',  # Blue
            'tiktok': '#A23B72',    # Purple
            'youtube': '#F18F01'    # Orange
        }
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Total Engagement by Platform (Bar Chart)
        ax1 = fig.add_subplot(gs[0, :2])
        platforms = [p.capitalize() for p in metrics.keys()]
        engagements = [metrics[p]['total_engagement'] for p in metrics.keys()]
        bars = ax1.bar(platforms, engagements, 
                       color=[colors[p] for p in metrics.keys()])
        ax1.set_title('Total Engagement by Platform', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Total Engagement', fontsize=11)
        ax1.ticklabel_format(style='plain', axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10)
        
        # 2. Engagement Distribution (Pie Chart)
        ax2 = fig.add_subplot(gs[0, 2])
        total_engagement = sum(engagements)
        
        if total_engagement > 0:
            # Filter out platforms with 0 engagement for cleaner pie chart
            non_zero_platforms = [p for i, p in enumerate(platforms) if engagements[i] > 0]
            non_zero_engagements = [e for e in engagements if e > 0]
            non_zero_colors = [colors[p.lower()] for p in non_zero_platforms]
            
            ax2.pie(non_zero_engagements, labels=non_zero_platforms, autopct='%1.1f%%',
                   colors=non_zero_colors, startangle=90)
            ax2.set_title('Engagement Distribution', fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
        
        # 3. Engagement Rate Comparison
        ax3 = fig.add_subplot(gs[1, :2])
        rates = [metrics[p]['engagement_rate'] for p in metrics.keys()]
        bars = ax3.bar(platforms, rates, 
                       color=[colors[p] for p in metrics.keys()])
        ax3.set_title('Engagement Rate by Platform', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Engagement Rate (%)', fontsize=11)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=10)
        
        # 4. Total Reach/Views Comparison
        ax4 = fig.add_subplot(gs[1, 2])
        reach_data = [metrics[p]['total_reach'] for p in metrics.keys()]
        bars = ax4.barh(platforms, reach_data,
                        color=[colors[p] for p in metrics.keys()])
        ax4.set_title('Total Reach/Views', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Reach/Views', fontsize=10)
        ax4.ticklabel_format(style='plain', axis='x')
        
        for bar in bars:
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width):,}',
                    ha='left', va='center', fontsize=9, style='italic')
        
        # 5. Posts per Platform
        ax5 = fig.add_subplot(gs[2, 0])
        posts = [metrics[p]['total_posts'] for p in metrics.keys()]
        ax5.bar(platforms, posts, color=[colors[p] for p in metrics.keys()])
        ax5.set_title('Total Posts by Platform', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Number of Posts', fontsize=10)
        
        for i, v in enumerate(posts):
            ax5.text(i, v, str(v), ha='center', va='bottom', fontsize=10)
        
        # 6. Average Engagement per Post
        ax6 = fig.add_subplot(gs[2, 1])
        avg_eng = [metrics[p]['avg_engagement'] for p in metrics.keys()]
        ax6.bar(platforms, avg_eng, color=[colors[p] for p in metrics.keys()])
        ax6.set_title('Avg Engagement per Post', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Avg Engagement', fontsize=10)
        
        for i, v in enumerate(avg_eng):
            ax6.text(i, v, f'{int(v):,}', ha='center', va='bottom', fontsize=9)
        
        # 7. Summary Stats Table
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('tight')
        ax7.axis('off')
        
        table_data = []
        for platform in metrics.keys():
            table_data.append([
                platform.capitalize(),
                f"{metrics[platform]['total_posts']:,}",
                f"{metrics[platform]['total_engagement']:,}",
                f"{metrics[platform]['engagement_rate']:.2f}%"
            ])
        
        table = ax7.table(cellText=table_data,
                         colLabels=['Platform', 'Posts', 'Engagement', 'Rate'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold')
        
        plt.suptitle('Cross-Platform Analytics Dashboard', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'cross_platform_analytics_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Visualizations saved as '{filename}'")
        
        plt.show()
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        if not self.connect():
            return
        
        print("\n🔄 Fetching data from databases...")
        fb_df = self.fetch_facebook_data()
        tt_df = self.fetch_tiktok_data()
        yt_df = self.fetch_youtube_data()
        
        metrics = self.calculate_engagement_metrics(fb_df, tt_df, yt_df)
        self.print_report(metrics)
        self.create_visualizations(metrics)
        
        if self.conn:
            self.conn.close()
        print("\n✓ Analysis complete!")

if __name__ == "__main__":
    analytics = CrossPlatformAnalytics(db_params)
    analytics.run_analysis()