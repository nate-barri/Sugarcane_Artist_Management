######################################################################
# TOTAL ENGAGEMENT FORECAST PREDICTOR
# Predicts aggregate total engagement (likes + comments + shares)
# for existing catalog over 6 months
######################################################################

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
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


class TotalEngagementPredictor:
    """
    Predicts aggregate total engagement growth for entire catalog.
    Single unified model for all engagement (likes + comments + shares).
    """

    def __init__(self, db_params):
        self.db_params = db_params
        self.df = None
        self.model = None
        self.scaler = None
        self.encoder = None

    def load_data(self):
        """Load and prepare data."""
        try:
            conn = psycopg2.connect(**self.db_params)
            query = "SELECT * FROM yt_video_etl"
            self.df = pd.read_sql(query, conn)
            conn.close()
            print(f"✓ Data loaded: {len(self.df)} videos")
            
            self._prepare_features()
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    def _prepare_features(self):
        """Feature engineering."""
        # Date parsing
        for col in ['publish_year', 'publish_month', 'publish_day']:
            if col not in self.df.columns:
                self.df[col] = datetime.now().year

        self.df['publish_date'] = pd.to_datetime(
            self.df['publish_year'].astype(str) + '-' +
            self.df['publish_month'].astype(str) + '-' +
            self.df['publish_day'].astype(str), 
            errors='coerce'
        )

        # Numeric columns
        numeric_cols = ['duration', 'impressions', 'impressions_ctr', 'views', 
                       'likes', 'shares', 'comments_added']
        for col in numeric_cols:
            if col not in self.df.columns:
                self.df[col] = 0
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        # Total engagement
        self.df['total_engagement'] = (
            self.df['likes'] + self.df['shares'] + self.df['comments_added']
        )
        self.df['engagement_rate'] = (
            self.df['total_engagement'] / np.maximum(self.df['views'], 1)
        )
        
        self.df['duration_minutes'] = self.df['duration'] / 60.0

        # Time features
        now = datetime.now()
        self.df['day_of_week'] = self.df['publish_date'].dt.dayofweek.fillna(0).astype(int)
        self.df['month'] = self.df['publish_date'].dt.month.fillna(1).astype(int)
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        self.df['days_since_publish'] = (
            (now - self.df['publish_date']).dt.days.fillna(0).astype(int)
        )
        self.df['months_since_publish'] = self.df['days_since_publish'] / 30.44

        # Content type
        self.df['content_type'] = self.df['video_title'].apply(self._classify_content)

        # Log transforms
        self.df['log_impressions'] = np.log1p(self.df['impressions'])
        self.df['log_views'] = np.log1p(self.df['views'])
        self.df['log_total_engagement'] = np.log1p(self.df['total_engagement'])

        # Features
        self.df['ctr_x_log_imp'] = self.df['impressions_ctr'] * self.df['log_impressions']
        self.df['high_ctr'] = (self.df['impressions_ctr'] > 0.08).astype(int)

        # Cyclical encoding
        self.df['sin_month'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['cos_month'] = np.cos(2 * np.pi * self.df['month'] / 12)

        print(f"✓ Features engineered: {self.df.shape}")

    def _classify_content(self, title):
        """Content type classifier."""
        t = str(title).lower()
        if 'official music video' in t:
            return 'music_video'
        if 'lyric' in t or 'lyrics' in t:
            return 'lyrics'
        if 'live' in t:
            return 'live'
        return 'other'

    def train_model(self):
        """Train single Gradient Boosting model for total engagement."""
        sep = '=' * 70
        dash = '─' * 70
        print(f"\n{sep}")
        print("TRAINING TOTAL ENGAGEMENT MODEL")
        print(sep)

        mature_df = self.df[self.df['months_since_publish'] >= 6].copy()
        
        print(f"\n📊 Training Set:")
        print(f"   Videos (6+ months):          {len(mature_df)}")
        
        # Current engagement state
        total_likes = mature_df['likes'].sum()
        total_comments = mature_df['comments_added'].sum()
        total_shares = mature_df['shares'].sum()
        total_engagement = mature_df['total_engagement'].sum()
        
        print(f"\n📈 Current Catalog Engagement:")
        print(f"   Total Likes:                 {total_likes:>15,}")
        print(f"   Total Comments:              {total_comments:>15,}")
        print(f"   Total Shares:                {total_shares:>15,}")
        print(f"   Total Engagement:            {total_engagement:>15,}")
        
        # Features
        feature_columns = [
            'log_impressions', 'impressions_ctr', 'ctr_x_log_imp', 'log_views',
            'duration_minutes', 'high_ctr', 'engagement_rate',
            'is_weekend', 'sin_month', 'cos_month', 'content_type'
        ]

        # Encode content_type
        self.encoder = LabelEncoder()
        mature_df['content_type_encoded'] = self.encoder.fit_transform(
            mature_df['content_type']
        )

        feature_cols_final = [
            c if c != 'content_type' else 'content_type_encoded' 
            for c in feature_columns
        ]
        X = mature_df[feature_cols_final].copy()
        y = mature_df['log_total_engagement'].copy()
        
        X = X.fillna(X.median())

        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols_final, index=X.index)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42
        )

        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            min_samples_split=8,
            min_samples_leaf=5,
            subsample=0.7,
            max_features='sqrt',
            random_state=42,
            verbose=0
        )

        print(f"\n🤖 Training Gradient Boosting...")
        self.model.fit(X_train, y_train)

        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Back-transform
        y_train_actual = np.expm1(y_train)
        y_test_actual = np.expm1(y_test)
        y_pred_train_actual = np.expm1(y_pred_train)
        y_pred_test_actual = np.expm1(y_pred_test)

        # Calculate comprehensive metrics
        train_metrics = self._calculate_metrics(y_train_actual, y_pred_train_actual, y_train_actual)
        test_metrics = self._calculate_metrics(y_test_actual, y_pred_test_actual, y_train_actual)

        # Display metrics
        print(f"\n{dash}")
        print("📊 COMPREHENSIVE EVALUATION METRICS")
        print(dash)
        print(f"\n{'Metric':<25} {'Train':<18} {'Test':<18}")
        print(dash)
        
        # R² Score
        print(f"{'R² Score (%)':<25} {train_metrics['r2']*100:>17.2f}% {test_metrics['r2']*100:>17.2f}%")
        
        # MAE
        train_mae_pct = (train_metrics['mae'] / np.mean(y_train_actual)) * 100
        test_mae_pct = (test_metrics['mae'] / np.mean(y_test_actual)) * 100
        print(f"{'MAE (% of mean)':<25} {train_mae_pct:>17.2f}% {test_mae_pct:>17.2f}%")
        print(f"{'MAE (absolute)':<25} {train_metrics['mae']:>17,.0f} {test_metrics['mae']:>17,.0f}")
        
        # RMSE
        train_rmse_pct = (train_metrics['rmse'] / np.mean(y_train_actual)) * 100
        test_rmse_pct = (test_metrics['rmse'] / np.mean(y_test_actual)) * 100
        print(f"{'RMSE (% of mean)':<25} {train_rmse_pct:>17.2f}% {test_rmse_pct:>17.2f}%")
        print(f"{'RMSE (absolute)':<25} {train_metrics['rmse']:>17,.0f} {test_metrics['rmse']:>17,.0f}")
        
        # MAPE
        print(f"{'MAPE (%)':<25} {train_metrics['mape']:>17.2f}% {test_metrics['mape']:>17.2f}%")
        
        # Median APE
        print(f"{'Median APE (%)':<25} {train_metrics['median_ape']:>17.2f}% {test_metrics['median_ape']:>17.2f}%")
        
        # MASE
        if not np.isnan(test_metrics['mase']):
            improvement_train = (1 - train_metrics['mase']) * 100
            improvement_test = (1 - test_metrics['mase']) * 100
            print(f"{'MASE (ratio)':<25} {train_metrics['mase']:>17.3f} {test_metrics['mase']:>17.3f}")
            print(f"{'  Improvement over naive':<25} {improvement_train:>17.1f}% {improvement_test:>17.1f}%")

        # Overfitting check
        overfit = train_metrics['r2'] - test_metrics['r2']
        overfit_pct = abs(overfit) * 100
        print(f"\n{'Generalization':<25}")
        print(dash)
        print(f"{'Overfit Gap (%)':<25} {overfit_pct:>17.2f}%", end='')
        if abs(overfit) < 0.05:
            print(" ✓ Excellent")
        elif abs(overfit) < 0.15:
            print(" ⚡ Good")
        else:
            print(" ⚠️  High")

        print(f"\n✓ Model trained successfully")

        return train_metrics, test_metrics

    def _calculate_metrics(self, y_true, y_pred, y_train=None):
        """Calculate comprehensive evaluation metrics."""
        # R² Score
        r2 = r2_score(y_true, y_pred)
        
        # MAE
        mae = mean_absolute_error(y_true, y_pred)
        
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
        
        # Median APE
        median_ape = np.median(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
        
        # MASE
        if y_train is not None and len(y_train) > 1:
            naive_pred = np.mean(y_train)
            naive_errors = np.abs(y_true - naive_pred)
            scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
            mase = mae / scale if scale > 0 else np.nan
        else:
            mase = np.nan
        
        return {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'median_ape': median_ape,
            'mase': mase
        }

    def predict_all_videos(self):
        """Generate total engagement predictions for all videos."""
        print(f"\n{'='*70}")
        print("GENERATING TOTAL ENGAGEMENT PREDICTIONS")
        print('='*70)

        predictions = []
        for idx, video in self.df.iterrows():
            if video['impressions'] <= 0:
                continue

            # Build features
            feature_dict = {
                'log_impressions': video['log_impressions'],
                'impressions_ctr': video['impressions_ctr'],
                'ctr_x_log_imp': video['ctr_x_log_imp'],
                'log_views': video['log_views'],
                'duration_minutes': video['duration_minutes'],
                'high_ctr': video['high_ctr'],
                'engagement_rate': video['engagement_rate'],
                'is_weekend': video['is_weekend'],
                'sin_month': video['sin_month'],
                'cos_month': video['cos_month'],
            }

            # Encode content type
            if video['content_type'] in self.encoder.classes_:
                content_type_encoded = self.encoder.transform([video['content_type']])[0]
            else:
                content_type_encoded = 0
            feature_dict['content_type_encoded'] = content_type_encoded

            # Predict
            X_new = pd.DataFrame([feature_dict])
            X_new = X_new[self.scaler.feature_names_in_]
            X_new_scaled = self.scaler.transform(X_new)
            
            pred_log = self.model.predict(X_new_scaled)[0]
            pred = np.expm1(pred_log)
            pred = int(max(0, pred))

            predictions.append({
                'title': video['video_title'],
                'type': video['content_type'],
                'months_old': video['months_since_publish'],
                'actual_engagement': video['total_engagement'],
                'predicted_engagement': pred,
            })

        pred_df = pd.DataFrame(predictions)
        
        print(f"\n✓ Generated predictions for {len(pred_df)} videos")
        
        return pred_df

    def forecast_engagement_growth(self, pred_df):
        """Forecast 6-month aggregate total engagement growth."""
        sep = '=' * 70
        dash = '─' * 70
        print(f"\n{sep}")
        print("AGGREGATE TOTAL ENGAGEMENT 6-MONTH FORECAST")
        print(sep)
        
        # Current state
        total_engagement = pred_df['actual_engagement'].sum()
        
        # Get component breakdown
        total_likes = self.df['likes'].sum()
        total_comments = self.df['comments_added'].sum()
        total_shares = self.df['shares'].sum()
        
        print(f"\n📊 CURRENT ENGAGEMENT STATE:")
        print(f"   Total Engagement:            {total_engagement:>15,}")
        print(f"   Breakdown:")
        print(f"     • Likes:                   {total_likes:>15,} ({total_likes/total_engagement*100:>5.1f}%)")
        print(f"     • Comments:                {total_comments:>15,} ({total_comments/total_engagement*100:>5.1f}%)")
        print(f"     • Shares:                  {total_shares:>15,} ({total_shares/total_engagement*100:>5.1f}%)")
        
        # Segment by age
        mature_videos = pred_df[pred_df['months_old'] >= 6]
        growing_videos = pred_df[(pred_df['months_old'] >= 3) & (pred_df['months_old'] < 6)]
        new_videos = pred_df[pred_df['months_old'] < 3]
        
        print(f"\n📅 CATALOG SEGMENTATION:")
        print(f"   Mature (6+ months):          {len(mature_videos):>15} videos ({mature_videos['actual_engagement'].sum():>12,})")
        print(f"   Growing (3-6 months):        {len(growing_videos):>15} videos ({growing_videos['actual_engagement'].sum():>12,})")
        print(f"   New (<3 months):             {len(new_videos):>15} videos ({new_videos['actual_engagement'].sum():>12,})")
        
        # Calculate 6-month growth
        mature_growth_rate = 0.06
        mature_growth = mature_videos['actual_engagement'].sum() * mature_growth_rate
        
        growing_growth = 0
        for idx, video in growing_videos.iterrows():
            predicted_mature = video['predicted_engagement']
            current = video['actual_engagement']
            remaining = max(0, predicted_mature - current)
            growing_growth += remaining * 0.7
        
        new_growth = 0
        for idx, video in new_videos.iterrows():
            predicted_mature = video['predicted_engagement']
            current = video['actual_engagement']
            remaining = max(0, predicted_mature - current)
            new_growth += remaining * 0.85
        
        total_growth = mature_growth + growing_growth + new_growth
        projected_total = total_engagement + total_growth
        growth_pct = (total_growth / total_engagement) * 100
        
        print(f"\n🔮 6-MONTH GROWTH PROJECTION:")
        print(dash)
        print(f"{'Segment':<25} {'Current':>15} {'Growth':>15} {'% Growth':>10}")
        print(dash)
        print(f"{'Mature catalog':<25} {mature_videos['actual_engagement'].sum():>15,} {mature_growth:>15,.0f} {mature_growth_rate*100:>9.1f}%")
        print(f"{'Growing videos':<25} {growing_videos['actual_engagement'].sum():>15,} {growing_growth:>15,.0f} {(growing_growth/max(growing_videos['actual_engagement'].sum(),1))*100:>9.1f}%")
        print(f"{'New videos':<25} {new_videos['actual_engagement'].sum():>15,} {new_growth:>15,.0f} {(new_growth/max(new_videos['actual_engagement'].sum(),1))*100:>9.1f}%")
        print(dash)
        print(f"{'TOTAL':<25} {total_engagement:>15,} {total_growth:>15,.0f} {growth_pct:>9.1f}%")
        
        print(f"\n📈 PROJECTED 6-MONTH TOTAL:    {projected_total:>15,.0f}")
        
        # Scenario analysis
        conservative_total = total_engagement + (total_growth * 0.7)
        optimistic_total = total_engagement + (total_growth * 1.3)
        
        print(f"\n💡 SCENARIO ANALYSIS:")
        print(dash)
        print(f"   Conservative (70%):          {conservative_total:>15,.0f}")
        print(f"   Baseline (expected):         {projected_total:>15,.0f}")
        print(f"   Optimistic (130%):           {optimistic_total:>15,.0f}")
        
        # Engagement rate trends
        current_views = self.df['views'].sum()
        projected_views = current_views * 1.065  # From views forecast
        
        current_eng_rate = (total_engagement / current_views) * 100
        projected_eng_rate = (projected_total / projected_views) * 100
        
        print(f"\n📈 ENGAGEMENT RATE TRENDS:")
        print(dash)
        print(f"   Current engagement rate:     {current_eng_rate:>14.3f}%")
        print(f"   Projected engagement rate:   {projected_eng_rate:>14.3f}%")
        print(f"   Rate change:                 {projected_eng_rate - current_eng_rate:>+14.3f}%")
        
        # Projected breakdown
        projected_likes = total_likes * (1 + growth_pct/100)
        projected_comments = total_comments * (1 + growth_pct/100)
        projected_shares = total_shares * (1 + growth_pct/100)
        
        print(f"\n📊 PROJECTED BREAKDOWN (6 months):")
        print(dash)
        print(f"   Likes:        {total_likes:>12,} → {projected_likes:>12,.0f} (+{growth_pct:.1f}%)")
        print(f"   Comments:     {total_comments:>12,} → {projected_comments:>12,.0f} (+{growth_pct:.1f}%)")
        print(f"   Shares:       {total_shares:>12,} → {projected_shares:>12,.0f} (+{growth_pct:.1f}%)")
        
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"   • Total engagement growth:   {growth_pct:.1f}% over 6 months")
        print(f"   • Monthly growth rate:       {growth_pct/6:.2f}% per month")
        print(f"   • Engagement rate:           {'Declining' if projected_eng_rate < current_eng_rate else 'Stable/Growing'}")
        print(f"   • This excludes NEW releases (existing catalog only)")
        
        # Visualizations
        self.plot_engagement_forecast(total_engagement, projected_total, 
                                      conservative_total, optimistic_total,
                                      total_growth, pred_df)
        
        return {
            'current_total': int(total_engagement),
            'projected_total': int(projected_total),
            'total_growth': int(total_growth),
            'growth_pct': growth_pct,
            'projected_likes': int(projected_likes),
            'projected_comments': int(projected_comments),
            'projected_shares': int(projected_shares)
        }

    def plot_engagement_forecast(self, current, projected, conservative, 
                                 optimistic, growth, pred_df):
        """Visualize total engagement forecast."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Total Engagement 6-Month Forecast', fontsize=16, fontweight='bold')
            
            # Plot 1: Growth trajectory
            ax1 = axes[0, 0]
            months = np.arange(0, 7)
            baseline_line = np.linspace(current, projected, 7)
            conservative_line = np.linspace(current, conservative, 7)
            optimistic_line = np.linspace(current, optimistic, 7)
            
            ax1.plot(months, baseline_line, 'g-o', linewidth=2.5, markersize=8, label='Baseline')
            ax1.fill_between(months, conservative_line, optimistic_line, 
                            alpha=0.2, color='green', label='Range (±30%)')
            ax1.set_xlabel('Months from Now')
            ax1.set_ylabel('Total Engagement')
            ax1.set_title('6-Month Growth Trajectory')
            ax1.legend()
            ax1.grid(alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, p: f'{x/1e6:.2f}M' if x >= 1e6 else f'{x/1e3:.0f}K'
            ))
            
            # Plot 2: Scenario comparison
            ax2 = axes[0, 1]
            scenarios = ['Conservative\n(-30%)', 'Baseline', 'Optimistic\n(+30%)']
            values = [conservative, projected, optimistic]
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            
            bars = ax2.bar(scenarios, values, color=colors, edgecolor='k', alpha=0.7)
            ax2.set_ylabel('Total Engagement')
            ax2.set_title('Scenario Analysis')
            ax2.grid(alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                label = f'{height/1e6:.2f}M' if height >= 1e6 else f'{height/1e3:.0f}K'
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        label, ha='center', va='bottom', fontweight='bold')
            
            # Plot 3: Growth by segment
            ax3 = axes[1, 0]
            mature = pred_df[pred_df['months_old'] >= 6]['actual_engagement'].sum()
            growing = pred_df[(pred_df['months_old'] >= 3) & (pred_df['months_old'] < 6)]['actual_engagement'].sum()
            new = pred_df[pred_df['months_old'] < 3]['actual_engagement'].sum()
            
            segments = ['Mature\n(6+ mo)', 'Growing\n(3-6 mo)', 'New\n(<3 mo)']
            segment_current = [mature, growing, new]
            segment_projected = [mature * 1.06, growing * 1.25 if growing > 0 else 0, 
                               new * 1.5 if new > 0 else 0]
            
            x = np.arange(len(segments))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, segment_current, width, label='Current', 
                          color='steelblue', edgecolor='k')
            bars2 = ax3.bar(x + width/2, segment_projected, width, label='Projected', 
                          color='orange', edgecolor='k')
            
            ax3.set_ylabel('Total Engagement')
            ax3.set_title('Growth by Video Age Segment')
            ax3.set_xticks(x)
            ax3.set_xticklabels(segments)
            ax3.legend()
            ax3.grid(alpha=0.3, axis='y')
            
            # Plot 4: Monthly growth breakdown
            ax4 = axes[1, 1]
            monthly_growth = growth / 6
            months_labels = ['Mo 1', 'Mo 2', 'Mo 3', 'Mo 4', 'Mo 5', 'Mo 6']
            cumulative = np.cumsum([monthly_growth] * 6)
            
            bars = ax4.bar(months_labels, [monthly_growth] * 6, color='coral', 
                         edgecolor='k', alpha=0.7, label='Monthly')
            ax4_twin = ax4.twinx()
            ax4_twin.plot(months_labels, cumulative, 'g-o', linewidth=2.5, 
                         markersize=8, label='Cumulative')
            
            ax4.set_ylabel('Monthly Growth', color='coral')
            ax4_twin.set_ylabel('Cumulative Growth', color='green')
            ax4.set_title('Monthly Growth Breakdown')
            ax4.grid(alpha=0.3, axis='y')
            ax4.legend(loc='upper left')
            ax4_twin.legend(loc='upper right')
            
            plt.tight_layout()
            plt.show()
            
            print(f"\n✓ Engagement forecast visualizations generated")
            
        except Exception as e:
            print(f"\n✗ Could not generate plots: {e}")
            import traceback
            traceback.print_exc()

    def run_pipeline(self):
        """Execute complete total engagement prediction pipeline."""
        header = '#' * 70
        print(f"\n{header}")
        print('TOTAL ENGAGEMENT FORECAST PREDICTOR')
        print('Existing Videos Only - No New Releases')
        print(header)

        if not self.load_data():
            return False

        # Train model
        train_metrics, test_metrics = self.train_model()

        # Predict all videos
        pred_df = self.predict_all_videos()

        # Forecast growth
        forecast = self.forecast_engagement_growth(pred_df)

        print(f"\n{header}")
        print('✅ TOTAL ENGAGEMENT FORECAST COMPLETE')
        print(header)
        print(f"\n💡 MODEL PERFORMANCE:")
        print(f"   Test R²:             {test_metrics['r2']*100:>6.2f}%")
        print(f"   Test MAE:            {test_metrics['mae']:>12,.0f}")
        print(f"   Test Median APE:     {test_metrics['median_ape']:>6.2f}%")
        print(f"   MASE Improvement:    {(1-test_metrics['mase'])*100:>6.1f}%")
        
        print(f"\n💡 ENGAGEMENT FORECAST SUMMARY:")
        print(f"   Current total:       {forecast['current_total']:>12,}")
        print(f"   6-month projection:  {forecast['projected_total']:>12,}")
        print(f"   Expected growth:     {forecast['total_growth']:>12,} (+{forecast['growth_pct']:.1f}%)")
        print(f"   Monthly growth rate: {forecast['growth_pct']/6:>12.2f}% per month")
        
        print(f"\n   Projected Breakdown:")
        print(f"   • Likes:             {forecast['projected_likes']:>12,}")
        print(f"   • Comments:          {forecast['projected_comments']:>12,}")
        print(f"   • Shares:            {forecast['projected_shares']:>12,}")
        
        print(f"\n   📌 This is CATALOG ONLY (no new releases included)")
        print(f"{header}\n")
        
        return True


if __name__ == '__main__':
    predictor = TotalEngagementPredictor(db_params)
    predictor.run_pipeline()