######################################################################
# AGGREGATE CATALOG GROWTH PREDICTOR (ENHANCED METRICS)
# Predicts total 6-month growth for ALL existing videos combined
# Focus: Catalog momentum without new releases
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


class AggregateCatalogPredictor:
    """
    Predicts aggregate 6-month growth for entire existing catalog.
    No individual video predictions - only total channel performance.
    """

    def __init__(self, db_params):
        self.db_params = db_params
        self.df = None
        self.model = None
        self.scaler = None
        self.encoder = None
        self.metrics = {}

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

        # Engagement
        self.df['engagement_rate'] = (
            (self.df['likes'] + self.df['shares'] + self.df['comments_added']) / 
            np.maximum(self.df['views'], 1)
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

    def calculate_comprehensive_metrics(self, y_true, y_pred, dataset_name=""):
        """Calculate R², MAE, RMSE, MAPE, MASE, Median APE."""
        # Remove any zero or negative values for percentage metrics
        mask = y_true > 0
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        # R² (Coefficient of Determination)
        r2 = r2_score(y_true, y_pred)
        
        # MAE (Mean Absolute Error)
        mae = mean_absolute_error(y_true, y_pred)
        
        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE (Mean Absolute Percentage Error)
        if len(y_true_filtered) > 0:
            mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
        else:
            mape = np.nan
        
        # Median APE (Median Absolute Percentage Error)
        if len(y_true_filtered) > 0:
            ape = np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered) * 100
            median_ape = np.median(ape)
        else:
            median_ape = np.nan
        
        # MASE (Mean Absolute Scaled Error)
        # Using naive forecast (persistence model) as baseline
        if len(y_true) > 1:
            naive_mae = np.mean(np.abs(np.diff(y_true)))
            mase = mae / naive_mae if naive_mae > 0 else np.nan
        else:
            mase = np.nan
        
        metrics = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'median_ape': median_ape,
            'mase': mase,
            'n_samples': len(y_true),
            'n_samples_filtered': len(y_true_filtered)
        }
        
        return metrics

    def print_metrics_table(self, train_metrics, test_metrics):
        """Print formatted metrics comparison table."""
        sep = '=' * 80
        dash = '─' * 80
        
        print(f"\n{sep}")
        print("MODEL PERFORMANCE METRICS")
        print(sep)
        
        print(f"\n{'Metric':<25} {'Training Set':>20} {'Test Set':>20} {'Status':>10}")
        print(dash)
        
        # R² Score
        print(f"{'R² Score':<25} {train_metrics['r2']:>20.4f} {test_metrics['r2']:>20.4f} ", end='')
        print("✓ Good" if test_metrics['r2'] > 0.7 else "⚠ Fair" if test_metrics['r2'] > 0.5 else "✗ Poor")
        
        # MAE
        print(f"{'MAE (views)':<25} {train_metrics['mae']:>20,.0f} {test_metrics['mae']:>20,.0f} ", end='')
        print("✓ Good" if test_metrics['mae'] < 50000 else "⚠ Fair" if test_metrics['mae'] < 100000 else "✗ Poor")
        
        # RMSE
        print(f"{'RMSE (views)':<25} {train_metrics['rmse']:>20,.0f} {test_metrics['rmse']:>20,.0f} ", end='')
        print("✓ Good" if test_metrics['rmse'] < 75000 else "⚠ Fair" if test_metrics['rmse'] < 150000 else "✗ Poor")
        
        # MAPE
        if not np.isnan(test_metrics['mape']):
            print(f"{'MAPE (%)':<25} {train_metrics['mape']:>20.2f} {test_metrics['mape']:>20.2f} ", end='')
            print("✓ Good" if test_metrics['mape'] < 20 else "⚠ Fair" if test_metrics['mape'] < 40 else "✗ Poor")
        else:
            print(f"{'MAPE (%)':<25} {'N/A':>20} {'N/A':>20} {'—':>10}")
        
        # Median APE
        if not np.isnan(test_metrics['median_ape']):
            print(f"{'Median APE (%)':<25} {train_metrics['median_ape']:>20.2f} {test_metrics['median_ape']:>20.2f} ", end='')
            print("✓ Good" if test_metrics['median_ape'] < 15 else "⚠ Fair" if test_metrics['median_ape'] < 30 else "✗ Poor")
        else:
            print(f"{'Median APE (%)':<25} {'N/A':>20} {'N/A':>20} {'—':>10}")
        
        # MASE
        if not np.isnan(test_metrics['mase']):
            print(f"{'MASE':<25} {train_metrics['mase']:>20.4f} {test_metrics['mase']:>20.4f} ", end='')
            print("✓ Good" if test_metrics['mase'] < 1 else "⚠ Fair" if test_metrics['mase'] < 1.5 else "✗ Poor")
        else:
            print(f"{'MASE':<25} {'N/A':>20} {'N/A':>20} {'—':>10}")
        
        print(dash)
        print(f"{'Sample Size':<25} {train_metrics['n_samples']:>20,} {test_metrics['n_samples']:>20,}")
        
        print(f"\n💡 METRIC INTERPRETATION:")
        print(f"   • R² = {test_metrics['r2']:.1%} of variance explained by model")
        print(f"   • MAE = Average prediction error of {test_metrics['mae']:,.0f} views")
        print(f"   • RMSE = {test_metrics['rmse']:,.0f} views (penalizes large errors)")
        if not np.isnan(test_metrics['mape']):
            print(f"   • MAPE = {test_metrics['mape']:.1f}% average percentage error")
        if not np.isnan(test_metrics['median_ape']):
            print(f"   • Median APE = {test_metrics['median_ape']:.1f}% (robust to outliers)")
        if not np.isnan(test_metrics['mase']):
            mase_interp = "better" if test_metrics['mase'] < 1 else "worse"
            print(f"   • MASE = {test_metrics['mase']:.2f} ({mase_interp} than naive baseline)")

    def train_model(self):
        """Train Gradient Boosting on mature videos."""
        sep = '=' * 70
        print(f"\n{sep}")
        print("TRAINING MODEL FOR CATALOG PREDICTIONS")
        print(sep)

        mature_df = self.df[self.df['months_since_publish'] >= 6].copy()
        
        print(f"\n📊 Training Set:")
        print(f"   Videos (6+ months):          {len(mature_df)}")
        
        # Features
        feature_columns = [
            'log_impressions', 'impressions_ctr', 'ctr_x_log_imp',
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
        y = mature_df['log_views'].copy()

        X = X.fillna(X.median())

        # Scale
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
        
        # Convert back from log space
        y_train_actual = np.expm1(y_train)
        y_test_actual = np.expm1(y_test)
        y_pred_train_actual = np.expm1(y_pred_train)
        y_pred_test_actual = np.expm1(y_pred_test)

        # Calculate comprehensive metrics
        train_metrics = self.calculate_comprehensive_metrics(
            y_train_actual.values, y_pred_train_actual, "Training"
        )
        test_metrics = self.calculate_comprehensive_metrics(
            y_test_actual.values, y_pred_test_actual, "Test"
        )
        
        # Store metrics
        self.metrics['train'] = train_metrics
        self.metrics['test'] = test_metrics

        # Print metrics table
        self.print_metrics_table(train_metrics, test_metrics)

        print(f"\n✓ Model trained successfully")

        return test_metrics['r2'], test_metrics['mae']

    def predict_all_videos(self):
        """Generate predictions for all existing videos."""
        print(f"\n{'='*70}")
        print("GENERATING PREDICTIONS FOR EXISTING CATALOG")
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
                'actual': video['views'],
                'predicted_mature': pred,
                'months_old': video['months_since_publish'],
                'publish_date': video['publish_date']
            })

        pred_df = pd.DataFrame(predictions)
        pred_df = pred_df.sort_values('publish_date')
        
        print(f"\n✓ Generated predictions for {len(pred_df)} videos")
        
        return pred_df

    def forecast_catalog_growth(self, pred_df):
        """
        Forecast 6-month aggregate growth for existing catalog.
        """
        sep = '=' * 70
        dash = '─' * 70
        print(f"\n{sep}")
        print("AGGREGATE CATALOG 6-MONTH GROWTH FORECAST")
        print(sep)
        
        # Current state
        total_current_views = pred_df['actual'].sum()
        avg_views_per_video = pred_df['actual'].mean()
        
        print(f"\n📊 CURRENT CATALOG STATE:")
        print(f"   Total videos:                {len(pred_df):>15}")
        print(f"   Total current views:         {total_current_views:>15,}")
        print(f"   Average per video:           {avg_views_per_video:>15,.0f}")
        
        # Segment by age
        mature_videos = pred_df[pred_df['months_old'] >= 6]
        growing_videos = pred_df[(pred_df['months_old'] >= 3) & (pred_df['months_old'] < 6)]
        new_videos = pred_df[pred_df['months_old'] < 3]
        
        print(f"\n📅 CATALOG SEGMENTATION:")
        print(f"   Mature (6+ months):          {len(mature_videos):>15} videos ({mature_videos['actual'].sum():>12,} views)")
        print(f"   Growing (3-6 months):        {len(growing_videos):>15} videos ({growing_videos['actual'].sum():>12,} views)")
        print(f"   New (<3 months):             {len(new_videos):>15} videos ({new_videos['actual'].sum():>12,} views)")
        
        # Calculate 6-month growth by segment
        # Mature videos: 5-8% growth (long-tail catalog views)
        mature_growth_rate = 0.065
        mature_growth = mature_videos['actual'].sum() * mature_growth_rate
        
        # Growing videos: Still maturing, predict to 6-month mark
        growing_growth = 0
        for idx, video in growing_videos.iterrows():
            # Estimate remaining growth to maturity
            predicted_mature = video['predicted_mature']
            current_views = video['actual']
            remaining_growth = max(0, predicted_mature - current_views)
            growing_growth += remaining_growth * 0.7  # 70% of remaining in next 6mo
        
        # New videos: High growth as they mature
        new_growth = 0
        for idx, video in new_videos.iterrows():
            predicted_mature = video['predicted_mature']
            current_views = video['actual']
            remaining_growth = max(0, predicted_mature - current_views)
            new_growth += remaining_growth * 0.85  # 85% of remaining in next 6mo
        
        total_6mo_growth = mature_growth + growing_growth + new_growth
        projected_6mo_total = total_current_views + total_6mo_growth
        
        print(f"\n🔮 6-MONTH GROWTH PROJECTION:")
        print(dash)
        print(f"{'Segment':<25} {'Current Views':>15} {'Growth':>15} {'% Growth':>10}")
        print(dash)
        print(f"{'Mature catalog':<25} {mature_videos['actual'].sum():>15,} {mature_growth:>15,.0f} {mature_growth_rate*100:>9.1f}%")
        print(f"{'Growing videos':<25} {growing_videos['actual'].sum():>15,} {growing_growth:>15,.0f} {(growing_growth/max(growing_videos['actual'].sum(),1))*100:>9.1f}%")
        print(f"{'New videos':<25} {new_videos['actual'].sum():>15,} {new_growth:>15,.0f} {(new_growth/max(new_videos['actual'].sum(),1))*100:>9.1f}%")
        print(dash)
        print(f"{'TOTAL':<25} {total_current_views:>15,} {total_6mo_growth:>15,.0f} {(total_6mo_growth/total_current_views)*100:>9.1f}%")
        
        print(f"\n📈 PROJECTED 6-MONTH TOTAL:    {projected_6mo_total:>15,} views")
        
        # Confidence intervals
        conservative_growth = total_6mo_growth * 0.7
        optimistic_growth = total_6mo_growth * 1.3
        
        print(f"\n💡 SCENARIO ANALYSIS:")
        print(dash)
        print(f"{'Scenario':<25} {'Total Views':>15} {'Growth':>15}")
        print(dash)
        print(f"{'Conservative (70%)':<25} {total_current_views + conservative_growth:>15,.0f} {conservative_growth:>15,.0f}")
        print(f"{'Baseline (expected)':<25} {projected_6mo_total:>15,.0f} {total_6mo_growth:>15,.0f}")
        print(f"{'Optimistic (130%)':<25} {total_current_views + optimistic_growth:>15,.0f} {optimistic_growth:>15,.0f}")
        
        # Key insights
        growth_pct = (total_6mo_growth / total_current_views) * 100
        
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"   • Expected catalog growth:   {growth_pct:>5.1f}% over 6 months")
        print(f"   • Monthly growth rate:       {growth_pct/6:>5.1f}% per month")
        print(f"   • This excludes NEW releases (only existing videos)")
        print(f"   • Mature catalog provides:   {(mature_growth/total_6mo_growth)*100:>5.1f}% of growth")
        
        # Visualizations
        self.plot_catalog_forecast(pred_df, total_current_views, projected_6mo_total,
                                   conservative_growth, optimistic_growth, total_6mo_growth)
        
        return {
            'current_total': int(total_current_views),
            'projected_6mo_total': int(projected_6mo_total),
            'total_growth': int(total_6mo_growth),
            'growth_pct': growth_pct,
            'mature_growth': int(mature_growth),
            'growing_growth': int(growing_growth),
            'new_growth': int(new_growth)
        }

    def plot_catalog_forecast(self, pred_df, current_total, projected_total,
                             conservative_growth, optimistic_growth, baseline_growth):
        """Visualize aggregate catalog forecast."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('Aggregate Catalog Growth Forecast (Existing Videos Only)', 
                        fontsize=16, fontweight='bold')
            
            # Plot 1: Cumulative growth projection
            ax1 = axes[0, 0]
            months = np.arange(0, 7)
            current_line = np.full(7, current_total)
            baseline_line = np.linspace(current_total, projected_total, 7)
            conservative_line = np.linspace(current_total, current_total + conservative_growth, 7)
            optimistic_line = np.linspace(current_total, current_total + optimistic_growth, 7)
            
            ax1.plot(months, current_line, 'k--', linewidth=2, alpha=0.5, label='Current Total')
            ax1.plot(months, baseline_line, 'g-', linewidth=3, marker='o', markersize=8, label='Baseline Projection')
            ax1.fill_between(months, conservative_line, optimistic_line, alpha=0.2, color='green', label='Confidence Range')
            
            ax1.set_xlabel('Months from Now', fontsize=11)
            ax1.set_ylabel('Total Catalog Views', fontsize=11)
            ax1.set_title('6-Month Catalog Growth Trajectory', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'
            ))
            
            # Plot 2: Growth by segment
            ax2 = axes[0, 1]
            mature_videos = pred_df[pred_df['months_old'] >= 6]
            growing_videos = pred_df[(pred_df['months_old'] >= 3) & (pred_df['months_old'] < 6)]
            new_videos = pred_df[pred_df['months_old'] < 3]
            
            segments = ['Mature\n(6+ mo)', 'Growing\n(3-6 mo)', 'New\n(<3 mo)']
            current_views = [mature_videos['actual'].sum(), growing_videos['actual'].sum(), new_videos['actual'].sum()]
            
            x = np.arange(len(segments))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, current_views, width, label='Current', color='steelblue', edgecolor='k')
            
            # Calculate projected for each segment
            mature_proj = current_views[0] * 1.065
            growing_proj = current_views[1] * 1.25  # Growing faster
            new_proj = current_views[2] * 1.50  # Fastest growth
            projected_views = [mature_proj, growing_proj, new_proj]
            
            bars2 = ax2.bar(x + width/2, projected_views, width, label='Projected (6mo)', color='orange', edgecolor='k')
            
            ax2.set_ylabel('Total Views', fontsize=11)
            ax2.set_title('Growth by Video Age Segment', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(segments)
            ax2.legend()
            ax2.grid(alpha=0.3, axis='y')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    label = f'{height/1e6:.1f}M' if height >= 1e6 else f'{height/1e3:.0f}K'
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            label, ha='center', va='bottom', fontsize=9)
            
            # Plot 3: Growth sources pie chart
            ax3 = axes[1, 0]
            mature_growth_calc = mature_videos['actual'].sum() * 0.065
            growing_growth_calc = growing_videos['actual'].sum() * 0.25
            new_growth_calc = new_videos['actual'].sum() * 0.50
            
            growth_sources = [mature_growth_calc, growing_growth_calc, new_growth_calc]
            colors = ['#66b3ff', '#99ff99', '#ffcc99']
            labels = [f'Mature Catalog\n{mature_growth_calc:,.0f}', 
                     f'Growing Videos\n{growing_growth_calc:,.0f}',
                     f'New Videos\n{new_growth_calc:,.0f}']
            
            wedges, texts, autotexts = ax3.pie(growth_sources, labels=labels, colors=colors,
                                                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
            ax3.set_title('6-Month Growth Sources', fontsize=12, fontweight='bold')
            
            # Plot 4: Monthly growth rate
            ax4 = axes[1, 1]
            monthly_growth = baseline_growth / 6
            months_labels = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6']
            cumulative_monthly = np.cumsum([monthly_growth] * 6)
            
            bars = ax4.bar(months_labels, [monthly_growth] * 6, color='coral', edgecolor='k', alpha=0.7)
            ax4_twin = ax4.twinx()
            ax4_twin.plot(months_labels, cumulative_monthly, 'g-o', linewidth=2.5, markersize=8, label='Cumulative')
            
            ax4.set_ylabel('Monthly Growth (views)', fontsize=11, color='coral')
            ax4_twin.set_ylabel('Cumulative Growth (views)', fontsize=11, color='green')
            ax4.set_title('Monthly Growth Breakdown', fontsize=12, fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(alpha=0.3, axis='y')
            ax4_twin.legend(loc='upper left')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                label = f'{height/1e3:.0f}K' if height >= 1e3 else f'{height:.0f}'
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        label, ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.show()
            
            print(f"\n✓ Catalog forecast visualizations generated")
            
        except Exception as e:
            print(f"\n✗ Could not generate plots: {e}")
            import traceback
            traceback.print_exc()

    def plot_metrics_comparison(self):
        """Visualize model performance metrics."""
        if not self.metrics:
            print("⚠ No metrics available to plot")
            return
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('Model Performance Metrics Analysis', fontsize=16, fontweight='bold')
            
            train_m = self.metrics['train']
            test_m = self.metrics['test']
            
            # Plot 1: R² Score
            ax1 = axes[0, 0]
            metrics_names = ['Training', 'Test']
            r2_values = [train_m['r2'], test_m['r2']]
            bars = ax1.bar(metrics_names, r2_values, color=['steelblue', 'coral'], edgecolor='k')
            ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (>0.7)')
            ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (>0.5)')
            ax1.set_ylabel('R² Score', fontsize=11)
            ax1.set_title('R² Score (Variance Explained)', fontsize=12, fontweight='bold')
            ax1.set_ylim([0, 1])
            ax1.legend(loc='lower right', fontsize=8)
            ax1.grid(alpha=0.3, axis='y')
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Plot 2: MAE & RMSE
            ax2 = axes[0, 1]
            x = np.arange(2)
            width = 0.35
            mae_values = [train_m['mae'], test_m['mae']]
            rmse_values = [train_m['rmse'], test_m['rmse']]
            bars1 = ax2.bar(x - width/2, mae_values, width, label='MAE', color='skyblue', edgecolor='k')
            bars2 = ax2.bar(x + width/2, rmse_values, width, label='RMSE', color='lightcoral', edgecolor='k')
            ax2.set_ylabel('Error (views)', fontsize=11)
            ax2.set_title('MAE & RMSE Comparison', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(metrics_names)
            ax2.legend()
            ax2.grid(alpha=0.3, axis='y')
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, p: f'{x/1e3:.0f}K' if x >= 1e3 else f'{x:.0f}'
            ))
            
            # Plot 3: MAPE & Median APE
            ax3 = axes[0, 2]
            if not np.isnan(test_m['mape']) and not np.isnan(test_m['median_ape']):
                mape_values = [train_m['mape'], test_m['mape']]
                medape_values = [train_m['median_ape'], test_m['median_ape']]
                bars1 = ax3.bar(x - width/2, mape_values, width, label='MAPE', color='lightgreen', edgecolor='k')
                bars2 = ax3.bar(x + width/2, medape_values, width, label='Median APE', color='wheat', edgecolor='k')
                ax3.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Good (<20%)')
                ax3.set_ylabel('Percentage Error (%)', fontsize=11)
                ax3.set_title('MAPE & Median APE', fontsize=12, fontweight='bold')
                ax3.set_xticks(x)
                ax3.set_xticklabels(metrics_names)
                ax3.legend(loc='upper right', fontsize=8)
                ax3.grid(alpha=0.3, axis='y')
            else:
                ax3.text(0.5, 0.5, 'MAPE/Median APE\nNot Available', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_title('MAPE & Median APE', fontsize=12, fontweight='bold')
            
            # Plot 4: MASE
            ax4 = axes[1, 0]
            if not np.isnan(test_m['mase']):
                mase_values = [train_m['mase'], test_m['mase']]
                bars = ax4.bar(metrics_names, mase_values, color=['mediumseagreen', 'tomato'], edgecolor='k')
                ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (Naive)')
                ax4.set_ylabel('MASE', fontsize=11)
                ax4.set_title('MASE (vs Naive Baseline)', fontsize=12, fontweight='bold')
                ax4.legend(loc='upper right', fontsize=9)
                ax4.grid(alpha=0.3, axis='y')
                for bar in bars:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                # Add interpretation
                if test_m['mase'] < 1:
                    ax4.text(0.5, 0.95, '✓ Better than baseline', 
                            transform=ax4.transAxes, ha='center', va='top',
                            fontsize=10, color='green', fontweight='bold')
                else:
                    ax4.text(0.5, 0.95, '✗ Worse than baseline', 
                            transform=ax4.transAxes, ha='center', va='top',
                            fontsize=10, color='red', fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'MASE\nNot Available', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('MASE (vs Naive Baseline)', fontsize=12, fontweight='bold')
            
            # Plot 5: Error Distribution (residuals)
            ax5 = axes[1, 1]
            ax5.text(0.5, 0.5, 'Metrics Summary\n\nSee detailed table\nin console output', 
                    ha='center', va='center', transform=ax5.transAxes, 
                    fontsize=12, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            ax5.set_title('Additional Info', fontsize=12, fontweight='bold')
            ax5.axis('off')
            
            # Plot 6: Metrics Radar Chart
            ax6 = axes[1, 2]
            # Normalize metrics for radar chart (0-1 scale, higher is better)
            metrics_radar = {
                'R²': test_m['r2'],
                'MAE\n(inv)': 1 - min(test_m['mae'] / 200000, 1),  # Inverse, normalize
                'RMSE\n(inv)': 1 - min(test_m['rmse'] / 300000, 1),  # Inverse, normalize
            }
            if not np.isnan(test_m['mape']):
                metrics_radar['MAPE\n(inv)'] = 1 - min(test_m['mape'] / 100, 1)
            if not np.isnan(test_m['median_ape']):
                metrics_radar['MedAPE\n(inv)'] = 1 - min(test_m['median_ape'] / 100, 1)
            if not np.isnan(test_m['mase']):
                metrics_radar['MASE\n(inv)'] = 1 - min(test_m['mase'] / 2, 1)
            
            categories = list(metrics_radar.keys())
            values = list(metrics_radar.values())
            
            # Radar chart
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            
            ax6 = plt.subplot(2, 3, 6, projection='polar')
            ax6.plot(angles, values, 'o-', linewidth=2, color='b', label='Test Performance')
            ax6.fill(angles, values, alpha=0.25, color='b')
            ax6.set_xticks(angles[:-1])
            ax6.set_xticklabels(categories, fontsize=9)
            ax6.set_ylim(0, 1)
            ax6.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax6.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
            ax6.set_title('Overall Performance\n(Normalized)', fontsize=12, fontweight='bold', pad=20)
            ax6.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            print(f"\n✓ Metrics visualization generated")
            
        except Exception as e:
            print(f"\n✗ Could not generate metrics plot: {e}")
            import traceback
            traceback.print_exc()

    def run_pipeline(self):
        """Execute complete aggregate prediction pipeline."""
        header = '#' * 70
        print(f"\n{header}")
        print('AGGREGATE CATALOG GROWTH PREDICTOR')
        print('Existing Videos Only - No New Releases')
        print(header)

        if not self.load_data():
            return False

        # Train model
        self.train_model()
        
        # Plot metrics
        self.plot_metrics_comparison()

        # Predict all videos
        pred_df = self.predict_all_videos()

        # Forecast aggregate growth
        forecast = self.forecast_catalog_growth(pred_df)

        print(f"\n{header}")
        print('✅ AGGREGATE FORECAST COMPLETE')
        print(header)
        print(f"\n💡 SUMMARY:")
        print(f"   Current catalog:     {forecast['current_total']:>12,} views")
        print(f"   6-month projection:  {forecast['projected_6mo_total']:>12,} views")
        print(f"   Expected growth:     {forecast['total_growth']:>12,} views ({forecast['growth_pct']:.1f}%)")
        print(f"   Monthly growth rate: {forecast['growth_pct']/6:>12.2f}% per month")
        print(f"\n   📌 This is CATALOG ONLY (no new releases included)")
        print(f"{header}\n")
        
        return True


if __name__ == '__main__':
    predictor = AggregateCatalogPredictor(db_params)
    predictor.run_pipeline()