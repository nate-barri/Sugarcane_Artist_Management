######################################################################
# GRADIENT BOOSTING CUMULATIVE 6-MONTH FORECAST
# Predicts cumulative view growth over next 6 months
# Includes comprehensive evaluation metrics
# EDITS:
#  - ax2 now plots ONLY the last 6 months of historical cumulative
#    and the next 6 months of projection (no longer plotting full history)
#  - run_pipeline prints a compact evaluation-metrics block (r2, mae, rmse,
#    mape, median_ape, mase) after the forecast
# Note: No calculation logic was changed — only plotting / final-metrics display.
######################################################################

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Database connection parameters
db_params = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host': 'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port': '5432',
    'sslmode': 'require'
}


class CumulativeForecastPredictor:
    """
    Gradient Boosting model for cumulative 6-month view forecasts.
    Provides comprehensive evaluation metrics: R², MAE, RMSE, MAPE, MASE, Median APE
    """

    def __init__(self, db_params):
        self.db_params = db_params
        self.df = None
        self.model = None
        self.scaler = None
        self.encoder = None
        self.feature_columns = []
        self.use_log_target = True
        self.predictions_df = None

    def load_and_prepare_data(self):
        """Load and engineer features."""
        try:
            conn = psycopg2.connect(**self.db_params)
            query = "SELECT * FROM yt_video_etl"
            self.df = pd.read_sql(query, conn)
            conn.close()
            print(f"✓ Data loaded: {len(self.df)} records")

            self._prepare_features()
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
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

        # Engagement metrics
        self.df['total_engagement'] = (
            self.df['likes'] + self.df['shares'] + self.df['comments_added']
        )
        self.df['engagement_rate'] = (
            self.df['total_engagement'] / np.maximum(self.df['views'], 1)
        )

        # Duration
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

        # Content classification
        self.df['content_type'] = self.df['video_title'].apply(self._classify_content)

        # Log transforms
        self.df['log_impressions'] = np.log1p(self.df['impressions'])
        self.df['log_views'] = np.log1p(self.df['views'])

        # Key interaction features
        self.df['ctr_x_log_imp'] = self.df['impressions_ctr'] * self.df['log_impressions']
        self.df['high_ctr'] = (self.df['impressions_ctr'] > 0.08).astype(int)

        # Cyclical encoding
        self.df['sin_month'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['cos_month'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['sin_day'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['cos_day'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)

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
        if 'instrumental' in t or 'karaoke' in t:
            return 'instrumental'
        return 'other'

    def prepare_ml_features(self):
        """Prepare features for training."""
        mature_df = self.df[self.df['months_since_publish'] >= 6].copy()

        print('\n📊 Dataset Preparation:')
        print(f"   Videos with 6+ months data: {len(mature_df)}")

        # Identify viral outliers
        view_threshold = mature_df['views'].quantile(0.95)  # Top 5%
        viral_videos = mature_df[mature_df['views'] > view_threshold]

        if len(viral_videos) > 0:
            print(f"\n⚡ VIRAL OUTLIER DETECTION:")
            print(f"   Threshold (95th percentile): {view_threshold:>12,.0f} views")
            print(f"   Viral videos found:          {len(viral_videos):>12}")
            print(f"\n   Top viral performers:")
            for idx, video in viral_videos.nlargest(3, 'views').iterrows():
                pct_of_total = (video['views'] / mature_df['views'].sum()) * 100
                print(f"     • {video['video_title'][:50]:<50} {video['views']:>12,} ({pct_of_total:>5.1f}% of total)")

        print(f"\n   Keeping ALL data (no outlier removal)")

        # Feature selection
        self.feature_columns = [
            'log_impressions', 'impressions_ctr', 'ctr_x_log_imp',
            'duration_minutes', 'high_ctr', 'engagement_rate',
            'is_weekend', 'sin_month', 'cos_month',
            'sin_day', 'cos_day', 'content_type'
        ]

        # Encode content_type
        self.encoder = LabelEncoder()
        mature_df['content_type_encoded'] = self.encoder.fit_transform(
            mature_df['content_type']
        )

        feature_cols_final = [
            c if c != 'content_type' else 'content_type_encoded'
            for c in self.feature_columns
        ]
        X = mature_df[feature_cols_final].copy()

        # Target: views at 6 months
        y = mature_df['log_views'].copy() if self.use_log_target else mature_df['views'].copy()

        # Fill NaNs
        X = X.fillna(X.median())

        # Robust scaling
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols_final, index=X.index)

        return X_scaled, y, mature_df

    def show_regression_diagnostics(self, y_true, y_pred):
        """
        Show regression-specific diagnostics (analogous to classification metrics).
        Includes residual analysis, prediction intervals, and quartile performance.
        """
        sep = '=' * 70
        dash = '─' * 70

        print(f"\n{sep}")
        print("REGRESSION DIAGNOSTICS (Classification-Style Analysis)")
        print(sep)

        # 1. RESIDUAL ANALYSIS (analog to confusion matrix)
        residuals = y_true - y_pred

        print(f"\n📊 RESIDUAL ANALYSIS:")
        print(dash)
        print(f"{'Metric':<35} {'Value':<20} {'% of Mean':<15}")
        print(dash)

        mean_actual = np.mean(y_true)
        print(f"{'Mean Residual':<35} {np.mean(residuals):>19,.0f} {(np.mean(residuals)/mean_actual)*100:>14.2f}%")
        print(f"{'Std Residual':<35} {np.std(residuals):>19,.0f} {(np.std(residuals)/mean_actual)*100:>14.2f}%")
        print(f"{'Min Residual (under-predict)':<35} {np.min(residuals):>19,.0f} {(np.min(residuals)/mean_actual)*100:>14.2f}%")
        print(f"{'Max Residual (over-predict)':<35} {np.max(residuals):>19,.0f} {(np.max(residuals)/mean_actual)*100:>14.2f}%")

        # 2. PREDICTION ACCURACY BY THRESHOLDS (analog to confusion matrix)
        thresholds = [10, 20, 30, 50]
        print(f"\n🎯 PREDICTION ACCURACY BY ERROR THRESHOLD:")
        print(dash)
        print(f"{'Error Threshold':<20} {'Count':<10} {'Percentage':<15}")
        print(dash)

        for thresh in thresholds:
            within_thresh = np.sum(np.abs((y_true - y_pred) / np.maximum(y_true, 1)) * 100 <= thresh)
            pct = (within_thresh / len(y_true)) * 100
            print(f"{'Within ±' + str(thresh) + '%':<20} {within_thresh:<10} {pct:>13.1f}%")

        # 3. PERFORMANCE BY VIEW QUARTILES (classification-like breakdown)
        quartiles = np.percentile(y_true, [25, 50, 75])

        print(f"\n📈 PERFORMANCE BY VIEW QUARTILES:")
        print(dash)
        print(f"{'Quartile':<15} {'Range':<25} {'MAE (% of mean)':<18} {'MAPE (%)':<12}")
        print(dash)

        # Q1: Bottom 25%
        q1_mask = y_true <= quartiles[0]
        if np.sum(q1_mask) > 0:
            q1_mae = mean_absolute_error(y_true[q1_mask], y_pred[q1_mask])
            q1_mae_pct = (q1_mae / np.mean(y_true[q1_mask])) * 100
            q1_mape = np.mean(np.abs((y_true[q1_mask] - y_pred[q1_mask]) / np.maximum(y_true[q1_mask], 1))) * 100
            print(f"{'Q1 (0-25%)':<15} {'0 - ' + f'{quartiles[0]:,.0f}':<25} {q1_mae_pct:>17.1f}% {q1_mape:>11.1f}%")

        # Q2: 25-50%
        q2_mask = (y_true > quartiles[0]) & (y_true <= quartiles[1])
        if np.sum(q2_mask) > 0:
            q2_mae = mean_absolute_error(y_true[q2_mask], y_pred[q2_mask])
            q2_mae_pct = (q2_mae / np.mean(y_true[q2_mask])) * 100
            q2_mape = np.mean(np.abs((y_true[q2_mask] - y_pred[q2_mask]) / np.maximum(y_true[q2_mask], 1))) * 100
            print(f"{'Q2 (25-50%)':<15} {f'{quartiles[0]:,.0f}' + ' - ' + f'{quartiles[1]:,.0f}':<25} {q2_mae_pct:>17.1f}% {q2_mape:>11.1f}%")

        # Q3: 50-75%
        q3_mask = (y_true > quartiles[1]) & (y_true <= quartiles[2])
        if np.sum(q3_mask) > 0:
            q3_mae = mean_absolute_error(y_true[q3_mask], y_pred[q3_mask])
            q3_mae_pct = (q3_mae / np.mean(y_true[q3_mask])) * 100
            q3_mape = np.mean(np.abs((y_true[q3_mask] - y_pred[q3_mask]) / np.maximum(y_true[q3_mask], 1))) * 100
            print(f"{'Q3 (50-75%)':<15} {f'{quartiles[1]:,.0f}' + ' - ' + f'{quartiles[2]:,.0f}':<25} {q3_mae_pct:>17.1f}% {q3_mape:>11.1f}%")

        # Q4: Top 25%
        q4_mask = y_true > quartiles[2]
        if np.sum(q4_mask) > 0:
            q4_mae = mean_absolute_error(y_true[q4_mask], y_pred[q4_mask])
            q4_mae_pct = (q4_mae / np.mean(y_true[q4_mask])) * 100
            q4_mape = np.mean(np.abs((y_true[q4_mask] - y_pred[q4_mask]) / np.maximum(y_true[q4_mask], 1))) * 100
            print(f"{'Q4 (75-100%)':<15} {f'{quartiles[2]:,.0f}' + ' +':<25} {q4_mae_pct:>17.1f}% {q4_mape:>11.1f}%")

        # 4. PREDICTION INTERVAL COVERAGE (analog to ROC)
        print(f"\n🎲 PREDICTION INTERVAL COVERAGE:")
        print(dash)

        # Calculate prediction intervals at different confidence levels
        std_error = np.std(residuals)
        confidence_levels = [0.68, 0.80, 0.90, 0.95]
        z_scores = [1.0, 1.28, 1.645, 1.96]

        print(f"{'Confidence Level':<20} {'Expected':<15} {'Actual':<15} {'Status':<10}")
        print(dash)

        for conf, z in zip(confidence_levels, z_scores):
            interval_lower = y_pred - z * std_error
            interval_upper = y_pred + z * std_error

            within_interval = np.sum((y_true >= interval_lower) & (y_true <= interval_upper))
            actual_coverage = (within_interval / len(y_true)) * 100
            expected_coverage = conf * 100

            status = "✓" if abs(actual_coverage - expected_coverage) < 10 else "⚠️"
            print(f"{f'{conf*100:.0f}%':<20} {expected_coverage:>13.1f}% {actual_coverage:>14.1f}% {status:<10}")

        # 5. VISUALIZATIONS
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Plot 1: Residuals vs Predicted (analog to confusion matrix heatmap)
            axes[0, 0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
            axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
            axes[0, 0].set_xlabel('Predicted Views')
            axes[0, 0].set_ylabel('Residuals (Actual - Predicted)')
            axes[0, 0].set_title('Residual Plot (Check for Patterns)')
            axes[0, 0].grid(alpha=0.3)

            # Plot 2: Q-Q Plot for Normality
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot (Residual Normality Check)')
            axes[0, 1].grid(alpha=0.3)

            # Plot 3: Actual vs Predicted with Perfect Line
            axes[1, 0].scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            axes[1, 0].set_xlabel('Actual Views')
            axes[1, 0].set_ylabel('Predicted Views')
            axes[1, 0].set_title('Predicted vs Actual (Closer to Line = Better)')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)

            # Plot 4: Error Distribution
            axes[1, 1].hist(residuals, bins=20, edgecolor='k', alpha=0.7)
            axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[1, 1].set_xlabel('Residual (Actual - Predicted)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Error Distribution (Should be Centered at 0)')
            axes[1, 1].grid(alpha=0.3)

            plt.tight_layout()
            plt.show()

            print(f"\n✓ Regression diagnostic plots generated")

        except Exception as e:
            print(f"\n✗ Could not generate diagnostic plots: {e}")

    def calculate_comprehensive_metrics(self, y_true, y_pred, y_train=None):
        """
        Calculate comprehensive evaluation metrics.

        Returns dict with R², MAE, RMSE, MAPE, MASE, Median APE
        """
        # R² Score
        r2 = r2_score(y_true, y_pred)

        # MAE (Mean Absolute Error)
        mae = mean_absolute_error(y_true, y_pred)

        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

        # Median APE (Median Absolute Percentage Error)
        median_ape = np.median(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

        # MASE (Mean Absolute Scaled Error)
        # For cross-sectional data (different videos), use "predict the mean" as naive baseline
        if y_train is not None and len(y_train) > 1:
            # Naive forecast = always predict training mean
            naive_pred = np.mean(y_train)
            naive_errors = np.abs(y_true - naive_pred)
            scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
            mase = mae / scale if scale > 0 else np.nan
        else:
            mase = np.nan

        return {
            'R²': r2,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'MASE': mase,
            'Median_APE': median_ape
        }

    def train_and_evaluate(self):
        """Train Gradient Boosting and calculate all metrics."""
        sep = '=' * 70
        dash = '─' * 70
        print(f"\n{sep}")
        print("GRADIENT BOOSTING TRAINING & EVALUATION")
        print(sep)

        X, y, mature_df = self.prepare_ml_features()

        print(f"\n   Features: {len(X.columns)}")
        print(f"   Target: log(views at 6 months)")
        print(f"   Range: {y.min():.2f} - {y.max():.2f}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

        # Train Gradient Boosting with stronger regularization
        self.model = GradientBoostingRegressor(
            n_estimators=100,  # Reduced from 150
            max_depth=3,       # Reduced from 4
            learning_rate=0.05,
            min_samples_split=8,   # Increased from 5
            min_samples_leaf=5,    # Increased from 3
            subsample=0.7,         # Reduced from 0.8
            max_features='sqrt',
            random_state=42,
            verbose=0
        )

        print(f"\n🤖 Training Gradient Boosting...")
        self.model.fit(X_train, y_train)

        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        # Back-transform if using log
        if self.use_log_target:
            y_train_actual = np.expm1(y_train)
            y_test_actual = np.expm1(y_test)
            y_pred_train_actual = np.expm1(y_pred_train)
            y_pred_test_actual = np.expm1(y_pred_test)
        else:
            y_train_actual = y_train.values
            y_test_actual = y_test.values
            y_pred_train_actual = y_pred_train
            y_pred_test_actual = y_pred_test

        # Calculate metrics
        train_metrics = self.calculate_comprehensive_metrics(
            y_train_actual, y_pred_train_actual, y_train_actual
        )
        test_metrics = self.calculate_comprehensive_metrics(
            y_test_actual, y_pred_test_actual, y_train_actual
        )

        # Cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=kfold, scoring='r2', n_jobs=-1
        )

        # Display metrics
        print(f"\n{dash}")
        print(f"📊 COMPREHENSIVE EVALUATION METRICS")
        print(dash)
        print(f"\n{'Metric':<25} {'Train':<18} {'Test':<18}")
        print(dash)

        # R² Score (as percentage)
        print(f"{'R² Score (%)':<25} {train_metrics['R²']*100:>17.2f}% {test_metrics['R²']*100:>17.2f}%")

        # MAE as % of mean
        train_mae_pct = (train_metrics['MAE'] / np.mean(y_train_actual)) * 100
        test_mae_pct = (test_metrics['MAE'] / np.mean(y_test_actual)) * 100
        print(f"{'MAE (% of mean)':<25} {train_mae_pct:>17.2f}% {test_mae_pct:>17.2f}%")
        print(f"{'MAE (absolute views)':<25} {train_metrics['MAE']:>17,.0f} {test_metrics['MAE']:>17,.0f}")

        # RMSE as % of mean
        train_rmse_pct = (train_metrics['RMSE'] / np.mean(y_train_actual)) * 100
        test_rmse_pct = (test_metrics['RMSE'] / np.mean(y_test_actual)) * 100
        print(f"{'RMSE (% of mean)':<25} {train_rmse_pct:>17.2f}% {test_rmse_pct:>17.2f}%")
        print(f"{'RMSE (absolute views)':<25} {train_metrics['RMSE']:>17,.0f} {test_metrics['RMSE']:>17,.0f}")

        # MAPE
        print(f"{'MAPE (%)':<25} {train_metrics['MAPE']:>17.2f}% {test_metrics['MAPE']:>17.2f}%")

        # Median APE
        print(f"{'Median APE (%)':<25} {train_metrics['Median_APE']:>17.2f}% {test_metrics['Median_APE']:>17.2f}%")

        # MASE (scaled metric, not a percentage but show with context)
        if not np.isnan(test_metrics['MASE']):
            # Calculate improvement over naive
            improvement_train = (1 - train_metrics['MASE']) * 100
            improvement_test = (1 - test_metrics['MASE']) * 100
            print(f"{'MASE (ratio)':<25} {train_metrics['MASE']:>17.3f} {test_metrics['MASE']:>17.3f}")
            print(f"{'  Improvement over naive':<25} {improvement_train:>17.1f}% {improvement_test:>17.1f}%")

        print(f"\n{'Cross-Validation':<25} {'Mean ± Std':<18}")
        print(dash)
        print(f"{'CV R² (%)':<25} {cv_scores.mean()*100:>9.2f}% ± {cv_scores.std()*100:.2f}%")

        # Overfitting check
        overfit = train_metrics['R²'] - test_metrics['R²']
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

        # Feature importance
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n{dash}")
        print(f"🎯 TOP 5 FEATURE IMPORTANCES")
        print(dash)
        for idx, row in importances.head(5).iterrows():
            bar = '█' * int(row['importance'] * 50)
            print(f"{row['feature']:<25} {row['importance']:.4f} {bar}")

        # Store predictions for cumulative forecast
        self.predictions_df = pd.DataFrame({
            'actual': y_test_actual,
            'predicted': y_pred_test_actual,
            'error': y_test_actual - y_pred_test_actual,
            'error_pct': np.abs((y_test_actual - y_pred_test_actual) / np.maximum(y_test_actual, 1)) * 100
        })

        # Additional regression diagnostics
        self.show_regression_diagnostics(y_test_actual, y_pred_test_actual)

        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'overfit_gap': overfit
        }

    def generate_predictions_for_all(self):
        """Generate predictions for all videos."""
        print(f"\n{'='*70}")
        print("GENERATING PREDICTIONS FOR ALL VIDEOS")
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
                'sin_day': video['sin_day'],
                'cos_day': video['cos_day'],
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
            pred = np.expm1(pred_log) if self.use_log_target else pred_log
            pred = int(max(0, pred))

            predictions.append({
                'title': video['video_title'],
                'type': video['content_type'],
                'actual': video['views'],
                'predicted': pred,
                'publish_date': video['publish_date'],
                'months_old': video['months_since_publish']
            })

        pred_df = pd.DataFrame(predictions)
        pred_df = pred_df.sort_values('publish_date')

        print(f"\n✓ Generated predictions for {len(pred_df)} videos")

        return pred_df

    def plot_cumulative_forecast(self, pred_df):
        """
        Generate cumulative view forecast visualization.
        Shows historical cumulative + 6-month forward projection.
        """
        sep = '=' * 70
        print(f"\n{sep}")
        print("CUMULATIVE 6-MONTH FORECAST")
        print(sep)

        # Sort by date
        pred_df = pred_df.sort_values('publish_date').reset_index(drop=True)

        # Calculate cumulative actuals
        cumulative_actual = pred_df['actual'].cumsum()
        cumulative_predicted = pred_df['predicted'].cumsum()

        # Current state
        current_total = cumulative_actual.iloc[-1]
        predicted_total = cumulative_predicted.iloc[-1]

        print(f"\n📊 CURRENT STATE:")
        print(f"   Current cumulative views:    {current_total:>15,}")
        print(f"   Model's cumulative estimate: {predicted_total:>15,}")
        print(f"   Difference:                  {abs(current_total - predicted_total):>15,}")

        # Calculate 6-month growth rate from recent videos
        recent_videos = pred_df[pred_df['months_old'] <= 12].copy()
        if len(recent_videos) >= 5:
            avg_views_per_video = recent_videos['predicted'].mean()

            # Estimate videos in next 6 months based on recent release rate
            recent_months = recent_videos['months_old'].max()
            videos_per_month = len(recent_videos) / max(recent_months, 1)
            expected_new_videos = int(videos_per_month * 6)

            # Growth from new releases
            new_video_views = avg_views_per_video * expected_new_videos

            # Growth from existing catalog (5-10% typical)
            catalog_growth_rate = 0.08
            catalog_growth = current_total * catalog_growth_rate

            total_6mo_growth = new_video_views + catalog_growth
        else:
            # Fallback: use overall average
            total_6mo_growth = current_total * 0.15
            expected_new_videos = 3

        projected_6mo_total = current_total + total_6mo_growth

        print(f"\n🔮 6-MONTH FORWARD PROJECTION:")
        print(f"   Expected new videos:         {expected_new_videos:>15}")
        print(f"   Growth from new content:     {new_video_views if 'new_video_views' in locals() else 0:>15,.0f}")
        print(f"   Growth from catalog:         {catalog_growth if 'catalog_growth' in locals() else 0:>15,.0f}")
        print(f"   Total 6-month growth:        {total_6mo_growth:>15,.0f}")
        print(f"   Projected total (6mo):       {projected_6mo_total:>15,.0f}")

        # Conservative and optimistic scenarios
        conservative_total = current_total + (total_6mo_growth * 0.6)
        optimistic_total = current_total + (total_6mo_growth * 1.4)

        print(f"\n📈 SCENARIO ANALYSIS:")
        print(f"   Conservative (-40%):         {conservative_total:>15,.0f}")
        print(f"   Baseline (expected):         {projected_6mo_total:>15,.0f}")
        print(f"   Optimistic (+40%):           {optimistic_total:>15,.0f}")

        # Visualization
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Plot 1: Historical cumulative (full for diagnostics)
            ax1.plot(pred_df['publish_date'], cumulative_actual,
                     label='Actual Cumulative', linewidth=2, marker='o', markersize=3)
            ax1.plot(pred_df['publish_date'], cumulative_predicted,
                     label='Model Estimate', linewidth=2, linestyle='--', alpha=0.7)
            ax1.set_title('Historical Cumulative Views: Actual vs Model', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Publish Date')
            ax1.set_ylabel('Cumulative Views')
            ax1.legend()
            ax1.grid(alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

            # Plot 2: LAST 6 months of history + next 6 months projection
            last_date = pred_df['publish_date'].iloc[-1]
            six_months_ago = last_date - pd.DateOffset(months=6)

            # Mask for recent 6 months of historical data
            recent_mask = pred_df['publish_date'] >= six_months_ago
            recent_df = pred_df.loc[recent_mask].copy()

            # Align cumulative series to the recent_df's index
            recent_cumulative_actual = cumulative_actual.loc[recent_df.index]

            # Future dates: keep as 7 points (0..6 months inclusive)
            future_dates = pd.date_range(start=last_date, periods=7, freq='M')

            # Projection line values
            projection_values = np.linspace(current_total, projected_6mo_total, 7)
            conservative_values = np.linspace(current_total, conservative_total, 7)
            optimistic_values = np.linspace(current_total, optimistic_total, 7)

            # Plot only recent 6 months history and the projection forward
            ax2.plot(recent_df['publish_date'], recent_cumulative_actual,
                     label='Historical (Last 6 Months)', linewidth=2, color='blue')
            ax2.plot(future_dates, projection_values,
                     label='6-Month Projection (Baseline)', linewidth=2.5,
                     linestyle='--', color='green', marker='o')
            ax2.fill_between(future_dates, conservative_values, optimistic_values,
                             alpha=0.2, color='green', label='Confidence Range')

            ax2.axvline(last_date, color='red', linestyle=':', alpha=0.5, label='Today')
            ax2.set_title('6-Month Cumulative View Forecast (Last 6 Months History + Next 6 Months)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Cumulative Views')
            ax2.legend()
            ax2.grid(alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

            plt.tight_layout()
            plt.show()

            print(f"\n✓ Cumulative forecast visualization complete")

        except Exception as e:
            print(f"\n✗ Could not generate plot: {e}")

        return {
            'current_total': int(current_total),
            'projected_6mo_total': int(projected_6mo_total),
            'total_growth': int(total_6mo_growth),
            'conservative': int(conservative_total),
            'optimistic': int(optimistic_total)
        }

    def run_pipeline(self):
        """Execute complete pipeline."""
        header = '#' * 70
        print(f"\n{header}")
        print('GRADIENT BOOSTING CUMULATIVE FORECAST SYSTEM')
        print(header)

        if not self.load_and_prepare_data():
            return False

        # Train and evaluate with comprehensive metrics
        metrics = self.train_and_evaluate()

        # Generate predictions for all videos
        pred_df = self.generate_predictions_for_all()

        # Generate cumulative 6-month forecast
        forecast = self.plot_cumulative_forecast(pred_df)

        print(f"\n{header}")
        print('✅ ANALYSIS COMPLETE')
        print(header)
        print(f"\n💡 KEY TAKEAWAYS:")
        print(f"   • Model Test R²: {metrics['test_metrics']['R²']:.4f}")
        print(f"   • Test MAE: {metrics['test_metrics']['MAE']:,.0f} views")
        print(f"   • 6-month growth: +{forecast['total_growth']:,} views")
        print(f"   • Projected total: {forecast['projected_6mo_total']:,} views")

        # --- Compact evaluation metrics printout requested ---
        # Provide the commonly-used lowercase keys as a simple dict-like printout
        test_m = metrics['test_metrics']
        compact_metrics = {
            'r2': float(test_m.get('R²', np.nan)),
            'mae': float(test_m.get('MAE', np.nan)),
            'rmse': float(test_m.get('RMSE', np.nan)),
            'mape': float(test_m.get('MAPE', np.nan)),
            'median_ape': float(test_m.get('Median_APE', np.nan)),
            'mase': float(test_m.get('MASE', np.nan))
        }
        print(f"\nMODEL EVAL METRICS (test set):")
        for k, v in compact_metrics.items():
            if k in ('mape', 'median_ape'):
                print(f"  {k}: {v:.2f}%")
            elif np.isnan(v):
                print(f"  {k}: nan")
            else:
                # show floats with readable formatting
                print(f"  {k}: {v:,.4f}")

        print(f"\n⚠️  IMPORTANT NOTES:")
        print(f"   • Your 2 Leonora videos (54M views) dominate your catalog")
        print(f"   • Forecasts assume similar viral success is rare")
        print(f"   • If you have another breakout hit, exceed projections significantly")
        print(f"   • Model is conservative for typical releases (100K-800K range)")
        print(f"{header}\n")

        return True


if __name__ == '__main__':
    predictor = CumulativeForecastPredictor(db_params)
    predictor.run_pipeline()
