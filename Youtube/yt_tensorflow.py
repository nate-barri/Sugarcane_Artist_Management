######################################################################
# YOUTUBE ML PREDICTOR - FULL PRODUCTION SCRIPT (500+ LINES)
# This file intentionally preserves the full original pipeline logic,
# extensive console reporting, model training, predictions for existing
# and unreleased videos, and visualization (interactive only).
#
# - Trains Ridge, ElasticNet, RandomForest, GradientBoosting with
#   regularization and outlier handling.
# - Computes MAE, RMSE, MAPE, MASE, Median APE and CV metrics.
# - Predicts for existing videos and unreleased concepts.
# - Displays interactive plots: model comparison, feature importances,
#   predicted vs actual, error distribution, cumulative 6-month forecast.
#
# IMPORTANT: This script connects to your database (yt_video_etl) using
# the db_params below. Ensure your environment can reach the database
# and has the required Python packages installed.
######################################################################

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------
# Database connection parameters (keep secure in production!)
# --------------------------------------------------------------------
db_params = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host': 'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port': '5432',
    'sslmode': 'require'
}


class ImprovedYouTubeMLPredictor:
    """Improved YouTube Predictor with Better Regularization and Outlier Handling

    This class is intentionally verbose and preserves the original
    pipeline structure. It provides methods for feature engineering,
    outlier removal, model training, predictions, and visualization.
    """

    def __init__(self, db_params):
        self.db_params = db_params
        self.df = None
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.best_model_name = None
        self.use_log_target = True  # Log-transform target to handle outliers

    # ----------------------------------------------------------------
    # DATA LOADING & PREPARATION
    # ----------------------------------------------------------------
    def load_and_prepare_data(self):
        """Load data from Postgres and run feature engineering."""
        try:
            conn = psycopg2.connect(**self.db_params)
            query = "SELECT * FROM yt_video_etl"
            self.df = pd.read_sql(query, conn)
            conn.close()
            print(f"✓ Data loaded: {len(self.df)} records")

            # Feature engineering step
            self._prepare_features()
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False

    def _prepare_features(self):
        """Perform the feature engineering used by the model pipeline.

        This includes date parsing, numeric coercion, engagement features,
        time-based features, cyclical encodings and interaction terms.
        """
        # Create publish_date column from year/month/day
        # defensive: ensure publish_year/month/day exist
        for col in ['publish_year', 'publish_month', 'publish_day']:
            if col not in self.df.columns:
                # If any are missing, create defaults (will be handled downstream)
                self.df[col] = datetime.now().year

        self.df['publish_date'] = pd.to_datetime(
            self.df['publish_year'].astype(str) + '-' +
            self.df['publish_month'].astype(str) + '-' +
            self.df['publish_day'].astype(str), errors='coerce'
        )

        # Ensure numeric columns exist
        numeric_cols = ['duration', 'impressions', 'impressions_ctr', 'views']
        for col in numeric_cols:
            if col not in self.df.columns:
                self.df[col] = 0

        # Convert types safely
        for col in ['duration', 'impressions', 'impressions_ctr', 'views']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        # Defensive: engagement columns
        for col in ['likes', 'shares', 'comments_added']:
            if col not in self.df.columns:
                self.df[col] = 0
            else:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        # Aggregate engagement
        self.df['total_engagement'] = (self.df['likes'] + self.df['shares'] + self.df['comments_added'])
        self.df['engagement_rate'] = self.df['total_engagement'] / np.maximum(self.df['views'], 1)

        # Duration in minutes
        self.df['duration_minutes'] = self.df['duration'] / 60.0

        # Time features
        self.df['day_of_week'] = self.df['publish_date'].dt.dayofweek.fillna(0).astype(int)
        self.df['month'] = self.df['publish_date'].dt.month.fillna(1).astype(int)
        self.df['quarter'] = self.df['publish_date'].dt.quarter.fillna(1).astype(int)
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)

        # Age
        now = datetime.now()
        self.df['days_since_publish'] = (now - self.df['publish_date']).dt.days.fillna(0).astype(int)
        self.df['months_since_publish'] = self.df['days_since_publish'] / 30.44

        # Content type classification
        self.df['content_type'] = self.df['video_title'].apply(self._classify_content)

        # Log transforms
        self.df['log_impressions'] = np.log1p(self.df['impressions'])
        self.df['log_views'] = np.log1p(self.df['views'])

        # Interaction feature
        self.df['ctr_x_log_imp'] = self.df['impressions_ctr'] * self.df['log_impressions']

        # Seasonal cyclical encoding
        self.df['sin_month'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['cos_month'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['sin_day'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['cos_day'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)

        print(f"✓ Features engineered: {self.df.shape}")

    def _classify_content(self, title):
        """Simple rule-based content classifier (keeps original logic)."""
        t = str(title).lower()
        if 'official music video' in t or 'music video' in t:
            return 'music_video'
        if 'lyric' in t:
            return 'lyrics'
        if 'live' in t:
            return 'live'
        return 'other'

    # ----------------------------------------------------------------
    # OUTLIER REMOVAL
    # ----------------------------------------------------------------
    def remove_outliers(self, df, column='views', method='iqr', threshold=3):
        """Remove outliers using IQR or z-score as specified."""
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            mask = (df[column] >= lower) & (df[column] <= upper)
        elif method == 'zscore':
            z = (df[column] - df[column].mean()) / df[column].std()
            mask = np.abs(z) < threshold
        else:
            mask = np.ones(len(df), dtype=bool)

        return df[mask].copy()

    # ----------------------------------------------------------------
    # FEATURE PREPARATION FOR ML
    # ----------------------------------------------------------------
    def prepare_ml_features(self, remove_outliers=True):
        """Prepare features X and target y for ML models."""
        # Select mature videos (6+ months)
        mature_df = self.df[self.df['months_since_publish'] >= 6].copy()

        print('\n📊 Dataset Preparation:')
        print(f"   Initial samples: {len(mature_df)}")

        if remove_outliers:
            orig_len = len(mature_df)
            mature_df = self.remove_outliers(mature_df, 'views', method='iqr', threshold=3)
            removed = orig_len - len(mature_df)
            if removed > 0:
                print(f"   Removed {removed} outliers (IQR method)")
                print(f"   Remaining samples: {len(mature_df)}")

        # Feature selection
        self.feature_columns = [
            'log_impressions', 'impressions_ctr', 'ctr_x_log_imp', 'duration_minutes',
            'is_weekend', 'sin_month', 'cos_month', 'sin_day', 'cos_day', 'content_type'
        ]

        # Encode content_type
        self.encoders['content_type'] = LabelEncoder()
        mature_df['content_type_encoded'] = self.encoders['content_type'].fit_transform(mature_df['content_type'])

        # Replace with encoded column
        feature_cols_final = [c if c != 'content_type' else 'content_type_encoded' for c in self.feature_columns]
        X = mature_df[feature_cols_final].copy()

        # Target
        if self.use_log_target:
            y = mature_df['log_views'].copy()
        else:
            y = mature_df['views'].copy()

        # Fill NaNs
        X = X.fillna(X.median())

        # Robust scaling
        self.scalers['robust'] = RobustScaler()
        X_scaled = self.scalers['robust'].fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols_final, index=X.index)

        return X_scaled, y, mature_df

    # ----------------------------------------------------------------
    # TRAIN MODELS
    # ----------------------------------------------------------------
    def train_models(self):
        """Train several models and print extensive metrics."""
        sep = '=' * 70
        dash = '─' * 70
        print(f"\n{sep}")
        print("TRAINING REGULARIZED ML MODELS")
        print(sep)

        X, y, mature_df = self.prepare_ml_features(remove_outliers=True)

        print(f"\n   Features: {len(X.columns)}")
        if self.use_log_target:
            print(f"   Target: log(views) - range: {y.min():.2f} - {y.max():.2f}")
        else:
            print(f"   Target: views - range: {y.min():,.0f} - {y.max():,.0f}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        print(f"   Train set: {len(X_train)} | Test set: {len(X_test)}")

        # Naive denom for MASE
        try:
            y_train_actual_for_mase = np.expm1(y_train) if self.use_log_target else y_train
            naive_diffs = np.abs(np.diff(y_train_actual_for_mase)) if len(y_train_actual_for_mase) > 1 else np.array([1.0])
            mase_denom = np.mean(naive_diffs) if len(naive_diffs) > 0 else 1.0
        except Exception:
            mase_denom = 1.0

        models_to_train = {
            'Ridge Regression': Ridge(alpha=10.0, random_state=42),
            'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=5000),
            'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, min_samples_split=10, min_samples_leaf=5, subsample=0.8, random_state=42)
        }

        results = []

        for model_name, model in models_to_train.items():
            print(f"\n{dash}")
            print(f"🤖 TRAINING: {model_name}")
            print(dash)

            # Fit model
            model.fit(X_train, y_train)

            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Back-transform if using log target
            if self.use_log_target:
                y_train_actual = np.expm1(y_train)
                y_test_actual = np.expm1(y_test)
                y_pred_train_actual = np.expm1(y_pred_train)
                y_pred_test_actual = np.expm1(y_pred_test)
            else:
                y_train_actual = y_train
                y_test_actual = y_test
                y_pred_train_actual = y_pred_train
                y_pred_test_actual = y_pred_test

            # Metrics
            train_r2 = r2_score(y_train_actual, y_pred_train_actual)
            test_r2 = r2_score(y_test_actual, y_pred_test_actual)
            test_mae = mean_absolute_error(y_test_actual, y_pred_test_actual)
            test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_test_actual))
            test_mape = np.mean(np.abs((y_test_actual - y_pred_test_actual) / np.maximum(y_test_actual, 1))) * 100
            test_median_ape = np.median(np.abs((y_test_actual - y_pred_test_actual) / np.maximum(y_test_actual, 1))) * 100
            test_mase = np.mean(np.abs(y_test_actual - y_pred_test_actual)) / mase_denom if mase_denom > 0 else np.nan

            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2', n_jobs=-1)

            overfit = train_r2 - test_r2

            print(f"\n  📊 Performance Metrics:")
            print(f"     Train R²:           {train_r2:>7.4f}")
            print(f"     Test R²:            {test_r2:>7.4f}")
            print(f"     CV R² (mean):       {cv_scores.mean():>7.4f} ± {cv_scores.std():.4f}")
            print(f"     Test MAE:           {test_mae:>12,.0f} views")
            print(f"     Test RMSE:          {test_rmse:>12,.0f} views")
            print(f"     Test MAPE:          {test_mape:>11.2f}%")
            print(f"     Test MASE:          {test_mase:>11.3f}")
            print(f"     Test Median APE:    {test_median_ape:>11.2f}%")

            if abs(overfit) > 0.15:
                print(f"     ⚠️  Overfit gap:        {overfit:>7.4f} (high)")
            elif abs(overfit) > 0.05:
                print(f"     ⚡ Overfit gap:        {overfit:>7.4f} (moderate)")
            else:
                print(f"     ✓ Good generalization (gap: {overfit:.4f})")

            # Feature importance or coefficients
            if hasattr(model, 'feature_importances_'):
                importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
                print('\n  🎯 Top 5 Features:')
                for idx, row in importances.head(5).iterrows():
                    print(f"     {row['feature']:<25} {row['importance']:.4f}")
            elif hasattr(model, 'coef_'):
                coef_abs = np.abs(model.coef_)
                top_idx = coef_abs.argsort()[-5:][::-1]
                print('\n  🎯 Top 5 Features (by coefficient magnitude):')
                for i in top_idx:
                    print(f"     {X.columns[i]:<25} {coef_abs[i]:.4f}")

            # Store
            self.models[model_name] = model
            results.append({
                'model': model_name,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_mape': test_mape,
                'test_mase': test_mase,
                'test_median_ape': test_median_ape,
                'overfit': overfit
            })

        # Model comparison
        print(f"\n{sep}")
        print("📊 MODEL COMPARISON")
        print(sep)

        results_df = pd.DataFrame(results)
        print(f"\n{'Model':<20} {'Test R²':>9} {'MAE':>12} {'MAPE':>10} {'MASE':>8} {'Median APE':>11} {'Overfit':>9}")
        print(dash)
        for _, r in results_df.iterrows():
            print(f"{r['model']:<20} {r['test_r2']:>9.4f} {r['test_mae']:>12,.0f} {r['test_mape']:>9.2f}% {r['test_mase']:>8.3f} {r['test_median_ape']:>10.1f}% {r['overfit']:>9.4f}")

        results_df['score'] = results_df['test_r2'] - 0.5 * np.abs(results_df['overfit'])
        best_idx = results_df['score'].idxmax()
        self.best_model_name = results_df.loc[best_idx, 'model']

        print(f"\n🏆 BEST MODEL: {self.best_model_name}")
        print(f"   Test R²:       {results_df.loc[best_idx, 'test_r2']:.4f}")
        print(f"   MAE:           {results_df.loc[best_idx, 'test_mae']:,.0f}")
        print(f"   MAPE:          {results_df.loc[best_idx, 'test_mape']:.2f}%")
        print(f"   MASE:          {results_df.loc[best_idx, 'test_mase']:.3f}")
        print(f"   Median APE:    {results_df.loc[best_idx, 'test_median_ape']:.1f}%")
        print(f"   Overfit Gap:   {results_df.loc[best_idx, 'overfit']:.4f}")

        # Visualizations
        try:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(results_df['model'], results_df['test_r2'], color='tab:blue')
            ax.set_title('Model Comparison - Test R² Scores')
            ax.set_ylabel('Test R²')
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"   ✗ Could not show model comparison plot: {e}")

        try:
            best_model = self.models[self.best_model_name]
            if hasattr(best_model, 'feature_importances_'):
                importances = pd.DataFrame({'feature': X.columns, 'importance': best_model.feature_importances_}).sort_values('importance', ascending=False)
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                ax2.barh(importances['feature'][:10], importances['importance'][:10], color='tab:orange')
                ax2.invert_yaxis()
                ax2.set_title(f'Top 10 Feature Importances - {self.best_model_name}')
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"   ✗ Could not show feature importance plot: {e}")

        return results_df

    # ----------------------------------------------------------------
    # PREDICTIONS
    # ----------------------------------------------------------------
    def predict_video(self, video_features):
        """Predict a single video's expected views (interpreted as ~6-month total)."""
        feature_dict = {
            'log_impressions': np.log1p(video_features.get('impressions', 0)),
            'impressions_ctr': video_features.get('impressions_ctr', 0.07),
            'ctr_x_log_imp': video_features.get('impressions_ctr', 0.07) * np.log1p(video_features.get('impressions', 0)),
            'duration_minutes': video_features.get('duration_minutes', 4.0),
            'is_weekend': 1 if video_features.get('day_of_week', 3) in [5, 6] else 0,
        }

        month = video_features.get('month', datetime.now().month)
        dow = video_features.get('day_of_week', 3)
        feature_dict['sin_month'] = np.sin(2 * np.pi * month / 12)
        feature_dict['cos_month'] = np.cos(2 * np.pi * month / 12)
        feature_dict['sin_day'] = np.sin(2 * np.pi * dow / 7)
        feature_dict['cos_day'] = np.cos(2 * np.pi * dow / 7)

        content_type = video_features.get('content_type', 'other')
        if content_type in self.encoders['content_type'].classes_:
            content_type_encoded = self.encoders['content_type'].transform([content_type])[0]
        else:
            content_type_encoded = 0
        feature_dict['content_type_encoded'] = content_type_encoded

        X_new = pd.DataFrame([feature_dict])
        X_new = X_new[self.scalers['robust'].feature_names_in_]
        X_new_scaled = self.scalers['robust'].transform(X_new)

        model = self.models[self.best_model_name]
        pred_log = model.predict(X_new_scaled)[0]
        pred = np.expm1(pred_log) if self.use_log_target else pred_log

        similar_mask = (self.df['content_type'] == content_type) & (self.df['months_since_publish'] >= 6)
        similar_views = self.df[similar_mask]['views']
        if len(similar_views) >= 3:
            std = similar_views.std()
            pessimistic = max(0, pred - std * 0.3)
            optimistic = pred + std * 0.3
        else:
            pessimistic = pred * 0.6
            optimistic = pred * 1.4

        return {'realistic': int(max(0, pred)), 'pessimistic': int(max(0, pessimistic)), 'optimistic': int(max(0, optimistic)), 'model_used': self.best_model_name}

    def predict_all_videos(self):
        """Generate predictions for all eligible videos and print accuracy stats."""
        sep = '=' * 70
        dash = '─' * 70
        print(f"\n{sep}")
        print("PREDICTIONS FOR ALL EXISTING VIDEOS")
        print(sep)

        predictions = []
        for idx, video in self.df.iterrows():
            if video['impressions'] <= 0 or video['months_since_publish'] < 1:
                continue

            features = {
                'impressions': video['impressions'],
                'impressions_ctr': video['impressions_ctr'],
                'duration_minutes': video['duration_minutes'],
                'day_of_week': video['day_of_week'],
                'month': video['publish_date'].month if pd.notnull(video['publish_date']) else datetime.now().month,
                'content_type': video['content_type']
            }

            pred = self.predict_video(features)
            error_pct = abs(video['views'] - pred['realistic']) / video['views'] * 100 if video['views'] > 0 else 0
            predictions.append({'title': video['video_title'][:45], 'type': video['content_type'], 'actual': video['views'], 'predicted': pred['realistic'], 'error_pct': error_pct, 'publish_date': video['publish_date'], 'index': idx})

        pred_df = pd.DataFrame(predictions)

        print(f"\n{'Title':<45} {'Type':<12} {'Actual':>10} {'Predicted':>10} {'Error':>7}")
        print(dash)
        for _, row in pred_df.nlargest(20, 'actual').iterrows():
            print(f"{row['title']:<45} {row['type']:<12} {row['actual']:>10,} {row['predicted']:>10,} {row['error_pct']:>6.1f}%")

        print('\n📊 PREDICTION ACCURACY:')
        print(f"   Median Error:  {pred_df['error_pct'].median():.1f}%")
        print(f"   Within ±30%:   {(pred_df['error_pct'] <= 30).sum()}/{len(pred_df)} ({(pred_df['error_pct'] <= 30).sum()/len(pred_df)*100:.1f}%)")
        print(f"   Within ±50%:   {(pred_df['error_pct'] <= 50).sum()}/{len(pred_df)} ({(pred_df['error_pct'] <= 50).sum()/len(pred_df)*100:.1f}%)")

        # Visuals
        try:
            plt.figure(figsize=(7, 7))
            plt.scatter(pred_df['actual'], pred_df['predicted'], alpha=0.7)
            lims = [min(pred_df['actual'].min(), pred_df['predicted'].min()), max(pred_df['actual'].max(), pred_df['predicted'].max())]
            plt.plot(lims, lims, 'r--')
            plt.title('Predicted vs Actual Views (All Videos)')
            plt.xlabel('Actual Views')
            plt.ylabel('Predicted Views')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"   ✗ Could not show predicted vs actual plot: {e}")

        try:
            plt.figure(figsize=(7, 4))
            plt.hist(pred_df['error_pct'].dropna(), bins=30, edgecolor='k', alpha=0.7)
            plt.title('Prediction Error Distribution (Absolute % error)')
            plt.xlabel('Absolute Percentage Error (%)')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"   ✗ Could not show error distribution plot: {e}")

        return pred_df

    def predict_unreleased(self, concept):
        """Produce forecast scenarios for an unreleased concept."""
        sep = '=' * 70
        dash = '─' * 70
        content_type = concept.get('content_type', 'other')
        print(f"\n{sep}\nUNRELEASED FORECAST: {content_type.upper()}\n{sep}")

        similar = self.df[(self.df['content_type'] == content_type) & (self.df['months_since_publish'] >= 6)]
        if len(similar) < 3:
            similar = self.df[self.df['months_since_publish'] >= 6]

        scenarios = {
            'Conservative (25th %ile)': int(similar['impressions'].quantile(0.25)),
            'Realistic (Median)': int(similar['impressions'].median()),
            'Optimistic (75th %ile)': int(similar['impressions'].quantile(0.75))
        }

        print(f"\n{'Scenario':<25} {'Impressions':>15} {'Predicted Views':>18}")
        print(dash)
        for scenario, impressions in scenarios.items():
            features = {**concept, 'impressions': impressions, 'impressions_ctr': similar['impressions_ctr'].median()}
            pred = self.predict_video(features)
            print(f"{scenario:<25} {impressions:>15,} {pred['realistic']:>18,}")

        realistic_imp = scenarios['Realistic (Median)']
        features = {**concept, 'impressions': realistic_imp, 'impressions_ctr': similar['impressions_ctr'].median()}
        pred = self.predict_video(features)

        print('\n🎯 BEST ESTIMATE:')
        print(f"   Expected Views:     {pred['realistic']:>12,}")
        print(f"   Confidence Range:   {pred['pessimistic']:>12,} - {pred['optimistic']:,}")
        print(f"   Model:              {pred['model_used']}")

        return pred

    def plot_cumulative_6month_forecast(self, pred_df):
        """Plot cumulative historical and a 6-month continuation forecast.

        This version produces a continuation that starts exactly at the
        current cumulative total and extends smoothly across 6 months
        using the predicted increment.
        """
        try:
            if pred_df.empty:
                print('   ✗ No predictions to plot cumulative chart.')
                return

            # Align and sort
            df_sorted = self.df.sort_values('publish_date').reset_index(drop=False)
            actuals_aligned = df_sorted['views'].reset_index(drop=True).astype(float)

            # Build predicted aligned series
            predicted_series = pd.Series(0, index=df_sorted.index, dtype=float)
            for _, row in pred_df.iterrows():
                if 'index' in row and not pd.isnull(row['index']):
                    orig_idx = int(row['index'])
                    matches = df_sorted[df_sorted['index'] == orig_idx]
                    if not matches.empty:
                        pos = matches.index[0]
                        predicted_series.at[pos] = row['predicted']

            actual_cum = actuals_aligned.cumsum().reset_index(drop=True)
            predicted_cum = predicted_series.cumsum().reset_index(drop=True)

            # Continuation logic: start at last actual cumulative, extend 6 months forward
            last_actual_total = actual_cum.iloc[-1] if len(actual_cum) > 0 else 0
            last_pred_total = predicted_cum.iloc[-1] if len(predicted_cum) > 0 else last_actual_total
            increment = last_pred_total - last_actual_total

            # Create monthly steps for 6 months
            last_date = df_sorted['publish_date'].iloc[-1] if not df_sorted['publish_date'].isna().all() else datetime.now()
            future_dates = pd.date_range(start=last_date, periods=7, freq='M')  # includes month 0..6
            projected = np.linspace(last_actual_total, last_actual_total + increment, num=len(future_dates))

            # Plot
            plt.figure(figsize=(11, 6))
            plt.plot(df_sorted['publish_date'].reset_index(drop=True), actual_cum, label='Historical Cumulative Views', linewidth=2)
            plt.plot(future_dates, projected, '--', label='Forecast Continuation (6 months)', linewidth=2)
            plt.scatter([df_sorted['publish_date'].iloc[-1]], [last_actual_total], color='tab:blue', zorder=5)
            plt.scatter([future_dates[-1]], [projected[-1]], color='tab:orange', zorder=5)
            plt.title('Cumulative Actual vs Forecast Continuation (6 months)')
            plt.xlabel('Publish Date')
            plt.ylabel('Cumulative Views')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"   ✗ Could not show cumulative 6-month forecast plot: {e}")

    def run_pipeline(self):
        """Execute entire pipeline end-to-end including forecast plots."""
        header = '#' * 70
        print(f"\n{header}")
        print('IMPROVED YOUTUBE ML PREDICTOR V2')
        print('With Regularization & Outlier Handling')
        print(header)

        if not self.load_and_prepare_data():
            return False

        # Train models
        self.train_models()

        # Predict existing videos
        pred_df = self.predict_all_videos()

        # Show cumulative continuation
        self.plot_cumulative_6month_forecast(pred_df)

        # Unreleased forecasts
        print(f"\n{header}")
        print('UNRELEASED VIDEO FORECASTS')
        print(header)

        concepts = [
            {'content_type': 'music_video', 'duration_minutes': 4.5, 'day_of_week': 3, 'month': 10},
            {'content_type': 'lyrics', 'duration_minutes': 4.0, 'day_of_week': 3, 'month': 10},
        ]
        for c in concepts:
            self.predict_unreleased(c)

        print(f"\n{header}")
        print('✅ PIPELINE COMPLETE')
        print(f"{header}\n")
        return True


# --------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------
if __name__ == '__main__':
    predictor = ImprovedYouTubeMLPredictor(db_params)
    predictor.run_pipeline()

# --------------------------------------------------------------------
# End of file - this script intentionally verbose to preserve production
# pipeline style and report format. It should be runnable as-is provided
# the database credentials and dependencies are available.
# --------------------------------------------------------------------
