# =============================================================================
# FACEBOOK REACH FORECASTER - ADVANCED ML WITH POST-LEVEL PREDICTION
# =============================================================================

import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
from scipy.stats import norm as norm_dist
warnings.filterwarnings("ignore")

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
db_params = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host': 'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port': '5432',
    'sslmode': 'require'
}


# =============================================================================
# FETCH DATA
# =============================================================================
def fetch_data():
    print("Connecting to database...")
    conn = psycopg2.connect(**db_params)
    query = """
    SELECT publish_time, post_type, reach, shares, comments, reactions,
           impressions, seconds_viewed, average_seconds_viewed, duration_sec
    FROM facebook_data_set
    WHERE publish_time IS NOT NULL
    ORDER BY publish_time;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['publish_time'] = pd.to_datetime(df['publish_time'])
    for col in ['reach', 'shares', 'comments', 'reactions', 'impressions', 'seconds_viewed', 'average_seconds_viewed', 'duration_sec']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['engagement'] = df['reactions'] + df['comments'] + df['shares']
    
    print(f"✓ Loaded {len(df)} posts from {df['publish_time'].min().date()} → {df['publish_time'].max().date()}")
    return df

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    """Create rich feature set for ML"""
    
    print(f"\n{'='*80}")
    print("FEATURE ENGINEERING")
    print(f"{'='*80}")
    
    df = df.copy()
    
    # Time-based features
    df['year'] = df['publish_time'].dt.year
    df['month'] = df['publish_time'].dt.month
    df['day_of_week'] = df['publish_time'].dt.dayofweek
    df['hour'] = df['publish_time'].dt.hour
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quarter'] = df['publish_time'].dt.quarter
    
    # Cyclical encoding for temporal features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Days since first post
    df['days_since_start'] = (df['publish_time'] - df['publish_time'].min()).dt.days
    
    # Video-specific features
    df['is_video'] = df['post_type'].isin(['Videos', 'Reels']).astype(int)
    df['video_duration'] = df['duration_sec'].fillna(0)
    df['has_duration'] = (df['duration_sec'] > 0).astype(int)
    
    # Post type encoding
    df['post_type_encoded'] = LabelEncoder().fit_transform(df['post_type'])
    
    # Historical performance features (rolling averages)
    df = df.sort_values('publish_time')
    
    # Last 7 days performance
    df['reach_7d_mean'] = df['reach'].rolling(window=7, min_periods=1).mean().shift(1)
    df['reach_7d_std'] = df['reach'].rolling(window=7, min_periods=1).std().shift(1).fillna(0)
    
    # Last 30 days performance
    df['reach_30d_mean'] = df['reach'].rolling(window=30, min_periods=1).mean().shift(1)
    df['reach_30d_max'] = df['reach'].rolling(window=30, min_periods=1).max().shift(1)
    
    # Post type performance (expanding window)
    for post_type in df['post_type'].unique():
        mask = df['post_type'] == post_type
        df.loc[mask, 'post_type_avg_reach'] = df.loc[mask, 'reach'].expanding().mean().shift(1)
    
    df['post_type_avg_reach'] = df['post_type_avg_reach'].fillna(df['reach'].mean())
    
    # Posts this month so far
    df['year_month'] = df['publish_time'].dt.to_period('M')
    df['posts_this_month'] = df.groupby('year_month').cumcount()
    
    # Fill any remaining NaN values
    df = df.fillna(0)
    
    print(f"✓ Created {len([c for c in df.columns if c not in ['publish_time', 'post_type', 'reach', 'engagement', 'year_month']])} features")
    
    return df

# =============================================================================
# CLEAN DATA (REMOVE OUTLIERS)
# =============================================================================
def clean_data(df, min_posts_per_month=5, outlier_percentile=95):
    """Remove incomplete months and extreme viral outliers"""
    
    print(f"\n{'='*80}")
    print("DATA CLEANING")
    print(f"{'='*80}")
    
    # Remove incomplete months
    df['year_month'] = df['publish_time'].dt.to_period('M')
    monthly_counts = df.groupby('year_month').size()
    complete_months = monthly_counts[monthly_counts >= min_posts_per_month].index
    df_complete = df[df['year_month'].isin(complete_months)].copy()
    
    print(f"\nMonths: {len(monthly_counts)} total → {len(complete_months)} complete")
    print(f"Posts: {len(df)} → {len(df_complete)} (removed {len(df) - len(df_complete)} from incomplete months)")
    
    # Remove top 5% outliers PER POST TYPE (extreme viral posts)
    df_clean = pd.DataFrame()
    outliers_list = []
    
    print(f"\nRemoving top {100-outlier_percentile}% outliers per post type:")
    print(f"{'Type':<10} {'Total':>8} {'Outliers':>10} {'Kept':>8} {'Threshold':>15}")
    print("-" * 60)
    
    for post_type in df_complete['post_type'].unique():
        type_df = df_complete[df_complete['post_type'] == post_type].copy()
        
        threshold = type_df['reach'].quantile(outlier_percentile / 100)
        
        outliers = type_df[type_df['reach'] > threshold]
        clean = type_df[type_df['reach'] <= threshold]
        
        outliers_list.append(outliers)
        df_clean = pd.concat([df_clean, clean])
        
        print(f"{post_type:<10} {len(type_df):>8} {len(outliers):>10} {len(clean):>8} {threshold:>15,.0f}")
    
    all_outliers = pd.concat(outliers_list) if outliers_list else pd.DataFrame()
    
    print(f"\nSummary: {len(df_clean)} clean posts, {len(all_outliers)} outliers removed")
    
    return df_clean.drop(columns=['year_month']), all_outliers

# =============================================================================
# ADVANCED ML MODEL WITH ENSEMBLE
# =============================================================================
class EnsembleReachPredictor:
    """
    Ensemble of multiple ML models for robust predictions
    """
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42
            ),
            'ridge': Ridge(alpha=10.0)
        }
        
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.weights = None
        self.trained = False
        
    def fit(self, X, y):
        """Train ensemble with adaptive weighting"""
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train each model
        for name, model in self.models.items():
            model.fit(X_scaled, y)
        
        # Calculate weights based on out-of-bag performance
        # Use last 20% of data as validation
        split_idx = int(len(X) * 0.8)
        X_val = X_scaled[split_idx:]
        y_val = y.iloc[split_idx:]
        
        errors = {}
        for name, model in self.models.items():
            pred = model.predict(X_val)
            errors[name] = mean_absolute_error(y_val, pred)
        
        # Inverse error weighting
        total_inv_error = sum(1/e for e in errors.values())
        self.weights = {name: (1/errors[name])/total_inv_error for name in errors.keys()}
        
        self.feature_cols = X.columns.tolist()
        self.trained = True
        
        return self
    
    def predict(self, X):
        """Ensemble prediction with weighted averaging"""
        if not self.trained:
            raise ValueError("Model must be fit before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            pred = np.maximum(pred, 0)  # No negative reach
            predictions.append(pred * self.weights[name])
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0)
        
        return ensemble_pred
    
    def get_feature_importance(self):
        """Get feature importance from tree-based models"""
        if not self.trained:
            return None
        
        importance = {}
        
        if 'rf' in self.models:
            rf_importance = self.models['rf'].feature_importances_
            for i, col in enumerate(self.feature_cols):
                importance[col] = importance.get(col, 0) + rf_importance[i] * 0.5
        
        if 'gb' in self.models:
            gb_importance = self.models['gb'].feature_importances_
            for i, col in enumerate(self.feature_cols):
                importance[col] = importance.get(col, 0) + gb_importance[i] * 0.5
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

# =============================================================================
# BACKTESTING WITH POST-LEVEL PREDICTION
# =============================================================================
def backtest_model(df, n_splits=5, forecast_horizon_months=3):
    """
    Backtest by training on posts, predicting future posts, then aggregating
    """
    
    print(f"\n{'='*80}")
    print(f"BACKTESTING - {forecast_horizon_months}-MONTH AHEAD FORECASTS")
    print(f"{'='*80}")
    
    # Prepare features
    df = engineer_features(df)
    
    feature_cols = [
        'year', 'month', 'day_of_week', 'hour', 'is_weekend', 'quarter',
        'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'days_since_start', 'is_video', 'video_duration', 'has_duration',
        'post_type_encoded', 'reach_7d_mean', 'reach_7d_std', 'reach_30d_mean',
        'reach_30d_max', 'post_type_avg_reach', 'posts_this_month'
    ]
    
    # Aggregate to monthly for evaluation
    df['year_month'] = df['publish_time'].dt.to_period('M')
    monthly = df.groupby('year_month').agg({'reach': 'sum'})
    
    min_train_months = 12
    if len(monthly) < min_train_months + forecast_horizon_months:
        print(f"⚠ Insufficient data for {forecast_horizon_months}-month forecasting")
        return None
    
    results = []
    
    print(f"\n{'Split':>5} {'Train':>8} {'Test':>8} {'Actual':>15} {'Predicted':>15} {'Error':>15} {'MAPE':>10}")
    print("-" * 90)
    
    # Time series splits
    n_test_splits = min(n_splits, len(monthly) - min_train_months - forecast_horizon_months + 1)
    
    for i in range(n_test_splits):
        test_end_idx = len(monthly) - i * forecast_horizon_months
        test_start_idx = test_end_idx - forecast_horizon_months
        train_end_idx = test_start_idx
        
        if train_end_idx < min_train_months:
            break
        
        train_periods = monthly.index[:train_end_idx]
        test_periods = monthly.index[test_start_idx:test_end_idx]
        
        # Get post-level train/test data
        df_train = df[df['year_month'].isin(train_periods)].copy()
        df_test = df[df['year_month'].isin(test_periods)].copy()
        
        if len(df_test) == 0:
            continue
        
        # Train model on posts
        X_train = df_train[feature_cols]
        y_train = df_train['reach']
        
        X_test = df_test[feature_cols]
        
        model = EnsembleReachPredictor()
        model.fit(X_train, y_train)
        
        # Predict each post in test period
        post_predictions = model.predict(X_test)
        
        # Aggregate to monthly
        df_test['predicted_reach'] = post_predictions
        monthly_pred = df_test.groupby('year_month')['predicted_reach'].sum()
        
        # Get actual monthly reach
        actual = monthly.loc[test_periods, 'reach'].sum()
        predicted = monthly_pred.sum()
        
        error = predicted - actual
        mape = abs(error / actual * 100) if actual > 0 else 0
        
        results.append({
            'actual': actual,
            'predicted': predicted,
            'error': error,
            'mape': mape,
            'test_posts': len(df_test)
        })
        
        print(f"{i+1:>5} {len(train_periods):>8} {len(test_periods):>8} "
              f"{actual:>15,.0f} {predicted:>15,.0f} {error:>15,.0f} {mape:>9.1f}%")
    
    if not results:
        print("⚠ Not enough data for backtesting")
        return None
    
    # Summary
    results_df = pd.DataFrame(results)
    
    print("-" * 90)
    print(f"{'AVG':>5} {'':<8} {'':<8} "
          f"{results_df['actual'].mean():>15,.0f} "
          f"{results_df['predicted'].mean():>15,.0f} "
          f"{results_df['error'].mean():>15,.0f} "
          f"{results_df['mape'].mean():>9.1f}%")
    
    # Calculate MASE
    naive_errors = np.abs(np.diff(results_df['actual']))
    naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else 1
    mae = results_df['error'].abs().mean()
    mase = mae / naive_mae if naive_mae > 0 else np.nan
    
    print(f"\n{'='*80}")
    print("BACKTEST SUMMARY")
    print(f"{'='*80}")
    print(f"Mean Absolute Error (MAE): {mae:,.0f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(results_df['actual'], results_df['predicted'])):,.0f}")
    print(f"Mean Absolute % Error (MAPE): {results_df['mape'].mean():.1f}%")
    print(f"Median Absolute % Error: {results_df['mape'].median():.1f}%")
    print(f"Mean Absolute Scaled Error (MASE): {mase:.3f}")
    print(f"R² Score: {r2_score(results_df['actual'], results_df['predicted']):.3f}")
    
    mean_error = results_df['error'].mean()
    bias = "overestimates" if mean_error > 0 else "underestimates"
    print(f"Bias: Model {bias} by {abs(mean_error):,.0f} on average")
    
    return results_df

# =============================================================================
# GENERATE FORECAST
# =============================================================================
def generate_forecast(df, months_ahead=6):
    """Generate future forecast"""
    
    print(f"\n{'='*80}")
    print(f"{months_ahead}-MONTH FORECAST")
    print(f"{'='*80}")
    
    # Engineer features
    df = engineer_features(df)
    
    feature_cols = [
        'year', 'month', 'day_of_week', 'hour', 'is_weekend', 'quarter',
        'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'days_since_start', 'is_video', 'video_duration', 'has_duration',
        'post_type_encoded', 'reach_7d_mean', 'reach_7d_std', 'reach_30d_mean',
        'reach_30d_max', 'post_type_avg_reach', 'posts_this_month'
    ]
    
    # Train on all data
    X_train = df[feature_cols]
    y_train = df['reach']
    
    model = EnsembleReachPredictor()
    model.fit(X_train, y_train)
    
    print(f"\n✓ Trained ensemble model on {len(df)} posts")
    
    # Feature importance
    importance = model.get_feature_importance()
    print(f"\nTop 10 Most Important Features:")
    for i, (feat, imp) in enumerate(list(importance.items())[:10], 1):
        print(f"  {i}. {feat}: {imp:.4f}")
    
    # Calculate recent posting patterns
    recent_months = 6
    cutoff = df['publish_time'].max() - pd.DateOffset(months=recent_months)
    recent = df[df['publish_time'] >= cutoff]
    
    post_type_rates = recent.groupby('post_type').size() / recent_months
    
    print(f"\nRecent Posting Patterns (last {recent_months} months):")
    print(f"{'Type':<10} {'Posts/Month':>12}")
    print("-" * 25)
    for post_type, rate in post_type_rates.items():
        print(f"{post_type:<10} {rate:>12.1f}")
    
    # Generate future post scenarios
    print(f"\nGenerating {months_ahead}-month forecast...")
    
    last_date = df['publish_time'].max()
    monthly_forecasts = []
    
    for month_offset in range(1, months_ahead + 1):
        future_date = last_date + pd.DateOffset(months=month_offset)
        month_reach = 0
        month_posts = []
        
        # Simulate posts for this month based on recent rates
        for post_type, rate in post_type_rates.items():
            n_posts = int(np.round(rate))
            
            for _ in range(n_posts):
                # Create feature vector for hypothetical post
                # Use typical posting time patterns from historical data
                typical_hour = df[df['post_type'] == post_type]['hour'].mode()[0] if len(df[df['post_type'] == post_type]) > 0 else 12
                typical_dow = df[df['post_type'] == post_type]['day_of_week'].mode()[0] if len(df[df['post_type'] == post_type]) > 0 else 3
                
                post_features = {
                    'year': future_date.year,
                    'month': future_date.month,
                    'day_of_week': typical_dow,
                    'hour': typical_hour,
                    'is_weekend': 1 if typical_dow in [5, 6] else 0,
                    'quarter': future_date.quarter,
                    'month_sin': np.sin(2 * np.pi * future_date.month / 12),
                    'month_cos': np.cos(2 * np.pi * future_date.month / 12),
                    'hour_sin': np.sin(2 * np.pi * typical_hour / 24),
                    'hour_cos': np.cos(2 * np.pi * typical_hour / 24),
                    'dow_sin': np.sin(2 * np.pi * typical_dow / 7),
                    'dow_cos': np.cos(2 * np.pi * typical_dow / 7),
                    'days_since_start': (future_date - df['publish_time'].min()).days,
                    'is_video': 1 if post_type in ['Videos', 'Reels'] else 0,
                    'video_duration': df[df['post_type'] == post_type]['video_duration'].median(),
                    'has_duration': 1 if post_type in ['Videos', 'Reels'] else 0,
                    'post_type_encoded': LabelEncoder().fit(df['post_type']).transform([post_type])[0],
                    'reach_7d_mean': df['reach'].tail(7).mean(),
                    'reach_7d_std': df['reach'].tail(7).std(),
                    'reach_30d_mean': df['reach'].tail(30).mean(),
                    'reach_30d_max': df['reach'].tail(30).max(),
                    'post_type_avg_reach': df[df['post_type'] == post_type]['reach'].mean(),
                    'posts_this_month': _ + 1
                }
                
                month_posts.append(post_features)
        
        # Predict reach for all posts in this month
        if month_posts:
            X_future = pd.DataFrame(month_posts)[feature_cols]
            predictions = model.predict(X_future)
            month_reach = predictions.sum()
        
        monthly_forecasts.append({
            'month': month_offset,
            'reach': month_reach,
            'posts': len(month_posts)
        })
    
    # Display forecast
    print(f"\n{'='*80}")
    print("FORECAST BREAKDOWN")
    print(f"{'='*80}")
    
    print(f"\n{'Month':>6} {'Est. Posts':>12} {'Predicted Reach':>18}")
    print("-" * 40)
    
    total_reach = 0
    for f in monthly_forecasts:
        print(f"{f['month']:>6} {f['posts']:>12} {f['reach']:>18,.0f}")
        total_reach += f['reach']
    
    print("-" * 40)
    print(f"{'TOTAL':>6} {sum(f['posts'] for f in monthly_forecasts):>12} {total_reach:>18,.0f}")
    
    # Confidence interval based on historical variance
    monthly_hist = df.groupby(df['publish_time'].dt.to_period('M'))['reach'].sum()
    monthly_std = monthly_hist.std()
    
    z_score = norm_dist.ppf(0.975)  # 95% CI
    margin = z_score * monthly_std * np.sqrt(months_ahead)
    
    print(f"\n95% Confidence Interval:")
    print(f"  Lower Bound:  {max(0, total_reach - margin):>15,.0f}")
    print(f"  Expected:     {total_reach:>15,.0f}")
    print(f"  Upper Bound:  {total_reach + margin:>15,.0f}")
    
    return {
        'total': total_reach,
        'monthly': monthly_forecasts,
        'lower': max(0, total_reach - margin),
        'upper': total_reach + margin,
        'model': model
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*80)
    print("ADVANCED ML FACEBOOK REACH FORECASTER")
    print("="*80)
    
    # Load data
    df_raw = fetch_data()
    
    # Clean data (remove extreme outliers only)
    df_clean, outliers = clean_data(df_raw, min_posts_per_month=5, outlier_percentile=95)
    
    print(f"\n✓ Using {len(df_clean)} posts for training ({len(outliers)} extreme outliers excluded)")
    
    # Backtest
    backtest_results = backtest_model(df_clean, n_splits=5, forecast_horizon_months=3)
    
    # Generate forecast
    if backtest_results is not None:
        forecast = generate_forecast(df_clean, months_ahead=6)
        
        # Final recommendations
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print(f"{'='*80}")
        
        mape = backtest_results['mape'].median()
        
        if mape < 30:
            confidence = "HIGH"
        elif mape < 50:
            confidence = "MODERATE"
        else:
            confidence = "LOW"
        
        print(f"\nForecast Confidence: {confidence}")
        print(f"Historical MAPE: {mape:.1f}%")
        print(f"\nKey Insights:")
        print(f"  • Model trained on {len(df_clean)} posts")
        print(f"  • Excluded {len(outliers)} extreme viral outliers (top 5%)")
        print(f"  • Ensemble of Random Forest + Gradient Boosting + Ridge")
        print(f"  • Predictions based on post-level patterns, then aggregated")
        
        print(f"\n{'='*80}\n")
        
        return forecast, backtest_results
    else:
        print("\n⚠ Insufficient data for forecasting")
        return None, None

if __name__ == "__main__":
    forecast, backtest = main()

# ==============================================================================
# VISUALIZATION SECTION (Fixed + With Forecast Markers + 3rd-Point Connector)
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import psycopg2

sns.set(style="whitegrid", font_scale=1.2)

print("\n================================================================================")
print("VISUALIZATION")
print("================================================================================")

# 1️⃣ BACKTEST VISUALIZATION
if backtest is not None and not backtest.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(backtest['actual'], label='Actual Reach', marker='o', linewidth=2)
    plt.plot(backtest['predicted'], label='Predicted Reach', marker='s', linewidth=2)
    plt.title('Backtest Forecast: Actual vs Predicted Reach (3-Month Rolling)', fontsize=14)
    plt.xlabel('Backtest Split (Time Order)')
    plt.ylabel('Monthly Reach')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("⚠ No backtest data available for visualization")

# 2️⃣ FORECAST VISUALIZATION (Continuous Timeline)
if forecast is not None and 'monthly' in forecast:
    print("Generating enhanced historical + forecast visualization...")

    # Pull historical reach from database
    conn = psycopg2.connect(**db_params)
    hist_query = """
        SELECT publish_time, reach
        FROM facebook_data_set
        WHERE publish_time IS NOT NULL
        ORDER BY publish_time;
    """
    df_hist = pd.read_sql_query(hist_query, conn)
    conn.close()

    df_hist['publish_time'] = pd.to_datetime(df_hist['publish_time'])
    df_hist['year_month'] = df_hist['publish_time'].dt.to_period('M')
    monthly_hist = df_hist.groupby('year_month')['reach'].sum().reset_index()
    monthly_hist['year_month'] = monthly_hist['year_month'].dt.to_timestamp()

    # Select last 6 months of history
    last_6m = monthly_hist.tail(6).copy()

    # Build forecast data
    last_hist_date = last_6m['year_month'].max()
    forecast_months = [last_hist_date + pd.DateOffset(months=i) for i in range(1, len(forecast['monthly']) + 1)]
    forecast_values = [m['reach'] for m in forecast['monthly']]
    forecast_df = pd.DataFrame({'year_month': forecast_months, 'forecast_reach': forecast_values})

    # Plot
    plt.figure(figsize=(14, 7))

    # Historical (solid green line with dots)
    plt.plot(last_6m['year_month'], last_6m['reach'],
             label='Historical Reach (Last 6 Months)',
             color='#2E8B57', linewidth=2.5, marker='o')

    # Connector between 3rd historical point and first forecast (black dashed)
    connector_start = last_6m.iloc[2]   # 3rd historical point (0-based index)
    connector_end = forecast_df.iloc[0] # 1st forecast month
    plt.plot(
        [connector_start['year_month'], connector_end['year_month']],
        [connector_start['reach'], connector_end['forecast_reach']],
        color='black', linestyle='--', linewidth=2, alpha=0.8
    )

    # Forecast (dashed blue line with markers)
    plt.plot(forecast_df['year_month'], forecast_df['forecast_reach'],
             label='Forecasted Reach (Next 6 Months)',
             color='#1E90FF', linewidth=2.5, linestyle='--', marker='o', markersize=6)

    # Confidence band (±20%)
    plt.fill_between(forecast_df['year_month'],
                     np.array(forecast_df['forecast_reach']) * 0.8,
                     np.array(forecast_df['forecast_reach']) * 1.2,
                     color='#1E90FF', alpha=0.2, label='Confidence Range')

    # Axis & style tweaks
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45, ha='right')

    plt.title('Facebook Reach: Last 6 Months (Actual) + Next 6 Months (Forecast)', fontsize=16, pad=20)
    plt.xlabel('Month')
    plt.ylabel('Monthly Reach')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='upper right', frameon=True)
    plt.tight_layout()
    plt.show()

else:
    print("⚠ No forecast data available for visualization")
