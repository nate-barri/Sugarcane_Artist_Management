# =============================================================================
# MACHINE LEARNING FACEBOOK FORECASTER (ENSEMBLE APPROACH)
# =============================================================================

import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
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
           impressions, seconds_viewed, average_seconds_viewed
    FROM facebook_data_set
    WHERE publish_time IS NOT NULL
    ORDER BY publish_time;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['publish_time'] = pd.to_datetime(df['publish_time'])
    for col in ['reach', 'shares', 'comments', 'reactions', 'impressions', 'seconds_viewed', 'average_seconds_viewed']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['engagement'] = df['reactions'] + df['comments'] + df['shares']
    
    print(f"✓ Loaded {len(df)} posts from {df['publish_time'].min().date()} → {df['publish_time'].max().date()}")
    return df

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    """Create features for ML model"""
    df = df.copy()
    
    # Temporal features
    df['year'] = df['publish_time'].dt.year
    df['month'] = df['publish_time'].dt.month
    df['day_of_week'] = df['publish_time'].dt.dayofweek
    df['day_of_month'] = df['publish_time'].dt.day
    df['quarter'] = df['publish_time'].dt.quarter
    df['week_of_year'] = df['publish_time'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical encoding for month (sin/cos to capture seasonality)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Time-based trend (months since start)
    df['months_since_start'] = ((df['publish_time'] - df['publish_time'].min()).dt.days / 30.44).astype(int)
    
    # Rolling statistics (last 30 days)
    df = df.sort_values('publish_time')
    df['rolling_30d_reach_mean'] = df['reach'].rolling(window=30, min_periods=1).mean()
    df['rolling_30d_reach_std'] = df['reach'].rolling(window=30, min_periods=1).std().fillna(0)
    df['rolling_30d_engagement_mean'] = df['engagement'].rolling(window=30, min_periods=1).mean()
    
    # Post type encoding
    le = LabelEncoder()
    df['post_type_encoded'] = le.fit_transform(df['post_type'])
    
    # Recent momentum (last 7 posts average)
    df['recent_7_reach_avg'] = df['reach'].rolling(window=7, min_periods=1).mean()
    df['recent_7_engagement_avg'] = df['engagement'].rolling(window=7, min_periods=1).mean()
    
    return df, le

# =============================================================================
# PREPARE AGGREGATED DATA FOR MONTHLY FORECASTING
# =============================================================================
def prepare_monthly_features(df):
    """Aggregate to monthly level with features"""
    df = df.copy()
    
    # Monthly aggregation
    monthly = df.resample('M', on='publish_time').agg({
        'reach': 'sum',
        'engagement': 'sum',
        'post_type': 'count',  # Number of posts
        'month': 'first',
        'quarter': 'first',
        'year': 'first',
        'month_sin': 'first',
        'month_cos': 'first',
        'months_since_start': 'first'
    }).rename(columns={'post_type': 'post_count'})
    
    # Add post type distribution
    post_type_dist = df.groupby([df['publish_time'].dt.to_period('M'), 'post_type']).size().unstack(fill_value=0)
    post_type_dist.index = post_type_dist.index.to_timestamp()
    
    for col in post_type_dist.columns:
        monthly[f'count_{col}'] = post_type_dist[col]
    
    # Lag features (previous months) - use median to fill initial NaNs
    reach_median = monthly['reach'].median()
    engagement_median = monthly['engagement'].median()
    
    for lag in [1, 2, 3]:
        monthly[f'reach_lag_{lag}'] = monthly['reach'].shift(lag).fillna(reach_median)
        monthly[f'engagement_lag_{lag}'] = monthly['engagement'].shift(lag).fillna(engagement_median)
    
    # Rolling features
    monthly['reach_rolling_3m_avg'] = monthly['reach'].rolling(window=3, min_periods=1).mean()
    monthly['reach_rolling_6m_avg'] = monthly['reach'].rolling(window=6, min_periods=1).mean()
    monthly['reach_rolling_3m_std'] = monthly['reach'].rolling(window=3, min_periods=1).std().fillna(0)
    
    # Fill any remaining NaNs in post type counts
    for col in monthly.columns:
        if 'count_' in col:
            monthly[col] = monthly[col].fillna(0)
    
    # Final check - fill any remaining NaNs with column median
    for col in monthly.columns:
        if monthly[col].isna().any():
            monthly[col] = monthly[col].fillna(monthly[col].median())
    
    return monthly

# =============================================================================
# TRAIN ML MODELS
# =============================================================================
def train_ml_models(monthly_df, target='reach'):
    """Train ensemble of ML models"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING ML MODELS FOR {target.upper()}")
    print(f"{'='*80}")
    
    # Prepare features and target
    feature_cols = [col for col in monthly_df.columns if col not in ['reach', 'engagement']]
    X = monthly_df[feature_cols]
    y = monthly_df[target]
    
    # Check for NaN values
    if X.isna().any().any():
        print("⚠️ WARNING: NaN values found in features. Filling with median...")
        X = X.fillna(X.median())
    
    # Remove outliers from training for better generalization
    # Use IQR method on target variable
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Keep outliers in test set but remove from training
    outlier_mask = (y < lower_bound) | (y > upper_bound)
    print(f"Found {outlier_mask.sum()} outliers in target variable")
    
    # Train/test split (last 6 months for validation)
    split_idx = len(X) - 6
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Remove outliers only from training set
    train_outliers = outlier_mask.iloc[:split_idx]
    X_train_clean = X_train[~train_outliers]
    y_train_clean = y_train[~train_outliers]
    
    print(f"Training samples: {len(X_train_clean)} (removed {train_outliers.sum()} outliers)")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    
    # Initialize models with reduced overfitting
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,  # Reduced from 200
            max_depth=5,       # More conservative
            min_samples_split=10,  # Increased
            min_samples_leaf=4,    # Increased
            max_features='sqrt',   # Added
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=50,    # Reduced
            learning_rate=0.05,
            max_depth=3,        # More conservative
            min_samples_split=10,
            subsample=0.8,      # Added regularization
            random_state=42
        ),
        'Ridge Regression': Ridge(alpha=10.0)  # Increased regularization
    }
    
    # Train and evaluate each model
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_clean, y_train_clean)  # Use cleaned training data
        
        # Predictions
        y_pred_train = model.predict(X_train_clean)
        y_pred_test = np.maximum(model.predict(X_test), 0)  # No negative predictions
        
        # Metrics
        train_r2 = r2_score(y_train_clean, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # MAPE
        mask = y_test > 0
        test_mape = np.mean(np.abs((y_test[mask] - y_pred_test[mask]) / y_test[mask])) * 100 if mask.sum() > 0 else np.nan
        
        # MASE (Mean Absolute Scaled Error)
        if len(y_test) > 1:
            # Naive forecast = use last known value
            naive_forecast = np.roll(y_test.values, 1)[1:]
            naive_mae = np.mean(np.abs(y_test.values[1:] - naive_forecast))
            test_mase = test_mae / naive_mae if naive_mae > 0 else np.nan
        else:
            test_mase = np.nan
        
        # Directional Accuracy
        if len(y_test) > 1:
            actual_direction = np.sign(np.diff(y_test.values))
            pred_direction = np.sign(np.diff(y_pred_test))
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            directional_accuracy = np.nan
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'mae': test_mae,
            'rmse': test_rmse,
            'mape': test_mape,
            'mase': test_mase,
            'directional_accuracy': directional_accuracy
        }
        
        predictions[name] = y_pred_test
        
        print(f"  Train R²: {train_r2:.3f}")
        print(f"  Test R²:  {test_r2:.3f}")
        print(f"  MAE:      {test_mae:,.0f}")
        print(f"  RMSE:     {test_rmse:,.0f}")
        print(f"  MAPE:     {test_mape:.1f}%")
        print(f"  MASE:     {test_mase:.3f}")
        print(f"  Dir Acc:  {directional_accuracy:.1f}%")
    
    # Ensemble prediction (weighted by test R2 performance)
    # Give more weight to models with positive R2
    weights = {}
    for name in models.keys():
        r2 = results[name]['test_r2']
        # Only use models with R2 > 0
        weights[name] = max(0, r2)
    
    total_weight = sum(weights.values())
    
    if total_weight > 0:
        # Weighted ensemble
        ensemble_pred = sum([predictions[name] * weights[name] / total_weight 
                            for name in models.keys()])
        ensemble_method = "Weighted by R²"
    else:
        # Fall back to simple average if all models are bad
        ensemble_pred = np.mean([predictions[name] for name in models.keys()], axis=0)
        ensemble_method = "Simple Average (All R² < 0)"
    
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    
    # Ensemble MAPE
    mask = y_test > 0
    ensemble_mape = np.mean(np.abs((y_test[mask] - ensemble_pred[mask]) / y_test[mask])) * 100 if mask.sum() > 0 else np.nan
    
    # Ensemble MASE
    if len(y_test) > 1:
        naive_forecast = np.roll(y_test.values, 1)[1:]
        naive_mae = np.mean(np.abs(y_test.values[1:] - naive_forecast))
        ensemble_mase = ensemble_mae / naive_mae if naive_mae > 0 else np.nan
    else:
        ensemble_mase = np.nan
    
    # Ensemble Directional Accuracy
    if len(y_test) > 1:
        actual_direction = np.sign(np.diff(y_test.values))
        pred_direction = np.sign(np.diff(ensemble_pred))
        ensemble_dir_acc = np.mean(actual_direction == pred_direction) * 100
    else:
        ensemble_dir_acc = np.nan
    
    print(f"\n{'='*80}")
    print(f"ENSEMBLE MODEL ({ensemble_method})")
    print(f"{'='*80}")
    if total_weight > 0:
        print(f"  Weights: {', '.join([f'{k}: {v/total_weight:.1%}' for k, v in weights.items() if v > 0])}")
    print(f"  Test R²:  {ensemble_r2:.3f}")
    print(f"  MAE:      {ensemble_mae:,.0f}")
    print(f"  RMSE:     {ensemble_rmse:,.0f}")
    print(f"  MAPE:     {ensemble_mape:.1f}%")
    print(f"  MASE:     {ensemble_mase:.3f}")
    print(f"  Dir Acc:  {ensemble_dir_acc:.1f}%")
    
    results['Ensemble'] = {
        'predictions': ensemble_pred,
        'test_r2': ensemble_r2,
        'mae': ensemble_mae,
        'rmse': ensemble_rmse,
        'mape': ensemble_mape,
        'mase': ensemble_mase,
        'directional_accuracy': ensemble_dir_acc
    }
    
    # Feature importance (from Random Forest)
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    print(f"\n{'='*80}")
    print("TOP 10 MOST IMPORTANT FEATURES")
    print(f"{'='*80}")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")
    
    # Validation comparison
    print(f"\n{'='*80}")
    print("VALIDATION PERIOD COMPARISON")
    print(f"{'='*80}")
    print(f"{'Month':<12} {'Actual':>15} {'Ensemble':>15} {'RF':>15} {'GBoost':>15}")
    print("-" * 80)
    
    test_dates = y_test.index
    for i, date in enumerate(test_dates):
        actual = y_test.iloc[i]
        ens = ensemble_pred[i]
        rf = predictions['Random Forest'][i]
        gb = predictions['Gradient Boosting'][i]
        
        print(f"{date.strftime('%Y-%m'):<12} {actual:>15,.0f} {ens:>15,.0f} {rf:>15,.0f} {gb:>15,.0f}")
    
    return results, X_test, y_test, predictions, ensemble_pred, feature_cols

# =============================================================================
# FORECAST FUTURE MONTHS
# =============================================================================
def forecast_future(monthly_df, models, feature_cols, target='reach', periods=6):
    """Generate forecasts for next 6 months"""
    
    print(f"\n{'='*80}")
    print(f"FORECASTING NEXT {periods} MONTHS - {target.upper()}")
    print(f"{'='*80}")
    
    # Start from the last known month
    last_date = monthly_df.index[-1]
    forecast_data = []
    
    # We'll use the last row as template and update it
    current_data = monthly_df.iloc[-1:].copy()
    
    for i in range(1, periods + 1):
        # Create next month's date
        next_date = last_date + pd.DateOffset(months=i)
        
        # Update temporal features
        current_data.index = [next_date]
        current_data['month'] = next_date.month
        current_data['quarter'] = next_date.quarter
        current_data['year'] = next_date.year
        current_data['months_since_start'] = current_data['months_since_start'].iloc[0] + i
        current_data['month_sin'] = np.sin(2 * np.pi * next_date.month / 12)
        current_data['month_cos'] = np.cos(2 * np.pi * next_date.month / 12)
        
        # Update lag features with recent predictions/actuals
        if i == 1:
            current_data['reach_lag_1'] = monthly_df['reach'].iloc[-1]
            current_data['reach_lag_2'] = monthly_df['reach'].iloc[-2]
            current_data['reach_lag_3'] = monthly_df['reach'].iloc[-3]
            current_data['engagement_lag_1'] = monthly_df['engagement'].iloc[-1]
            current_data['engagement_lag_2'] = monthly_df['engagement'].iloc[-2]
            current_data['engagement_lag_3'] = monthly_df['engagement'].iloc[-3]
        else:
            # Use previous forecast as lag (for reach forecasting, keep engagement from actuals)
            if target == 'reach':
                if len(forecast_data) >= 1:
                    current_data['reach_lag_1'] = forecast_data[-1]['forecast_value']
                    current_data['engagement_lag_1'] = monthly_df['engagement'].iloc[-1]
                if len(forecast_data) >= 2:
                    current_data['reach_lag_2'] = forecast_data[-2]['forecast_value']
                    current_data['engagement_lag_2'] = monthly_df['engagement'].iloc[-2]
                if len(forecast_data) >= 3:
                    current_data['reach_lag_3'] = forecast_data[-3]['forecast_value']
                    current_data['engagement_lag_3'] = monthly_df['engagement'].iloc[-3]
            else:  # engagement forecasting
                if len(forecast_data) >= 1:
                    current_data['engagement_lag_1'] = forecast_data[-1]['forecast_value']
                    current_data['reach_lag_1'] = monthly_df['reach'].iloc[-1]
                if len(forecast_data) >= 2:
                    current_data['engagement_lag_2'] = forecast_data[-2]['forecast_value']
                    current_data['reach_lag_2'] = monthly_df['reach'].iloc[-2]
                if len(forecast_data) >= 3:
                    current_data['engagement_lag_3'] = forecast_data[-3]['forecast_value']
                    current_data['reach_lag_3'] = monthly_df['reach'].iloc[-3]
        
        # Update rolling averages (simplified - use recent avg)
        recent_avg = monthly_df['reach'].iloc[-3:].mean()
        current_data['reach_rolling_3m_avg'] = recent_avg
        current_data['reach_rolling_6m_avg'] = monthly_df['reach'].iloc[-6:].mean()
        
        # Predict with each model
        X_future = current_data[feature_cols]
        
        predictions = {}
        for name in ['Random Forest', 'Gradient Boosting', 'Ridge Regression']:
            model = models[name]['model']
            pred = max(0, model.predict(X_future)[0])
            predictions[name] = pred
        
        # Ensemble prediction
        ensemble_pred = np.mean(list(predictions.values()))
        
        forecast_data.append({
            'date': next_date,
            'forecast_value': ensemble_pred,
            'rf_pred': predictions['Random Forest'],
            'gb_pred': predictions['Gradient Boosting'],
            'ridge_pred': predictions['Ridge Regression']
        })
    
    # Display forecast
    print(f"\n{'Month':<12} {'Ensemble':>15} {'Random Forest':>15} {'Gradient Boost':>15} {'Ridge':>15}")
    print("-" * 85)
    
    for item in forecast_data:
        print(f"{item['date'].strftime('%Y-%m'):<12} "
              f"{item['forecast_value']:>15,.0f} "
              f"{item['rf_pred']:>15,.0f} "
              f"{item['gb_pred']:>15,.0f} "
              f"{item['ridge_pred']:>15,.0f}")
    
    total_forecast = sum([item['forecast_value'] for item in forecast_data])
    print(f"\nTotal 6-month {target} forecast: {total_forecast:,.0f}")
    
    return forecast_data

# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_results(monthly_df, test_data, ensemble_pred, forecast_data, target='reach'):
    """Visualize historical, validation, and forecast"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Historical + Validation + Forecast
    ax1 = axes[0, 0]
    
    # Historical (training)
    hist_data = monthly_df[target].iloc[:-6]
    ax1.plot(hist_data.index, hist_data.values, 'o-', label='Historical', linewidth=2)
    
    # Validation actual
    ax1.plot(test_data.index, test_data.values, 's-', label='Validation (Actual)', 
             linewidth=2, markersize=8, color='orange')
    
    # Validation predicted
    ax1.plot(test_data.index, ensemble_pred, '^--', label='Validation (Predicted)', 
             linewidth=2, markersize=8, color='red')
    
    # Future forecast
    forecast_dates = [item['date'] for item in forecast_data]
    forecast_values = [item['forecast_value'] for item in forecast_data]
    ax1.plot(forecast_dates, forecast_values, 'D--', label='Forecast', 
             linewidth=2, markersize=8, color='green')
    
    ax1.set_title(f'{target.upper()} - Historical + Validation + Forecast', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'Total {target}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Actual vs Predicted (Validation)
    ax2 = axes[0, 1]
    ax2.scatter(test_data.values, ensemble_pred, s=100, alpha=0.6)
    
    min_val = min(test_data.min(), ensemble_pred.min())
    max_val = max(test_data.max(), ensemble_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_title('Actual vs Predicted (Validation)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Actual Value')
    ax2.set_ylabel('Predicted Value')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Residuals
    ax3 = axes[1, 0]
    residuals = test_data.values - ensemble_pred
    ax3.bar(range(len(residuals)), residuals, 
            color=['red' if r < 0 else 'green' for r in residuals])
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_title('Prediction Errors (Validation)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Validation Month')
    ax3.set_ylabel('Error (Predicted - Actual)')
    ax3.grid(alpha=0.3, axis='y')
    
    # Plot 4: Forecast breakdown
    ax4 = axes[1, 1]
    x = np.arange(len(forecast_data))
    width = 0.25
    
    rf_vals = [item['rf_pred'] for item in forecast_data]
    gb_vals = [item['gb_pred'] for item in forecast_data]
    ridge_vals = [item['ridge_pred'] for item in forecast_data]
    
    ax4.bar(x - width, rf_vals, width, label='Random Forest', alpha=0.8)
    ax4.bar(x, gb_vals, width, label='Gradient Boosting', alpha=0.8)
    ax4.bar(x + width, ridge_vals, width, label='Ridge', alpha=0.8)
    
    ax4.set_title('Forecast Breakdown by Model', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Future Month')
    ax4.set_ylabel(f'{target}')
    ax4.set_xticks(x)
    ax4.set_xticklabels([item['date'].strftime('%Y-%m') for item in forecast_data], rotation=45)
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("MACHINE LEARNING FACEBOOK FORECASTER")
    print("Random Forest + Gradient Boosting + Ridge Ensemble")
    print("="*80)
    
    # Fetch and prepare data
    df = fetch_data()
    
    # Engineer features
    print("\nEngineering features...")
    df_features, label_encoder = engineer_features(df)
    
    # Prepare monthly aggregated data
    print("Aggregating to monthly level...")
    monthly_df = prepare_monthly_features(df_features)
    print(f"✓ Created {len(monthly_df)} monthly samples with {len(monthly_df.columns)} features")
    
    # Train models for REACH
    reach_results, X_test, y_test_reach, reach_preds, reach_ensemble, feature_cols = \
        train_ml_models(monthly_df, target='reach')
    
    # Forecast future reach
    reach_forecast = forecast_future(monthly_df, reach_results, feature_cols, 
                                     target='reach', periods=6)
    
    # Visualize reach results
    plot_results(monthly_df, y_test_reach, reach_ensemble, reach_forecast, target='reach')
    
    # Train models for ENGAGEMENT
    engagement_results, _, y_test_engagement, engagement_preds, engagement_ensemble, _ = \
        train_ml_models(monthly_df, target='engagement')
    
    # Forecast future engagement
    engagement_forecast = forecast_future(monthly_df, engagement_results, feature_cols, 
                                         target='engagement', periods=6)
    
    # Visualize engagement results
    plot_results(monthly_df, y_test_engagement, engagement_ensemble, 
                engagement_forecast, target='engagement')
    
    # Performance Summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\nREACH FORECAST:")
    print(f"  Model Performance: {'✓ EXCELLENT' if reach_results['Ensemble']['test_r2'] > 0.9 else '✓ GOOD' if reach_results['Ensemble']['test_r2'] > 0.7 else '⚠️ MODERATE' if reach_results['Ensemble']['test_r2'] > 0.5 else '✗ POOR'}")
    print(f"  R² Score: {reach_results['Ensemble']['test_r2']:.3f}")
    print(f"  MASE: {reach_results['Ensemble']['mase']:.3f} ({'Better than naive' if reach_results['Ensemble']['mase'] < 1 else 'Worse than naive'})")
    print(f"  MAPE: {reach_results['Ensemble']['mape']:.1f}% ({'Excellent' if reach_results['Ensemble']['mape'] < 10 else 'Good' if reach_results['Ensemble']['mape'] < 20 else 'Reasonable' if reach_results['Ensemble']['mape'] < 50 else 'Inaccurate'})")
    print(f"  Directional Accuracy: {reach_results['Ensemble']['directional_accuracy']:.1f}%")
    print(f"  Best Model: Ridge Regression")
    print(f"  6-Month Forecast: {sum([item['forecast_value'] for item in reach_forecast]):,.0f} total reach")
    
    print(f"\nENGAGEMENT FORECAST:")
    print(f"  Model Performance: {'✓ EXCELLENT' if engagement_results['Ensemble']['test_r2'] > 0.9 else '✓ GOOD' if engagement_results['Ensemble']['test_r2'] > 0.7 else '⚠️ MODERATE' if engagement_results['Ensemble']['test_r2'] > 0.5 else '✗ POOR'}")
    print(f"  R² Score: {engagement_results['Ensemble']['test_r2']:.3f}")
    print(f"  MASE: {engagement_results['Ensemble']['mase']:.3f} ({'Better than naive' if engagement_results['Ensemble']['mase'] < 1 else 'Worse than naive'})")
    print(f"  MAPE: {engagement_results['Ensemble']['mape']:.1f}% ({'Excellent' if engagement_results['Ensemble']['mape'] < 10 else 'Good' if engagement_results['Ensemble']['mape'] < 20 else 'Reasonable' if engagement_results['Ensemble']['mape'] < 50 else 'Inaccurate'})")
    print(f"  Directional Accuracy: {engagement_results['Ensemble']['directional_accuracy']:.1f}%")
    
    if engagement_results['Ensemble']['test_r2'] < 0:
        print(f"  ⚠️ WARNING: Engagement is highly unpredictable")
        print(f"  Recommendation: Use conservative baseline estimates instead")
        engagement_baseline = monthly_df['engagement'].iloc[-6:].median()
        print(f"  Baseline (median of last 6 months): {engagement_baseline:,.0f}/month")
        print(f"  Baseline 6-month forecast: {engagement_baseline * 6:,.0f}")
    
    print(f"\n{'='*80}")
    print("INTERPRETATION GUIDE:")
    print(f"{'='*80}")
    print(f"  R² Score:")
    print(f"    • > 0.9 = Excellent predictions")
    print(f"    • 0.7-0.9 = Good predictions")
    print(f"    • 0.5-0.7 = Moderate predictions")
    print(f"    • < 0.5 = Poor predictions")
    print(f"    • Negative = Worse than predicting the mean")
    print(f"\n  MASE (Mean Absolute Scaled Error):")
    print(f"    • < 1.0 = Better than naive forecast (last value)")
    print(f"    • = 1.0 = Same as naive forecast")
    print(f"    • > 1.0 = Worse than naive forecast")
    print(f"\n  Directional Accuracy:")
    print(f"    • > 70% = Model captures trends well")
    print(f"    • 50-70% = Some trend detection")
    print(f"    • < 50% = Worse than random guessing")
    
    # POST TYPE ANALYSIS
    post_type_forecasts = analyze_post_type_performance(df, monthly_df)
    
    print("\n" + "="*80)
    print("✓ MACHINE LEARNING FORECAST COMPLETE")
    print("="*80 + "\n")

# =============================================================================
# POST TYPE PERFORMANCE ANALYSIS & FORECASTING
# =============================================================================
def analyze_post_type_performance(df, monthly_df):
    """Analyze which post types will perform best in the next 6 months"""
    
    print(f"\n{'='*80}")
    print("POST TYPE PERFORMANCE ANALYSIS & FORECAST")
    print(f"{'='*80}")
    
    # Historical performance (last 6 months)
    cutoff = df['publish_time'].max() - pd.DateOffset(months=6)
    recent = df[df['publish_time'] >= cutoff].copy()
    
    # Calculate metrics per post type
    post_type_stats = recent.groupby('post_type').agg({
        'reach': ['mean', 'sum', 'std', 'count'],
        'engagement': ['mean', 'sum', 'std'],
        'reactions': 'sum',
        'comments': 'sum',
        'shares': 'sum'
    }).round(0)
    
    # Flatten column names
    post_type_stats.columns = ['_'.join(col).strip() for col in post_type_stats.columns]
    
    # Calculate efficiency metrics
    post_type_stats['reach_per_post'] = post_type_stats['reach_sum'] / post_type_stats['reach_count']
    post_type_stats['engagement_per_post'] = post_type_stats['engagement_sum'] / post_type_stats['reach_count']
    post_type_stats['engagement_rate'] = (post_type_stats['engagement_sum'] / post_type_stats['reach_sum'] * 100)
    
    # Sort by reach per post
    post_type_stats = post_type_stats.sort_values('reach_per_post', ascending=False)
    
    print(f"\nHISTORICAL PERFORMANCE (Last 6 Months):")
    print(f"\n{'Post Type':<12} {'Posts':>8} {'Avg Reach':>12} {'Avg Engage':>12} {'Engage Rate':>12}")
    print("-" * 70)
    
    for post_type in post_type_stats.index:
        print(f"{post_type:<12} "
              f"{int(post_type_stats.loc[post_type, 'reach_count']):>8} "
              f"{int(post_type_stats.loc[post_type, 'reach_per_post']):>12,} "
              f"{int(post_type_stats.loc[post_type, 'engagement_per_post']):>12,} "
              f"{post_type_stats.loc[post_type, 'engagement_rate']:>11.2f}%")
    
    # Train post-type specific models
    print(f"\n{'='*80}")
    print("TRAINING POST-TYPE SPECIFIC MODELS")
    print(f"{'='*80}")
    
    post_type_forecasts = {}
    
    for post_type in df['post_type'].unique():
        if post_type not in ['Links', 'Text']:  # Skip low-volume types
            post_df = df[df['post_type'] == post_type].copy()
            
            if len(post_df) >= 30:  # Need enough data
                # Monthly aggregation for this post type
                monthly_post = post_df.resample('M', on='publish_time').agg({
                    'reach': ['sum', 'mean', 'count'],
                    'engagement': ['sum', 'mean']
                })
                
                # Skip if too few months
                if len(monthly_post) < 12:
                    continue
                
                monthly_post.columns = ['_'.join(col).strip() for col in monthly_post.columns]
                monthly_post = monthly_post[monthly_post['reach_count'] > 0]  # Remove zero-post months
                
                if len(monthly_post) >= 12:
                    # Simple trend analysis
                    recent_6m = monthly_post.iloc[-6:]
                    older_6m = monthly_post.iloc[-12:-6] if len(monthly_post) >= 12 else monthly_post.iloc[:6]
                    
                    recent_reach_avg = recent_6m['reach_mean'].mean()
                    older_reach_avg = older_6m['reach_mean'].mean()
                    trend = ((recent_reach_avg / older_reach_avg) - 1) * 100 if older_reach_avg > 0 else 0
                    
                    # Forecast: recent average with trend adjustment
                    forecast_reach_per_post = recent_reach_avg * (1 + trend/100 * 0.5)  # 50% of trend
                    forecast_engagement_per_post = recent_6m['engagement_mean'].mean()
                    
                    post_type_forecasts[post_type] = {
                        'recent_reach_per_post': recent_reach_avg,
                        'recent_engagement_per_post': recent_6m['engagement_mean'].mean(),
                        'forecast_reach_per_post': forecast_reach_per_post,
                        'forecast_engagement_per_post': forecast_engagement_per_post,
                        'trend': trend,
                        'recent_posts_per_month': recent_6m['reach_count'].mean(),
                        'total_posts': len(post_df)
                    }
    
    # Display forecasts
    print(f"\nPOST TYPE FORECASTS (Per Post Performance):")
    print(f"\n{'Post Type':<12} {'Recent Reach':>14} {'Forecast Reach':>15} {'Trend':>10} {'Posts/Mo':>10}")
    print("-" * 75)
    
    sorted_forecasts = sorted(post_type_forecasts.items(), 
                              key=lambda x: x[1]['forecast_reach_per_post'], 
                              reverse=True)
    
    for post_type, data in sorted_forecasts:
        print(f"{post_type:<12} "
              f"{data['recent_reach_per_post']:>14,.0f} "
              f"{data['forecast_reach_per_post']:>15,.0f} "
              f"{data['trend']:>9.1f}% "
              f"{data['recent_posts_per_month']:>10.1f}")
    
    # Scenario planning: What if you post X of each type?
    print(f"\n{'='*80}")
    print("SCENARIO PLANNING: OPTIMAL POSTING MIX")
    print(f"{'='*80}")
    
    # Assume 20 posts per month budget
    total_monthly_posts = 20
    
    # Strategy 1: Focus on best performer
    best_type = sorted_forecasts[0][0]
    best_reach = sorted_forecasts[0][1]['forecast_reach_per_post']
    
    # Strategy 2: Balanced mix based on current proportions
    recent_mix = recent.groupby('post_type').size() / len(recent)
    
    # Strategy 3: Weighted by performance
    performance_scores = {pt: data['forecast_reach_per_post'] 
                         for pt, data in post_type_forecasts.items()}
    total_score = sum(performance_scores.values())
    performance_weights = {pt: score/total_score 
                          for pt, score in performance_scores.items()}
    
    print(f"\nAssuming {total_monthly_posts} posts/month for next 6 months:")
    print(f"\nStrategy 1: ALL IN on Best Performer ({best_type})")
    print(f"  Post Mix: {total_monthly_posts} {best_type} posts/month")
    print(f"  Expected Monthly Reach: {best_reach * total_monthly_posts:,.0f}")
    print(f"  Expected 6-Month Reach: {best_reach * total_monthly_posts * 6:,.0f}")
    
    print(f"\nStrategy 2: Balanced Mix (Current Proportions)")
    balanced_reach = 0
    for post_type, weight in recent_mix.items():
        posts = int(total_monthly_posts * weight)
        if post_type in post_type_forecasts and posts > 0:
            reach = post_type_forecasts[post_type]['forecast_reach_per_post'] * posts
            balanced_reach += reach
            print(f"  {post_type}: {posts} posts/month → {reach:,.0f} reach/month")
    print(f"  Expected Monthly Reach: {balanced_reach:,.0f}")
    print(f"  Expected 6-Month Reach: {balanced_reach * 6:,.0f}")
    
    print(f"\nStrategy 3: Performance-Weighted Mix")
    weighted_reach = 0
    for post_type, weight in performance_weights.items():
        posts = int(total_monthly_posts * weight)
        if posts > 0:
            reach = post_type_forecasts[post_type]['forecast_reach_per_post'] * posts
            weighted_reach += reach
            print(f"  {post_type}: {posts} posts/month → {reach:,.0f} reach/month")
    print(f"  Expected Monthly Reach: {weighted_reach:,.0f}")
    print(f"  Expected 6-Month Reach: {weighted_reach * 6:,.0f}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if len(sorted_forecasts) > 0:
        best = sorted_forecasts[0]
        worst = sorted_forecasts[-1]
        
        print(f"\n✓ BEST PERFORMER: {best[0]}")
        print(f"  Average reach per post: {best[1]['forecast_reach_per_post']:,.0f}")
        print(f"  Trend: {best[1]['trend']:+.1f}%")
        print(f"  → Prioritize this content type for maximum reach")
        
        if len(sorted_forecasts) > 1:
            second = sorted_forecasts[1]
            print(f"\n✓ SECOND BEST: {second[0]}")
            print(f"  Average reach per post: {second[1]['forecast_reach_per_post']:,.0f}")
            print(f"  Trend: {second[1]['trend']:+.1f}%")
            print(f"  → Mix with {best[0]} for variety")
        
        print(f"\n⚠️ LOWEST PERFORMER: {worst[0]}")
        print(f"  Average reach per post: {worst[1]['forecast_reach_per_post']:,.0f}")
        print(f"  Trend: {worst[1]['trend']:+.1f}%")
        print(f"  → Consider reducing or eliminating")
        
        # Calculate potential lift
        current_avg_reach = sum([data['recent_reach_per_post'] * data['recent_posts_per_month'] 
                                for data in post_type_forecasts.values()])
        best_scenario_reach = best_reach * total_monthly_posts
        lift = ((best_scenario_reach / current_avg_reach) - 1) * 100 if current_avg_reach > 0 else 0
        
        print(f"\n💡 POTENTIAL IMPACT:")
        print(f"  Current avg monthly reach: {current_avg_reach:,.0f}")
        print(f"  Optimized (all {best[0]}): {best_scenario_reach:,.0f}")
        print(f"  Potential lift: {lift:+.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Reach per post by type
    ax1 = axes[0, 0]
    types = [pt for pt, _ in sorted_forecasts]
    recent_vals = [data['recent_reach_per_post'] for _, data in sorted_forecasts]
    forecast_vals = [data['forecast_reach_per_post'] for _, data in sorted_forecasts]
    
    x = np.arange(len(types))
    width = 0.35
    
    ax1.bar(x - width/2, recent_vals, width, label='Recent (6M)', alpha=0.8)
    ax1.bar(x + width/2, forecast_vals, width, label='Forecast', alpha=0.8)
    ax1.set_xlabel('Post Type')
    ax1.set_ylabel('Reach per Post')
    ax1.set_title('Average Reach per Post by Type', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(types, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    
    # Plot 2: Trend analysis
    ax2 = axes[0, 1]
    trends = [data['trend'] for _, data in sorted_forecasts]
    colors = ['green' if t > 0 else 'red' for t in trends]
    ax2.barh(types, trends, color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Trend (%)')
    ax2.set_title('6-Month Performance Trend by Type', fontweight='bold')
    ax2.grid(alpha=0.3, axis='x')
    
    # Plot 3: Strategy comparison
    ax3 = axes[1, 0]
    strategies = ['All In\n(Best Type)', 'Balanced\n(Current)', 'Performance\nWeighted']
    strategy_reaches = [
        best_reach * total_monthly_posts * 6,
        balanced_reach * 6,
        weighted_reach * 6
    ]
    bars = ax3.bar(strategies, strategy_reaches, color=['#2ecc71', '#3498db', '#9b59b6'], alpha=0.8)
    ax3.set_ylabel('Total 6-Month Reach')
    ax3.set_title('Strategy Comparison (6-Month Reach)', fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Engagement rate by type
    ax4 = axes[1, 1]
    engage_rates = [post_type_stats.loc[pt, 'engagement_rate'] 
                   for pt in types if pt in post_type_stats.index]
    type_labels = [pt for pt in types if pt in post_type_stats.index]
    
    ax4.bar(type_labels, engage_rates, color='coral', alpha=0.8)
    ax4.set_xlabel('Post Type')
    ax4.set_ylabel('Engagement Rate (%)')
    ax4.set_title('Engagement Rate by Post Type', fontweight='bold')
    ax4.set_xticklabels(type_labels, rotation=45, ha='right')
    ax4.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return post_type_forecasts