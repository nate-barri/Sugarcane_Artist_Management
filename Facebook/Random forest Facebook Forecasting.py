import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Database connection
DB_PARAMS = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host': 'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port': '5432',
    'sslmode': 'require'
}

def fetch_data():
    """Fetch data from PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        query = """
        SELECT 
            post_id, page_id, page_name, title, description, post_type,
            duration_sec, publish_time, year, month, day, time,
            permalink, is_crosspost, is_share, funded_content_status,
            reach, shares, comments, reactions, seconds_viewed,
            average_seconds_viewed, impressions
        FROM public.facebook_data_set
        ORDER BY publish_time
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"✓ Fetched {len(df)} posts from database")
        return df
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def prepare_monthly_data(df):
    """Prepare monthly aggregates with outlier handling"""
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    df['year_month'] = df['publish_time'].dt.to_period('M')
    
    for col in ['reactions', 'comments', 'shares', 'reach']:
        df[col] = df[col].fillna(0)
    
    df['total_engagement'] = df['reactions'] + df['comments'] + df['shares']
    
    monthly = df.groupby('year_month').agg({
        'post_id': 'count',
        'reach': ['sum', 'median'],
        'reactions': ['sum', 'median'],
        'comments': ['sum', 'median'],
        'shares': ['sum', 'median'],
        'total_engagement': ['sum', 'median']
    })
    
    monthly.columns = ['_'.join(col).strip() for col in monthly.columns.values]
    monthly = monthly.reset_index()
    
    monthly = monthly.rename(columns={
        'post_id_count': 'post_count',
        'reach_sum': 'total_reach',
        'reach_median': 'median_reach',
        'reactions_sum': 'total_reactions',
        'reactions_median': 'median_reactions',
        'comments_sum': 'total_comments',
        'comments_median': 'median_comments',
        'shares_sum': 'total_shares',
        'shares_median': 'median_shares',
        'total_engagement_sum': 'total_engagement',
        'total_engagement_median': 'median_engagement'
    })
    
    monthly['engagement_rate'] = (monthly['total_engagement'] / 
                                 monthly['total_reach'].replace(0, np.nan)).fillna(0)
    
    monthly['date'] = monthly['year_month'].dt.to_timestamp()
    monthly = monthly.sort_values('date').reset_index(drop=True)
    
    # Detect and cap outliers using IQR method
    for col in ['total_reach', 'total_engagement', 'total_reactions', 'total_comments', 'total_shares']:
        Q1 = monthly[col].quantile(0.25)
        Q3 = monthly[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 2 * IQR
        upper = Q3 + 2 * IQR
        
        monthly[f'{col}_capped'] = monthly[col].clip(lower=lower, upper=upper)
    
    return monthly

def create_robust_features(df, target_col):
    """Create features with robust transformations"""
    df = df.copy()
    
    # Use capped version if available
    if f'{target_col}_capped' in df.columns:
        working_col = f'{target_col}_capped'
    else:
        working_col = target_col
    
    # Time features
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['time_index'] = np.arange(len(df))
    
    # Cyclical
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lags with robust handling
    for lag in [1, 2, 3]:
        df[f'lag_{lag}'] = df[working_col].shift(lag)
    
    # Rolling statistics (median more robust than mean)
    df['rolling_median_3'] = df[working_col].rolling(window=3, min_periods=1).median()
    df['rolling_median_6'] = df[working_col].rolling(window=6, min_periods=1).median()
    
    # Trend indicator
    if len(df) > 6:
        df['trend'] = df[working_col].rolling(window=6, min_periods=3).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 2 else 0
        )
    
    df = df.fillna(method='bfill').fillna(0)
    
    return df, working_col

class RobustEnsemble:
    """Ensemble of robust ML models"""
    
    def __init__(self):
        self.models = {
            'huber': HuberRegressor(epsilon=1.5, max_iter=200),
            'ransac': RANSACRegressor(random_state=42, min_samples=0.7),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                min_samples_split=8, subsample=0.8, random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_split=6,
                min_samples_leaf=3, random_state=42
            )
        }
        self.scaler = RobustScaler()
        self.weights = {}
        
    def fit(self, X, y):
        """Train all models and calculate weights"""
        X_scaled = self.scaler.fit_transform(X)
        
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
                # Weight by inverse of training error
                pred = model.predict(X_scaled)
                error = mean_absolute_error(y, pred)
                self.weights[name] = 1 / (error + 1)  # +1 to avoid division by zero
            except Exception as e:
                print(f"  Warning: {name} failed - {e}")
                self.weights[name] = 0
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
    def predict(self, X):
        """Weighted ensemble prediction"""
        X_scaled = self.scaler.transform(X)
        predictions = []
        
        for name, model in self.models.items():
            if self.weights.get(name, 0) > 0:
                try:
                    pred = model.predict(X_scaled)
                    predictions.append(pred * self.weights[name])
                except:
                    pass
        
        if len(predictions) > 0:
            return np.sum(predictions, axis=0)
        else:
            return np.zeros(len(X))
    
    def get_prediction_intervals(self, X):
        """Get prediction intervals from model disagreement"""
        X_scaled = self.scaler.transform(X)
        all_preds = []
        
        for name, model in self.models.items():
            if self.weights.get(name, 0) > 0:
                try:
                    pred = model.predict(X_scaled)
                    all_preds.append(pred)
                except:
                    pass
        
        if len(all_preds) > 1:
            all_preds = np.array(all_preds)
            lower = np.percentile(all_preds, 25, axis=0)
            upper = np.percentile(all_preds, 75, axis=0)
            return lower, upper
        else:
            pred = all_preds[0] if len(all_preds) > 0 else np.zeros(len(X))
            return pred * 0.8, pred * 1.2

def calculate_metrics(y_true, y_pred, y_train):
    """Calculate comprehensive metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MASE
    naive_error = np.mean(np.abs(np.diff(y_train)))
    mase = mae / naive_error if naive_error > 0 else np.nan
    
    # MAPE
    mask = y_true != 0
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100 if mask.sum() > 0 else np.nan
    
    # Accuracy metrics
    errors = np.abs((y_pred - y_true) / y_true) * 100
    errors = errors[np.isfinite(errors)]
    within_30 = (errors <= 30).sum() / len(errors) * 100 if len(errors) > 0 else 0
    within_50 = (errors <= 50).sum() / len(errors) * 100 if len(errors) > 0 else 0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MASE': mase,
        'MAPE': mape,
        'Accuracy_30%': within_30,
        'Accuracy_50%': within_50
    }

def forecast_with_ensemble(monthly_data, target_col, test_size=6, forecast_periods=6):
    """Forecast using robust ensemble"""
    
    # Prepare features
    df, working_col = create_robust_features(monthly_data, target_col)
    
    if len(df) < 15:
        return None
    
    # Feature selection
    feature_cols = ['time_index', 'month', 'quarter', 'month_sin', 'month_cos',
                   'post_count', 'lag_1', 'lag_2', 'lag_3', 
                   'rolling_median_3', 'rolling_median_6', 'trend']
    
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols]
    y = df[working_col]
    
    # Train/test split
    train_size = len(X) - test_size
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Train ensemble
    print(f"  → Training robust ensemble (4 models)...")
    ensemble = RobustEnsemble()
    ensemble.fit(X_train, y_train)
    
    # Predictions
    test_pred = ensemble.predict(X_test)
    train_pred = ensemble.predict(X_train)
    
    # Metrics
    metrics = calculate_metrics(y_test.values, test_pred, y_train.values)
    
    print(f"  ✓ MAPE: {metrics['MAPE']:.1f}% | R²: {metrics['R²']:.3f} | "
          f"Accuracy ±30%: {metrics['Accuracy_30%']:.0f}% | ±50%: {metrics['Accuracy_50%']:.0f}%")
    
    # Model weights
    weights_str = ", ".join([f"{k}: {v:.2f}" for k, v in ensemble.weights.items() if v > 0.01])
    print(f"  → Weights: {weights_str}")
    
    # Future forecasts
    future_preds = []
    future_lower = []
    future_upper = []
    
    last_row = df.iloc[-1:].copy()
    
    for i in range(forecast_periods):
        future_row = last_row.copy()
        
        # Update time features
        future_date = last_row['date'].values[0] + pd.DateOffset(months=i+1)
        future_row['date'] = future_date
        future_row['time_index'] = last_row['time_index'].values[0] + i + 1
        future_row['month'] = pd.Timestamp(future_date).month
        future_row['quarter'] = pd.Timestamp(future_date).quarter
        future_row['month_sin'] = np.sin(2 * np.pi * future_row['month'] / 12)
        future_row['month_cos'] = np.cos(2 * np.pi * future_row['month'] / 12)
        
        # Update lags
        if i == 0:
            future_row['lag_1'] = y.iloc[-1]
            future_row['lag_2'] = y.iloc[-2]
            future_row['lag_3'] = y.iloc[-3]
        elif i == 1:
            future_row['lag_1'] = future_preds[0]
            future_row['lag_2'] = y.iloc[-1]
            future_row['lag_3'] = y.iloc[-2]
        elif i == 2:
            future_row['lag_1'] = future_preds[1]
            future_row['lag_2'] = future_preds[0]
            future_row['lag_3'] = y.iloc[-1]
        else:
            future_row['lag_1'] = future_preds[i-1]
            future_row['lag_2'] = future_preds[i-2]
            future_row['lag_3'] = future_preds[i-3]
        
        # Update rolling medians
        recent_values = list(y.iloc[-6:].values) + future_preds
        future_row['rolling_median_3'] = np.median(recent_values[-3:])
        future_row['rolling_median_6'] = np.median(recent_values[-6:])
        
        # Predict
        X_future = future_row[feature_cols]
        pred = ensemble.predict(X_future)[0]
        lower, upper = ensemble.get_prediction_intervals(X_future)
        
        future_preds.append(max(0, pred))
        future_lower.append(max(0, lower[0]))
        future_upper.append(upper[0])
    
    return {
        'target': target_col,
        'ensemble': ensemble,
        'metrics': metrics,
        'test_pred': test_pred,
        'test_actual': y_test.values,
        'test_dates': df['date'].iloc[train_size:].values,
        'train_pred': train_pred,
        'train_actual': y_train.values,
        'train_dates': df['date'].iloc[:train_size].values,
        'future_pred': np.array(future_preds),
        'future_lower': np.array(future_lower),
        'future_upper': np.array(future_upper)
    }

def plot_results(results_dict, monthly_data, future_dates):
    """Plot forecast results"""
    
    n = len(results_dict)
    fig, axes = plt.subplots(n, 1, figsize=(16, 5*n))
    if n == 1:
        axes = [axes]
    
    for idx, (target, res) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        # Historical
        ax.plot(res['train_dates'], res['train_actual'],
               label='Historical', linewidth=2.5, color='#2c3e50', marker='o', markersize=6)
        
        # Test actual
        ax.plot(res['test_dates'], res['test_actual'],
               label='Actual (Test)', linewidth=2.5, color='#27ae60', marker='o', markersize=7)
        
        # Test predictions
        ax.plot(res['test_dates'], res['test_pred'],
               label='Predicted (Test)', linewidth=2.5, color='#e74c3c', 
               marker='x', markersize=8, linestyle='--')
        
        # Future forecast
        ax.plot(future_dates, res['future_pred'],
               label='6-Month Forecast', linewidth=3, color='#9b59b6',
               marker='s', markersize=8)
        
        # Confidence band
        ax.fill_between(future_dates, res['future_lower'], res['future_upper'],
                       alpha=0.3, color='#9b59b6', label='Prediction Range')
        
        ax.set_title(f"{target.replace('_', ' ').title()} - Robust Ensemble Forecast\n"
                    f"MAPE: {res['metrics']['MAPE']:.1f}% | R²: {res['metrics']['R²']:.3f} | "
                    f"Accuracy ±30%: {res['metrics']['Accuracy_30%']:.0f}%",
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel(target.replace('_', ' ').title(), fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        ax.axvline(x=res['train_dates'][-1], color='black', linestyle='--', alpha=0.3, linewidth=2)
        ax.axvline(x=res['test_dates'][-1], color='red', linestyle='--', alpha=0.3, linewidth=2)
    
    plt.tight_layout()
    plt.show()

def print_metrics_table(results_dict):
    """Print comprehensive metrics table"""
    
    print("\n" + "="*120)
    print("MODEL PERFORMANCE METRICS - ALL TARGETS")
    print("="*120)
    
    metrics_data = []
    for target, res in results_dict.items():
        metrics_data.append({
            'Target': target.replace('_', ' ').title(),
            'MAE': res['metrics']['MAE'],
            'RMSE': res['metrics']['RMSE'],
            'R²': res['metrics']['R²'],
            'MASE': res['metrics']['MASE'],
            'MAPE': res['metrics']['MAPE'],
            'Acc_30%': res['metrics']['Accuracy_30%'],
            'Acc_50%': res['metrics']['Accuracy_50%']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Format display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    print(metrics_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    
    print("\n" + "="*120)
    print("\nINTERPRETATION:")
    print("  MAE   = Mean Absolute Error (lower is better)")
    print("  RMSE  = Root Mean Squared Error (lower is better)")
    print("  R²    = Coefficient of Determination (closer to 1 is better, negative means worse than baseline)")
    print("  MASE  = Mean Absolute Scaled Error (<1 is better than naive forecast)")
    print("  MAPE  = Mean Absolute Percentage Error (<50% is good, <100% is acceptable)")
    print("  Acc   = Percentage of predictions within X% of actual values")
    print("\n" + "="*120)

def print_summary(results_dict, future_dates, avg_posts):
    """Print forecast summary"""
    
    print("\n" + "="*100)
    print(f"ROBUST ML ENSEMBLE FORECAST - NEXT 6 MONTHS (~{avg_posts:.0f} Posts/Month)")
    print("="*100)
    
    for target, res in results_dict.items():
        print(f"\n{target.upper().replace('_', ' ')}:")
        print("-" * 100)
        
        forecast_df = pd.DataFrame({
            'Month': future_dates,
            'Forecast': res['future_pred'],
            'Lower_Bound': res['future_lower'],
            'Upper_Bound': res['future_upper']
        })
        
        print(forecast_df.to_string(index=False, float_format=lambda x: f'{x:,.0f}'))
        
        print(f"\n6-MONTH SUMMARY:")
        print(f"  Total Forecast:      {forecast_df['Forecast'].sum():>15,.0f}")
        print(f"  Range:               {forecast_df['Lower_Bound'].sum():>15,.0f} to {forecast_df['Upper_Bound'].sum():>15,.0f}")
        
        if 'total' in target:
            avg = forecast_df['Forecast'].sum() / (6 * avg_posts)
            print(f"  Per Post:            {avg:>15,.0f}")
        
        print(f"\nRELIABILITY:")
        print(f"  MAE:                 {res['metrics']['MAE']:>15,.2f}")
        print(f"  RMSE:                {res['metrics']['RMSE']:>15,.2f}")
        print(f"  R²:                  {res['metrics']['R²']:>15.3f}")
        print(f"  MASE:                {res['metrics']['MASE']:>15.3f}")
        print(f"  MAPE:                {res['metrics']['MAPE']:>15.1f}%")
        print(f"  Accuracy (±30%):     {res['metrics']['Accuracy_30%']:>15.0f}%")
        print(f"  Accuracy (±50%):     {res['metrics']['Accuracy_50%']:>15.0f}%")
        
        if res['metrics']['MAPE'] < 100:
            print(f"  ✓ Reliable for planning")
        else:
            print(f"  ⚠ High uncertainty - use with caution")

def main():
    """Main execution"""
    
    print("="*100)
    print("ROBUST ML ENSEMBLE FORECASTER - FACEBOOK PERFORMANCE")
    print("Using: Huber Regression, RANSAC, Gradient Boosting, Random Forest")
    print("="*100)
    
    print("\n[1/4] Fetching data...")
    df = fetch_data()
    if df is None:
        return
    
    print("\n[2/4] Preparing monthly data with outlier handling...")
    monthly = prepare_monthly_data(df)
    print(f"  ✓ {len(monthly)} months | {monthly['date'].min().date()} to {monthly['date'].max().date()}")
    print(f"  ✓ Avg posts/month: {monthly['post_count'].mean():.1f}")
    
    print("\n[3/4] Training ensemble models...")
    print("="*100)
    
    targets = ['total_reach', 'total_reactions', 'total_engagement', 
               'total_comments', 'total_shares']
    
    results_dict = {}
    
    for target in targets:
        print(f"\n{target.upper().replace('_', ' ')}:")
        res = forecast_with_ensemble(monthly, target, test_size=6, forecast_periods=6)
        if res:
            results_dict[target] = res
    
    if not results_dict:
        print("\n✗ No successful forecasts")
        return
    
    future_dates = pd.date_range(
        start=monthly['date'].max() + pd.DateOffset(months=1),
        periods=6, freq='MS'
    )
    
    print("\n[4/4] Results:")
    
    # Print comprehensive metrics table
    print_metrics_table(results_dict)
    
    # Print forecast summary
    print_summary(results_dict, future_dates, monthly['post_count'].mean())
    
    print("\nGenerating visualizations...")
    plot_results(results_dict, monthly, future_dates)
    
    print("\n" + "="*100)
    print("✓ FORECASTING COMPLETE")
    print("="*100)

if __name__ == "__main__":
    main()