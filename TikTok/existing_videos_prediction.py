# -*- coding: utf-8 -*-
"""
existing_videos_total_6mo_with_graph.py
Aggregated 6-month forecast for all existing TikTok videos
+ Visualization of historical vs predicted total views.
"""

import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to use XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except:
    from sklearn.ensemble import GradientBoostingRegressor
    XGBOOST_AVAILABLE = False

# ================= DB CONNECTION =================
db_params = {
    'dbname':   'neondb',
    'user':     'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host':     'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port':     '5432',
    'sslmode':  'require'
}

# --------------------- HELPERS ---------------------
def fetch_all_videos(conn):
    q = """
    SELECT video_id, title, views, likes, shares, comments_added, saves,
           duration_sec, publish_time,
           CASE WHEN views>0 THEN ((COALESCE(likes,0)+COALESCE(shares,0)+
                COALESCE(comments_added,0)+COALESCE(saves,0))::FLOAT/views)*100 ELSE 0 END as engagement_rate,
           CASE WHEN views>0 THEN (COALESCE(saves,0)::FLOAT/views)*100 ELSE 0 END as save_rate,
           CASE WHEN views>0 THEN (COALESCE(shares,0)::FLOAT/views)*100 ELSE 0 END as share_rate
    FROM public.tt_video_etl
    WHERE views IS NOT NULL AND views > 0
      AND publish_time IS NOT NULL
    ORDER BY publish_time;
    """
    return pd.read_sql_query(q, conn)

def calculate_mase(y_true, y_pred, y_train):
    n = len(y_train)
    if n <= 1:
        return np.inf
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    if d == 0:
        return np.inf
    return np.mean(np.abs(y_true - y_pred)) / d

# --------------------- DATA PREP ---------------------
def prepare_training_data(df):
    df = df.copy()
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    now = pd.Timestamp.now(tz='UTC')
    df['days_old'] = (now - df['publish_time']).dt.days
    train_df = df[df['days_old'] >= 180].copy()
    train_df['estimated_daily_views'] = train_df['views'] / train_df['days_old']
    train_df['views_at_90d'] = train_df['estimated_daily_views'] * 90
    train_df['views_90_to_270'] = train_df['estimated_daily_views'] * 180
    decay = np.exp(-train_df['days_old'] / 365)
    train_df['views_90_to_270'] *= (0.5 + 0.5 * decay)
    train_df['log_views_90d'] = np.log1p(train_df['views_at_90d'])
    train_df['log_duration'] = np.log1p(train_df['duration_sec'])
    train_df['quality_score'] = (train_df['save_rate'] * 5 +
                                 train_df['share_rate'] * 3 +
                                 train_df['engagement_rate'])
    bins = [0, 1e4, 1e5, 1e6, float('inf')]
    train_df['view_category'] = pd.cut(train_df['views'], bins=bins, labels=[0,1,2,3]).astype(int)
    return train_df

# --------------------- MODEL ---------------------
def train_temporal_model(train_df, features=None):
    if features is None:
        features = [
            'log_views_90d', 'log_duration', 'duration_sec',
            'engagement_rate', 'save_rate', 'share_rate',
            'quality_score', 'view_category', 'days_old'
        ]
    train_df = train_df.sort_values('publish_time').reset_index(drop=True)
    X = train_df[features].fillna(0)
    y = train_df['views_90_to_270'].replace([np.inf, -np.inf], 0)
    q_low, q_high = y.quantile(0.01), y.quantile(0.99)
    mask = (y >= q_low) & (y <= q_high)
    X, y = X[mask], y[mask]
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    if XGBOOST_AVAILABLE:
        model = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, reg_alpha=1.0, reg_lambda=1.0, random_state=42
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mase = calculate_mase(y_test.values, y_pred, y_train.values)
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
    return model, scaler, features, {'mae': mae, 'rmse': rmse, 'r2': r2, 'mase': mase, 'mape': mape}

# --------------------- FORECAST ---------------------
def forecast_total_6mo(df, model, scaler, features, apply_age_decay=True):
    df = df.copy()
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    now = pd.Timestamp.now(tz='UTC')
    df['days_old'] = (now - df['publish_time']).dt.days
    df['estimated_daily_views'] = df['views'] / df['days_old'].replace(0,1)
    df['views_at_90d'] = df['estimated_daily_views'] * 90
    df['log_views_90d'] = np.log1p(df['views_at_90d'])
    df['log_duration'] = np.log1p(df['duration_sec'])
    df['quality_score'] = (df['save_rate'] * 5 + df['share_rate'] * 3 + df['engagement_rate'])
    bins = [0, 1e4, 1e5, 1e6, float('inf')]
    df['view_category'] = pd.cut(df['views'], bins=bins, labels=[0,1,2,3]).astype(int)
    X_all = df[features].fillna(0)
    X_all_s = scaler.transform(X_all)
    preds = model.predict(X_all_s)
    if apply_age_decay:
        decay = np.exp(-df['days_old'] / 365)
        preds *= (0.5 + 0.5 * decay)
    preds = np.maximum(preds, 0)
    total_pred = preds.sum()
    return total_pred, preds, df

# --------------------- VISUALIZATION ---------------------
def plot_historical_and_forecast(df, total_pred, metrics):
    df = df.copy()
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    # Historical: cumulative total views over time
    hist = df.groupby(df['publish_time'].dt.to_period('M'))['views'].sum().to_timestamp()
    hist_cum = hist.cumsum()
    # Extend 6 months into future
    last_date = hist_cum.index[-1]
    future_dates = pd.date_range(last_date + pd.offsets.MonthEnd(1), periods=6, freq='M')
    forecast_vals = np.linspace(hist_cum.iloc[-1], hist_cum.iloc[-1] + total_pred, num=6)
    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(hist_cum.index, hist_cum.values, label='Historical Total Views', color='blue', linewidth=2)
    plt.plot(future_dates, forecast_vals, '--', color='orange', linewidth=2, label='Predicted (6 mo)')
    plt.fill_between(future_dates,
                     forecast_vals*(1 - metrics['mape']/100),
                     forecast_vals*(1 + metrics['mape']/100),
                     color='orange', alpha=0.2, label='±MAPE confidence range')
    plt.title("Total Channel Views: Historical + 6-Month Forecast", fontsize=14, fontweight='bold')
    plt.ylabel("Cumulative Views")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --------------------- MAIN ---------------------
def main():
    print("="*70)
    print("TOTAL 6-MONTH EXISTING-VIDEO FORECAST WITH GRAPH")
    print("="*70)
    conn = psycopg2.connect(**db_params)
    df = fetch_all_videos(conn)
    conn.close()
    print(f"✓ Loaded {len(df)} videos")
    train_df = prepare_training_data(df)
    model, scaler, features, metrics = train_temporal_model(train_df)
    print(f"Model: R²={metrics['r2']:.3f}, MASE={metrics['mase']:.3f}, MAPE={metrics['mape']:.1f}%")
    total_pred, preds, df = forecast_total_6mo(df, model, scaler, features)
    print(f"\nPredicted 6-month total additional views: {int(total_pred):,}")
    print(f"Weekly average: {int(total_pred/26):,} views/week")
    # Visualize
    plot_historical_and_forecast(df, total_pred, metrics)
    print("\n✅ Forecast visualization complete.")

if __name__ == "__main__":
    main()
