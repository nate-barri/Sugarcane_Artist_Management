"""
================================================================================
FACEBOOK REACH & ENGAGEMENT FORECASTER (IMPROVED VISUALIZATION)
================================================================================
Now with separate lines per post type for better insight
================================================================================
"""

import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# =============================================================================
# DATABASE CONNECTION
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
    SELECT publish_time, post_type, reach, shares, comments, reactions, impressions
    FROM facebook_data_set
    WHERE publish_time IS NOT NULL AND reach IS NOT NULL;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    df['engagement'] = df['reactions'] + df['comments'] + df['shares']
    df = df.fillna(0)
    print(f"✓ Loaded {len(df):,} posts from {df['publish_time'].min().date()} → {df['publish_time'].max().date()}")
    return df

# =============================================================================
# CLEAN + AGGREGATE MONTHLY
# =============================================================================
def prepare_monthly(df):
    df['year_month'] = df['publish_time'].dt.to_period('M')
    grouped = (
        df.groupby(['year_month', 'post_type'])
          .agg({'reach': 'mean', 'engagement': 'mean', 'impressions': 'mean',
                'reactions': 'mean', 'comments': 'mean', 'shares': 'mean'})
          .reset_index()
    )
    grouped['date'] = grouped['year_month'].dt.to_timestamp()
    grouped['month_num'] = (grouped['date'].dt.year - grouped['date'].dt.year.min()) * 12 + grouped['date'].dt.month
    grouped['post_type_encoded'] = LabelEncoder().fit_transform(grouped['post_type'])
    print(f"✓ Aggregated to {len(grouped)} monthly post-type records")
    return grouped

# =============================================================================
# ENSEMBLE MODEL
# =============================================================================
class EnsembleModel:
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1),
            'gb': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
            'ridge': Ridge(alpha=5)
        }
        self.scaler = StandardScaler()
        self.trained = False

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        for m in self.models.values():
            m.fit(Xs, y)
        self.trained = True

    def predict(self, X):
        Xs = self.scaler.transform(X)
        preds = np.mean([m.predict(Xs) for m in self.models.values()], axis=0)
        return np.maximum(preds, 0)

# =============================================================================
# EVALUATION
# =============================================================================
def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    floor = np.percentile(y_true, 10)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, floor))) * 100
    naive = np.mean(np.abs(np.diff(y_true))) if len(y_true) > 1 else 1
    mase = mae / max(naive, 1)
    bias = np.mean(y_pred - y_true)

    print(f"\n{'='*80}\nMODEL PERFORMANCE ({label})\n{'='*80}")
    print(f"MAE:  {mae:,.1f}")
    print(f"RMSE: {rmse:,.1f}")
    print(f"R²:   {r2:.3f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"MASE: {mase:.3f}")
    print(f"Bias: {'Overestimates' if bias>0 else 'Underestimates'} by {abs(bias):,.1f}")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'MASE': mase, 'Bias': bias}

# =============================================================================
# TRAIN + FORECAST (TIME-BASED SPLIT)
# =============================================================================
def train_and_forecast(df, months_ahead=6):
    features = ['month_num', 'post_type_encoded', 'impressions', 'reactions', 'comments', 'shares']
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    results = {}
    for target in ['reach', 'engagement']:
        model = EnsembleModel()
        model.fit(train[features], train[target])
        preds = model.predict(test[features])
        metrics = evaluate(test[target], preds, target.upper())
        results[target] = {'model': model, 'metrics': metrics}

        # Future forecast (per post type)
        last_date = df['date'].max()
        post_types = df['post_type'].unique()
        label_encoder = LabelEncoder().fit(df['post_type'])

        future_rows = []
        for i in range(1, months_ahead + 1):
            for pt in post_types:
                future_rows.append({
                    'month_num': df['month_num'].max() + i,
                    'post_type_encoded': label_encoder.transform([pt])[0],
                    'impressions': df[df['post_type'] == pt]['impressions'].mean(),
                    'reactions': df[df['post_type'] == pt]['reactions'].mean(),
                    'comments': df[df['post_type'] == pt]['comments'].mean(),
                    'shares': df[df['post_type'] == pt]['shares'].mean(),
                    'post_type': pt,
                    'date': last_date + pd.DateOffset(months=i)
                })

        future_df = pd.DataFrame(future_rows)
        preds_future = model.predict(future_df[features])
        future_df[f'forecast_{target}'] = preds_future
        results[target]['forecast'] = future_df

    return results

# =============================================================================
# IMPROVED VISUALIZATION - SEPARATE LINES PER POST TYPE
# =============================================================================
def plot_forecasts(results, df):
    # Color palette for post types
    colors = {
        'Photos': '#FF6B6B',
        'Videos': '#4ECDC4', 
        'Reels': '#95E1D3',
        'Links': '#F38181',
        'Text': '#AA96DA'
    }
    
    for target in ['reach', 'engagement']:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Get historical data by post type (last 12 months for clarity)
        cutoff_date = df['date'].max() - pd.DateOffset(months=12)
        hist_filtered = df[df['date'] >= cutoff_date]
        
        # Plot historical lines per post type
        for post_type in df['post_type'].unique():
            hist_subset = hist_filtered[hist_filtered['post_type'] == post_type]
            if len(hist_subset) > 0:
                color = colors.get(post_type, '#999999')
                ax.plot(hist_subset['date'], hist_subset[target], 
                       'o-', label=f'{post_type} (Historical)', 
                       color=color, linewidth=2, markersize=6, alpha=0.7)
        
        # Plot forecast lines per post type
        forecast_df = results[target]['forecast']
        for post_type in forecast_df['post_type'].unique():
            forecast_subset = forecast_df[forecast_df['post_type'] == post_type]
            color = colors.get(post_type, '#999999')
            ax.plot(forecast_subset['date'], forecast_subset[f'forecast_{target}'], 
                   's--', label=f'{post_type} (Forecast)', 
                   color=color, linewidth=2, markersize=6, alpha=0.9)
        
        ax.set_title(f"Facebook {target.capitalize()} Forecast by Post Type", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Date", fontsize=12, fontweight='bold')
        ax.set_ylabel(f"Average {target.capitalize()}", fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*80)
    print("FACEBOOK REACH & ENGAGEMENT FORECASTER (IMPROVED VISUALIZATION)")
    print("="*80)
    df = fetch_data()
    df = prepare_monthly(df)
    results = train_and_forecast(df)
    plot_forecasts(results, df)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for t in ['reach', 'engagement']:
        print(f"\n{t.upper()} METRICS:")
        for k, v in results[t]['metrics'].items():
            print(f"  {k}: {v:,.3f}")

        forecast = results[t]['forecast']
        print("\nPost-Type 6-Month Forecast Averages:")
        print(forecast.groupby('post_type')[f'forecast_{t}'].mean().round(1))

        total_forecast = forecast[f'forecast_{t}'].sum()
        avg_monthly = forecast.groupby('date')[f'forecast_{t}'].sum().mean()
        print(f"\n➡️ TOTAL PROJECTED {t.upper()} (6 Months): {total_forecast:,.0f}")
        print(f"   ≈ Average per Month: {avg_monthly:,.0f}")

    print("\nForecast completed successfully.\n")

if __name__ == "__main__":
    main()