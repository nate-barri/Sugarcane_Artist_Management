import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Database connection
db_params = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host': 'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port': '5432',
    'sslmode': 'require'
}

class ImprovedSocialMediaForecaster:
    """
    Improved forecaster addressing key issues from previous version
    """
    def __init__(self, db_params):
        self.db_params = db_params
        self.df = None
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        
    def fetch_data(self):
        print("Connecting to database...")
        conn = psycopg2.connect(**self.db_params)
        
        query = """
        SELECT publish_time, post_type, reach, shares, comments, reactions, 
               impressions, seconds_viewed, average_seconds_viewed
        FROM facebook_data_set
        WHERE publish_time IS NOT NULL
        ORDER BY publish_time
        """
        
        self.df = pd.read_sql_query(query, conn)
        conn.close()
        
        self.df['publish_time'] = pd.to_datetime(self.df['publish_time'])
        numeric_cols = ['reach', 'shares', 'comments', 'reactions', 'impressions', 
                       'seconds_viewed', 'average_seconds_viewed']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        print(f"✓ Loaded {len(self.df)} posts from {self.df['publish_time'].min().date()} to {self.df['publish_time'].max().date()}")
        return self.df
    
    def prepare_data(self):
        """Prepare weekly aggregated data with better outlier handling"""
        print("\n" + "="*80)
        print("DATA PREPARATION")
        print("="*80)
        
        df_week = self.df.set_index('publish_time').resample('W').agg({
            'reach': 'sum',
            'reactions': 'sum',
            'comments': 'sum',
            'shares': 'sum',
            'impressions': 'sum',
            'post_type': 'count'
        }).rename(columns={'post_type': 'post_count'})
        
        df_week = df_week[df_week['post_count'] > 0].copy()
        df_week['engagement'] = df_week['reactions'] + df_week['comments'] + df_week['shares']
        df_week['engagement_rate'] = np.where(
            df_week['reach'] > 0,
            (df_week['engagement'] / df_week['reach']) * 100,
            0
        )
        
        # More conservative outlier removal using IQR method
        for metric in ['reach', 'engagement', 'impressions']:
            Q1 = df_week[metric].quantile(0.25)
            Q3 = df_week[metric].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 3 * IQR  # More lenient than standard 1.5*IQR
            df_week = df_week[df_week[metric] <= upper_bound]
        
        print(f"✓ Using {len(df_week)} weeks")
        print(f"  Date range: {df_week.index.min().date()} to {df_week.index.max().date()}")
        print(f"\nWeekly Statistics:")
        print(df_week[['reach', 'engagement', 'post_count']].describe())
        
        return df_week
    
    def create_features(self, data, target_metric):
        """Create comprehensive feature set with proper handling"""
        df = data.copy().reset_index()
        df.rename(columns={'publish_time': 'date'}, inplace=True)
        
        # Time-based features
        df['week_number'] = range(len(df))
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Cyclical encoding for seasonality
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        # Multiple lag features
        for lag in [1, 2, 3, 4, 8, 12]:
            df[f'{target_metric}_lag{lag}'] = df[target_metric].shift(lag)
        
        # Rolling statistics with multiple windows
        for window in [4, 8, 12, 26]:
            min_periods = max(2, window // 3)
            
            df[f'{target_metric}_ma{window}'] = df[target_metric].shift(1).rolling(
                window=window, min_periods=min_periods
            ).mean()
            
            df[f'{target_metric}_std{window}'] = df[target_metric].shift(1).rolling(
                window=window, min_periods=min_periods
            ).std()
            
            df[f'{target_metric}_max{window}'] = df[target_metric].shift(1).rolling(
                window=window, min_periods=min_periods
            ).max()
            
            df[f'{target_metric}_min{window}'] = df[target_metric].shift(1).rolling(
                window=window, min_periods=min_periods
            ).min()
        
        # Momentum and trend features
        for window in [4, 8, 12]:
            df[f'{target_metric}_momentum{window}'] = df[target_metric].diff(window)
            df[f'{target_metric}_pct_change{window}'] = df[target_metric].pct_change(window)
        
        # Exponential weighted moving average
        df[f'{target_metric}_ewma'] = df[target_metric].shift(1).ewm(span=8, min_periods=2).mean()
        
        # Post count features
        df['post_count_ma4'] = df['post_count'].shift(1).rolling(window=4, min_periods=2).mean()
        df['post_count_ma8'] = df['post_count'].shift(1).rolling(window=8, min_periods=2).mean()
        
        # Cross-metric features (if not forecasting post_count)
        if target_metric != 'post_count':
            df['engagement_rate_ma4'] = df['engagement_rate'].shift(1).rolling(window=4, min_periods=2).mean()
            df['per_post_avg'] = df[target_metric] / df['post_count'].replace(0, 1)
            df['per_post_avg_ma4'] = df['per_post_avg'].shift(1).rolling(window=4, min_periods=2).mean()
        
        # Clean up
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics"""
        y_pred = np.maximum(y_pred, 0)
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE with handling for zero values
        mask = y_true > 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 999
        
        return {
            'R²': r2,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': min(mape, 999)
        }
    
    def train_model(self, X_train, y_train, X_test, y_test, metric_name):
        """Train multiple models and select best"""
        
        print(f"\n{'='*80}")
        print(f"TRAINING: {metric_name.upper()}")
        print(f"{'='*80}")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Models to test
        models = {
            'Ridge (α=1)': Ridge(alpha=1),
            'Ridge (α=10)': Ridge(alpha=10),
            'Ridge (α=50)': Ridge(alpha=50),
            'Lasso (α=1)': Lasso(alpha=1, max_iter=5000),
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                min_samples_split=10,
                random_state=42
            )
        }
        
        print(f"\n{'Model':<20} | {'CV R² Mean':>12} | {'CV R² Std':>11} | {'Test R²':>8} | {'Test MAE':>10}")
        print("-" * 85)
        
        best_model = None
        best_scaler = None
        best_score = -np.inf
        best_name = None
        best_metrics = None
        
        # Use more splits for better CV estimation
        tscv = TimeSeriesSplit(n_splits=min(5, len(X_train) // 20))
        
        for name, model in models.items():
            try:
                # Determine if model needs scaling
                needs_scaling = 'Ridge' in name or 'Lasso' in name
                X_tr = X_train_scaled if needs_scaling else X_train
                X_te = X_test_scaled if needs_scaling else X_test
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_tr, y_train, cv=tscv, scoring='r2')
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                
                # Train on full training set
                model.fit(X_tr, y_train)
                y_pred = np.maximum(model.predict(X_te), 0)
                
                # Calculate test metrics
                metrics = self.calculate_metrics(y_test, y_pred)
                
                print(f"{name:<20} | {cv_mean:>12.4f} | {cv_std:>11.4f} | {metrics['R²']:>8.4f} | {metrics['MAE']:>10.1f}")
                
                # Select best model based on test R²
                if metrics['R²'] > best_score:
                    best_score = metrics['R²']
                    best_model = model
                    best_scaler = scaler if needs_scaling else None
                    best_name = name
                    best_metrics = metrics
                    
            except Exception as e:
                print(f"{name:<20} | ERROR: {str(e)}")
        
        print("-" * 85)
        print(f"✓ Best Model: {best_name} (R² = {best_score:.4f})")
        
        return best_model, best_scaler, best_metrics, best_name
    
    def generate_forecast(self, periods=26):
        """Generate forecasts for all metrics"""
        print("\n" + "="*80)
        print("IMPROVED SOCIAL MEDIA FORECASTING")
        print("="*80)
        
        weekly_data = self.prepare_data()
        
        for target_metric in ['post_count', 'reach', 'engagement']:
            print(f"\n{'='*80}")
            print(f"Processing: {target_metric.upper()}")
            print(f"{'='*80}")
            
            df_features = self.create_features(weekly_data, target_metric)
            
            # Drop the first few rows where we have too many NaN from lags
            df_features = df_features.iloc[15:].reset_index(drop=True)
            
            feature_cols = [c for c in df_features.columns if c not in 
                           ['date', 'reach', 'engagement', 'post_count', 
                            'engagement_rate', 'reactions', 'comments', 'shares',
                            'impressions', 'per_post_avg']]
            
            print(f"Features: {len(feature_cols)}")
            print(f"Samples after cleaning: {len(df_features)}")
            
            # Use 80/20 split
            split_idx = int(len(df_features) * 0.80)
            
            X = df_features[feature_cols].values
            y = df_features[target_metric].values
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train
            model, scaler, metrics, model_name = self.train_model(
                X_train, y_train, X_test, y_test, target_metric
            )
            
            self.models[target_metric] = model
            self.scalers[target_metric] = scaler
            
            # Generate future predictions
            self.generate_predictions(
                df_features, model, scaler, target_metric, 
                feature_cols, periods, weekly_data
            )
            
            print(f"\nTest Set Performance:")
            print(f"  R²: {metrics['R²']:.4f}")
            print(f"  MAE: {metrics['MAE']:.1f}")
            print(f"  MAPE: {metrics['MAPE']:.1f}%")
    
    def generate_predictions(self, df_features, model, scaler, target_metric, 
                           feature_cols, periods, weekly_data):
        """Generate future predictions"""
        
        predictions = []
        historical = df_features[target_metric].values.tolist()
        last_date = df_features['date'].iloc[-1]
        
        # Keep track of all related metrics for cross-features
        recent_data = weekly_data.tail(30).copy()
        
        for i in range(periods):
            future_date = last_date + timedelta(weeks=i+1)
            all_values = historical + predictions
            
            features = {}
            
            # Time features
            features['week_number'] = df_features['week_number'].iloc[-1] + i + 1
            features['month'] = future_date.month
            features['quarter'] = (future_date.month - 1) // 3 + 1
            features['year'] = future_date.year
            features['week_of_year'] = future_date.isocalendar()[1]
            
            # Cyclical features
            features['month_sin'] = np.sin(2 * np.pi * future_date.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * future_date.month / 12)
            features['week_sin'] = np.sin(2 * np.pi * features['week_of_year'] / 52)
            features['week_cos'] = np.cos(2 * np.pi * features['week_of_year'] / 52)
            
            # Lag features
            for lag in [1, 2, 3, 4, 8, 12]:
                if len(all_values) >= lag:
                    features[f'{target_metric}_lag{lag}'] = all_values[-lag]
                else:
                    features[f'{target_metric}_lag{lag}'] = all_values[-1] if all_values else 0
            
            # Rolling features
            for window in [4, 8, 12, 26]:
                if len(all_values) >= window:
                    window_data = all_values[-window:]
                    features[f'{target_metric}_ma{window}'] = np.mean(window_data)
                    features[f'{target_metric}_std{window}'] = np.std(window_data)
                    features[f'{target_metric}_max{window}'] = np.max(window_data)
                    features[f'{target_metric}_min{window}'] = np.min(window_data)
                else:
                    features[f'{target_metric}_ma{window}'] = np.mean(all_values) if all_values else 0
                    features[f'{target_metric}_std{window}'] = 0
                    features[f'{target_metric}_max{window}'] = max(all_values) if all_values else 0
                    features[f'{target_metric}_min{window}'] = min(all_values) if all_values else 0
            
            # Momentum features
            for window in [4, 8, 12]:
                if len(all_values) > window:
                    features[f'{target_metric}_momentum{window}'] = all_values[-1] - all_values[-window-1]
                    features[f'{target_metric}_pct_change{window}'] = (all_values[-1] - all_values[-window-1]) / (all_values[-window-1] + 1e-6)
                else:
                    features[f'{target_metric}_momentum{window}'] = 0
                    features[f'{target_metric}_pct_change{window}'] = 0
            
            # EWMA
            if len(all_values) >= 2:
                features[f'{target_metric}_ewma'] = pd.Series(all_values).ewm(span=8).mean().iloc[-1]
            else:
                features[f'{target_metric}_ewma'] = all_values[-1] if all_values else 0
            
            # Post count features
            features['post_count_ma4'] = recent_data['post_count'].tail(4).mean()
            features['post_count_ma8'] = recent_data['post_count'].tail(8).mean()
            
            # Cross-metric features
            if target_metric != 'post_count':
                features['engagement_rate_ma4'] = recent_data['engagement_rate'].tail(4).mean()
                avg_per_post = recent_data[target_metric].sum() / max(recent_data['post_count'].sum(), 1)
                features['per_post_avg_ma4'] = avg_per_post
            
            # Create feature vector
            X_future = np.array([[features.get(col, 0) for col in feature_cols]])
            
            # Scale if needed
            if scaler is not None:
                X_future = scaler.transform(X_future)
            
            # Predict
            pred = max(0, model.predict(X_future)[0])
            predictions.append(pred)
        
        self.predictions[target_metric] = {
            'forecast': predictions,
            'historical': historical,
            'dates': [last_date + timedelta(weeks=i+1) for i in range(periods)]
        }
    
    def visualize_results(self):
        """Visualize forecast results"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Improved Social Media Forecast', fontsize=16, fontweight='bold')
        
        plot_info = [
            ('reach', axes[0, 0], 'Weekly Reach'),
            ('engagement', axes[0, 1], 'Weekly Engagement'),
            ('post_count', axes[1, 0], 'Weekly Post Count')
        ]
        
        for metric, ax, title in plot_info:
            if metric not in self.predictions:
                continue
            
            hist = np.array(self.predictions[metric]['historical'])
            forecast = np.array(self.predictions[metric]['forecast'])
            
            hist_x = range(len(hist))
            forecast_x = range(len(hist), len(hist) + len(forecast))
            
            # Plot
            ax.plot(hist_x, hist, label='Historical', color='#2c3e50', linewidth=2)
            ax.plot(forecast_x, forecast, label='Forecast', color='#e74c3c', 
                   linewidth=2, linestyle='--', marker='o', markersize=4)
            
            # Confidence intervals based on recent volatility
            recent_std = np.std(hist[-26:]) if len(hist) >= 26 else np.std(hist)
            
            upper_95 = forecast + 1.96 * recent_std
            lower_95 = np.maximum(forecast - 1.96 * recent_std, 0)
            
            ax.fill_between(forecast_x, lower_95, upper_95, alpha=0.2, 
                          color='#e74c3c', label='95% CI')
            
            ax.axvline(len(hist)-1, color='gray', linestyle=':', alpha=0.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Week')
            ax.set_ylabel(metric.replace('_', ' ').title())
        
        # Summary stats
        ax_summary = axes[1, 1]
        ax_summary.axis('off')
        
        summary_text = "FORECAST SUMMARY\n" + "="*50 + "\n\n"
        
        for metric in ['post_count', 'reach', 'engagement']:
            if metric not in self.predictions:
                continue
            
            fc = self.predictions[metric]['forecast']
            hist = self.predictions[metric]['historical']
            
            summary_text += f"{metric.upper().replace('_', ' ')}:\n"
            summary_text += f"  Historical Avg: {np.mean(hist):,.0f}/week\n"
            summary_text += f"  Forecast Avg: {np.mean(fc):,.0f}/week\n"
            summary_text += f"  Change: {((np.mean(fc)/np.mean(hist)-1)*100):+.1f}%\n\n"
        
        ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('improved_forecast.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'improved_forecast.png'")
        plt.show()

# Main execution
if __name__ == "__main__":
    print("\n" + "="*80)
    print("IMPROVED SOCIAL MEDIA FORECASTER")
    print("="*80 + "\n")
    
    forecaster = ImprovedSocialMediaForecaster(db_params)
    forecaster.fetch_data()
    forecaster.generate_forecast(periods=26)
    forecaster.visualize_results()
    
    print("\n" + "="*80)
    print("✓ FORECAST COMPLETE")
    print("="*80)