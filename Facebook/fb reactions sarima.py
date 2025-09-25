#!/usr/bin/env python3
"""
Sugarcane Strategic Social Media Forecasting System
Aggregate-level time series forecasting for long-term planning (6-12 months)

This complements the existing tactical post-level system with strategic insights:
- Monthly/quarterly aggregate forecasts by content type
- ARIMA/SARIMAX models on smooth time series
- Validation via sMAPE (robust near zero)
- Use cases: Budget planning, resource allocation, growth targets

MODIFIED VERSION - No pmdarima dependency required
FIXED VERSION - Resolves pandas frequency issues, clips negatives, guards charts
"""

import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns  # ok to keep; not required for plots here
from datetime import datetime, timedelta
import warnings
import itertools
warnings.filterwarnings('ignore')

# Time series and forecasting
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------
# Configuration
# -----------------------
FORECAST_PERIODS = 12           # months ahead to forecast
MIN_HISTORY_MONTHS = 18         # minimum months needed to build model
CONFIDENCE_LEVEL = 0.95         # for prediction intervals
SEASONALITY_PERIOD = 12         # monthly seasonality
EPS = 1e-6                      # small constant to avoid divide-by-zero

# -----------------------
# Helpers
# -----------------------
def smape(y_true, y_pred):
    """Symmetric MAPE (%) — robust when values are near zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom < EPS, EPS, denom)
    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom)

def clip_nonnegative(arr):
    """Clip forecasts to be >= 0 (engagement cannot be negative)."""
    return np.clip(np.asarray(arr, dtype=float), a_min=0.0, a_max=None)

def safe_growth(next_value, last_value):
    """
    Growth = (next / last - 1)*100 with safe denominator.
    If last_value <= 0, return np.nan.
    """
    if last_value is None or last_value <= 0 or not np.isfinite(last_value):
        return np.nan
    return (float(next_value) / float(last_value) - 1.0) * 100.0

def is_finite_pos(x):
    return (x is not None) and np.isfinite(x)

# -----------------------
# Manual ARIMA Selection (replaces pmdarima)
# -----------------------
def manual_arima_selection(ts_data, seasonal=True, max_p=3, max_q=3, max_d=2, verbose=False):
    """
    Manual ARIMA model selection as alternative to pmdarima.auto_arima
    """
    best_aic = float('inf')
    best_model = None
    best_order = None
    best_seasonal_order = None
    models_tried = 0

    # Parameter ranges
    p_range = range(0, max_p + 1)
    d_range = range(0, max_d + 1)
    q_range = range(0, max_q + 1)

    if seasonal and len(ts_data) >= 24:
        P_range = range(0, 2)
        D_range = range(0, 2)
        Q_range = range(0, 2)
        s = 12  # monthly seasonality

        # Try seasonal first
        for params in itertools.product(p_range, d_range, q_range):
            for seasonal_params in itertools.product(P_range, D_range, Q_range):
                if seasonal_params == (0, 0, 0):
                    continue  # skip non-seasonal in this branch
                try:
                    model = SARIMAX(ts_data, order=params, seasonal_order=seasonal_params + (s,))
                    fitted_model = model.fit(disp=False)
                    models_tried += 1
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_model = fitted_model
                        best_order = params
                        best_seasonal_order = seasonal_params + (s,)
                except Exception as e:
                    if verbose:
                        print(f"    Failed SARIMA{params}x{seasonal_params + (s,)}: {e}")
                    continue
        if best_model is None:
            seasonal = False

    if not seasonal or best_model is None:
        for params in itertools.product(p_range, d_range, q_range):
            try:
                model = ARIMA(ts_data, order=params)
                fitted_model = model.fit()
                models_tried += 1
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_model = fitted_model
                    best_order = params
                    best_seasonal_order = None
            except Exception as e:
                if verbose:
                    print(f"    Failed ARIMA{params}: {e}")
                continue

    if verbose:
        print(f"    Tried {models_tried} models, best AIC: {best_aic:.2f}")

    # Wrapper to mimic predict API with confints
    class ARIMAWrapper:
        def __init__(self, model, order, seasonal_order=None):
            self.model = model
            self.order = order
            self.seasonal_order = seasonal_order

        def aic(self):
            return self.model.aic

        def predict(self, n_periods, return_conf_int=False, alpha=0.05):
            if return_conf_int:
                forecast = self.model.forecast(steps=n_periods)
                conf_int = self.model.get_forecast(steps=n_periods).conf_int(alpha=alpha)
                return forecast, conf_int.values
            else:
                return self.model.forecast(steps=n_periods)

    if best_model is None:
        raise ValueError("Could not fit any ARIMA model to the data")

    return ARIMAWrapper(best_model, best_order, best_seasonal_order)

# -----------------------
# Data Access
# -----------------------
def get_facebook_data():
    """Load data from Sugarcane database"""
    db_params = {
        'dbname':'neondb',
        'user':'neondb_owner',
        'password':'npg_dGzvq4CJPRx7',
        'host':'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
        'port':'5432',
        'sslmode':'require'
    }
    print("Connecting to Sugarcane social media database...")
    try:
        conn = psycopg2.connect(**db_params)
        query = """
        SELECT post_id, page_id, page_name, title, description, post_type,
               duration_sec, publish_time, year, month, day, time,
               permalink, is_crosspost, is_share, funded_content_status,
               reach, shares, comments, reactions, seconds_viewed, average_seconds_viewed
        FROM public.facebook_data_set
        WHERE reach >= 0 AND reactions >= 0 AND comments >= 0 AND shares >= 0
        ORDER BY publish_time;
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# -----------------------
# Strategic Analytics Class
# -----------------------
class SugarcaneStrategicForecasting:
    def __init__(self):
        self.data = None
        self.monthly_data = None
        self.models = {}
        self.forecasts = {}
        self.model_diagnostics = {}

    def load_and_aggregate_data(self):
        """Load raw data and create monthly aggregates"""
        print("\n=== LOADING DATA FOR STRATEGIC FORECASTING ===")
        df = get_facebook_data()
        if df is None or df.empty:
            print("❌ No data available")
            return False

        # Derived metrics
        df['publish_time'] = pd.to_datetime(df['publish_time'])
        df['total_engagement'] = df['reactions'].fillna(0) + df['comments'].fillna(0) + df['shares'].fillna(0)
        df['engagement_rate'] = np.where(df['reach'] > 0, df['total_engagement'] / df['reach'] * 100, 0)
        df['post_type_clean'] = df['post_type'].fillna('Unknown')

        # Monthly aggregates
        df['month_year'] = df['publish_time'].dt.to_period('M')

        monthly_total = df.groupby('month_year').agg({
            'total_engagement': 'sum',
            'reach': 'sum',
            'shares': 'sum',
            'comments': 'sum',
            'reactions': 'sum',
            'post_id': 'count'
        }).reset_index()
        monthly_total['content_type'] = 'Overall'

        content_types = ['video', 'photo', 'reel', 'text', 'link']
        monthly_by_type = []
        for ctype in content_types:
            mask = df['post_type_clean'].str.lower().str.contains(ctype, na=False)
            if mask.sum() >= 12:  # at least 12 posts of this type
                type_monthly = df[mask].groupby('month_year').agg({
                    'total_engagement': 'sum',
                    'reach': 'sum',
                    'shares': 'sum',
                    'comments': 'sum',
                    'reactions': 'sum',
                    'post_id': 'count'
                }).reset_index()
                type_monthly['content_type'] = ctype.title() + 's'
                monthly_by_type.append(type_monthly)

        all_monthly = [monthly_total] + monthly_by_type
        self.monthly_data = pd.concat(all_monthly, ignore_index=True)

        # Period -> datetime
        self.monthly_data['date'] = self.monthly_data['month_year'].dt.to_timestamp()
        self.monthly_data = self.monthly_data.sort_values(['content_type', 'date']).reset_index(drop=True)

        self.data = df
        print(f"✅ Loaded {len(df):,} posts aggregated into {len(self.monthly_data)} monthly data points")
        print(f"📊 Content types: {', '.join(self.monthly_data['content_type'].unique())}")
        print(f"📅 Date range: {self.monthly_data['date'].min().strftime('%b %Y')} to {self.monthly_data['date'].max().strftime('%b %Y')}")
        return True

    def analyze_time_series_properties(self):
        """Analyze stationarity, seasonality, and autocorrelation"""
        print("\n=== TIME SERIES ANALYSIS ===")
        analysis_results = {}
        content_types = self.monthly_data['content_type'].unique()

        for ctype in content_types:
            type_data = self.monthly_data[self.monthly_data['content_type'] == ctype].copy()
            if len(type_data) < MIN_HISTORY_MONTHS:
                continue

            ts = type_data.set_index('date')['total_engagement'].sort_index()
            # Fill missing months
            full_range = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq='MS')
            ts = ts.reindex(full_range).interpolate(method='linear')

            adf_result = adfuller(ts.dropna())
            is_stationary = adf_result[1] <= 0.05

            seasonality_strength = 0
            trend_strength = 0
            if len(ts) >= 24:
                try:
                    decomp = seasonal_decompose(ts, model='additive', period=12)
                    seasonality_strength = np.var(decomp.seasonal) / np.var(ts)
                    trend_strength = np.var(decomp.trend.dropna()) / np.var(ts)
                except Exception as e:
                    print(f"    Decomposition failed for {ctype}: {e}")

            analysis_results[ctype] = {
                'n_observations': len(ts),
                'mean_engagement': ts.mean(),
                'cv': ts.std() / ts.mean() if ts.mean() > 0 else 0,
                'is_stationary': is_stationary,
                'adf_pvalue': adf_result[1],
                'seasonality_strength': seasonality_strength,
                'trend_strength': trend_strength
            }

            print(f"\n{ctype}:")
            print(f"  Observations: {len(ts)}")
            print(f"  Mean monthly engagement: {ts.mean():,.0f}")
            print(f"  Coefficient of variation: {analysis_results[ctype]['cv']:.2f}")
            print(f"  Stationary: {'Yes' if is_stationary else 'No'} (p={adf_result[1]:.3f})")
            print(f"  Seasonality strength: {seasonality_strength:.2f}")
            print(f"  Trend strength: {trend_strength:.2f}")

        self.ts_analysis = analysis_results
        return analysis_results

    def build_arima_models(self):
        """Build ARIMA/SARIMAX models for each content type"""
        print("\n=== BUILDING FORECASTING MODELS (Manual ARIMA Selection) ===")
        content_types = self.monthly_data['content_type'].unique()

        for ctype in content_types:
            type_data = self.monthly_data[self.monthly_data['content_type'] == ctype].copy()
            if len(type_data) < MIN_HISTORY_MONTHS:
                print(f"⚠️  Skipping {ctype}: only {len(type_data)} months (need ≥{MIN_HISTORY_MONTHS})")
                continue

            print(f"\n📈 Building model for {ctype}...")
            ts_data = type_data.set_index('date')['total_engagement'].sort_index()
            full_range = pd.date_range(start=ts_data.index.min(), end=ts_data.index.max(), freq='MS')
            ts_data = ts_data.reindex(full_range).interpolate(method='linear')

            try:
                if len(ts_data) >= 24:
                    print(f"  Trying seasonal models (need 24+ obs, have {len(ts_data)})...")
                    auto_model = manual_arima_selection(ts_data, seasonal=True, verbose=False)
                else:
                    print(f"  Trying non-seasonal models (have {len(ts_data)} obs)...")
                    auto_model = manual_arima_selection(ts_data, seasonal=False, verbose=False)

                self.models[ctype] = {
                    'model': auto_model,
                    'order': auto_model.order,
                    'seasonal_order': auto_model.seasonal_order,
                    'aic': auto_model.aic(),
                    'data_points': len(ts_data),
                    'last_value': float(ts_data.iloc[-1]),
                    'ts_data': ts_data
                }

                order_str = str(auto_model.order)
                if auto_model.seasonal_order:
                    order_str += f" x {auto_model.seasonal_order}"
                print(f"  ✅ {ctype}: {order_str}, AIC={auto_model.aic():.1f}")

            except Exception as e:
                print(f"  ❌ Failed to build model for {ctype}: {e}")
                continue

        print(f"\n✅ Built {len(self.models)} forecasting models")
        return len(self.models) > 0

    def validate_models(self):
        """Validate models using walk-forward validation (sMAPE + nonnegative clip)."""
        print("\n=== MODEL VALIDATION ===")
        validation_results = {}

        for ctype, model_info in self.models.items():
            ts_data = model_info['ts_data']
            if len(ts_data) < MIN_HISTORY_MONTHS + 6:
                print(f"  ⚠️  Skipping validation for {ctype}: insufficient data")
                continue

            n_train = len(ts_data) - 6
            train_data = ts_data.iloc[:n_train]
            test_data = ts_data.iloc[n_train:]

            try:
                if model_info['seasonal_order']:
                    val_model = SARIMAX(
                        train_data,
                        order=model_info['order'],
                        seasonal_order=model_info['seasonal_order']
                    ).fit(disp=False)
                else:
                    val_model = ARIMA(train_data, order=model_info['order']).fit()

                forecast_raw = val_model.forecast(steps=len(test_data))
                forecast = clip_nonnegative(forecast_raw)

                mae = mean_absolute_error(test_data, forecast)
                smape_val = smape(test_data, forecast)
                rmse = np.sqrt(mean_squared_error(test_data, forecast))

                validation_results[ctype] = {
                    'mae': mae,
                    'smape': smape_val,
                    'rmse': rmse,
                    'test_months': len(test_data),
                    'mean_actual': test_data.mean()
                }

                rel_mae = mae / (test_data.mean() if test_data.mean() > EPS else 1.0) * 100.0
                print(f"  {ctype}:")
                print(f"    sMAPE: {smape_val:.1f}%")
                print(f"    MAE: {mae:,.0f}")
                print(f"    Relative MAE: {rel_mae:.1f}%")

            except Exception as e:
                print(f"  ❌ Validation failed for {ctype}: {e}")
                continue

        self.validation_results = validation_results
        return validation_results

    def generate_forecasts(self):
        """Generate strategic forecasts for all content types (nonnegative, safe growth)."""
        print(f"\n=== GENERATING {FORECAST_PERIODS}-MONTH STRATEGIC FORECASTS ===")

        for ctype, model_info in self.models.items():
            try:
                model = model_info['model']
                fc_raw, conf_intervals = model.predict(
                    n_periods=FORECAST_PERIODS,
                    return_conf_int=True,
                    alpha=1 - CONFIDENCE_LEVEL
                )

                forecast_values = clip_nonnegative(fc_raw)
                lower_ci = clip_nonnegative(conf_intervals[:, 0])
                upper_ci = clip_nonnegative(conf_intervals[:, 1])

                last_date = self.monthly_data[self.monthly_data['content_type'] == ctype]['date'].max()
                forecast_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=FORECAST_PERIODS,
                    freq='MS'
                )

                growth = safe_growth(forecast_values[-1], model_info['last_value'])

                self.forecasts[ctype] = {
                    'dates': forecast_dates,
                    'forecast': forecast_values,
                    'lower_ci': lower_ci,
                    'upper_ci': upper_ci,
                    'total_forecast': float(np.sum(forecast_values)),
                    'avg_monthly': float(np.mean(forecast_values)),
                    'growth_rate': growth,
                    'model_info': model_info
                }

                val_smape = getattr(self, 'validation_results', {}).get(ctype, {}).get('smape', None)
                val_str = f" (Val sMAPE: {val_smape:.1f}%)" if val_smape is not None else ""

                print(f"\n  {ctype} Forecast{val_str}:")
                print(f"    Next month: {forecast_values[0]:,.0f} engagement")
                print(f"    {FORECAST_PERIODS}-month total: {np.sum(forecast_values):,.0f}")
                print(f"    Monthly average: {np.mean(forecast_values):,.0f}")
                if is_finite_pos(growth):
                    print(f"    Growth trajectory: {growth:.1f}%")
                else:
                    print(f"    Growth trajectory: N/A (insufficient or zero last value)")

            except Exception as e:
                print(f"  ❌ Forecast failed for {ctype}: {e}")
                continue

        print(f"\n✅ Generated forecasts for {len(self.forecasts)} content types")
        return len(self.forecasts) > 0

    def create_strategic_dashboard(self):
        """Create visualization dashboard with guards for NaN/negatives."""
        print("\n=== CREATING STRATEGIC DASHBOARD ===")

        n_content_types = len(self.forecasts)
        if n_content_types == 0:
            print("❌ No forecasts available for dashboard")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # 1) Historical trends
        ax1 = axes[0]
        for ctype in self.forecasts.keys():
            type_data = self.monthly_data[self.monthly_data['content_type'] == ctype]
            ax1.plot(type_data['date'], type_data['total_engagement'],
                     marker='o', linewidth=2, label=ctype, alpha=0.8)
        ax1.set_title('Historical Monthly Engagement by Content Type', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Engagement')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2) Forecast lines + CI
        ax2 = axes[1]
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.forecasts)))
        for i, (ctype, fcd) in enumerate(self.forecasts.items()):
            dates, fc, low, up = fcd['dates'], fcd['forecast'], fcd['lower_ci'], fcd['upper_ci']
            ax2.plot(dates, fc, marker='o', linewidth=2, label=f"{ctype} Forecast", color=colors[i])
            ax2.fill_between(dates, low, up, alpha=0.2, color=colors[i])
        ax2.set_title(f'{FORECAST_PERIODS}-Month Strategic Forecasts', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Predicted Engagement')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3) Totals bar
        ax3 = axes[2]
        content_types = list(self.forecasts.keys())
        totals = [self.forecasts[ct]['total_forecast'] for ct in content_types]
        bars = ax3.bar(content_types, totals, alpha=0.7, color=colors[:len(content_types)])
        ax3.set_title(f'{FORECAST_PERIODS}-Month Total Engagement Forecast', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Total Predicted Engagement')
        plt.setp(ax3.get_xticklabels(), rotation=45)

        # Value labels with guards
        if totals:
            max_abs = max(np.abs(t) for t in totals)
        else:
            max_abs = 0.0
        bump = max_abs * 0.02 if max_abs > 0 else 1.0
        for bar, total in zip(bars, totals):
            if is_finite_pos(total):
                y = total + (bump if total >= 0 else -bump)
                va = 'bottom' if total >= 0 else 'top'
                ax3.text(bar.get_x() + bar.get_width()/2, y, f'{total:,.0f}', ha='center', va=va, fontweight='bold')

        # 4) Growth bar
        ax4 = axes[3]
        growth_rates = [self.forecasts[ct]['growth_rate'] for ct in content_types]
        colors_growth = ['green' if (is_finite_pos(g) and g > 0) else 'red' for g in growth_rates]
        bars_g = ax4.bar(content_types, [g if is_finite_pos(g) else 0.0 for g in growth_rates],
                         alpha=0.7, color=colors_growth)
        ax4.set_title('12-Month Growth Trajectory (%)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Growth Rate (%)')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.setp(ax4.get_xticklabels(), rotation=45)

        for bar, g in zip(bars_g, growth_rates):
            if is_finite_pos(g):
                y_pos = bar.get_height() + (1 if g > 0 else -3)
                va = 'bottom' if g > 0 else 'top'
                ax4.text(bar.get_x() + bar.get_width()/2, y_pos, f'{g:.1f}%', ha='center', va=va, fontweight='bold')
            else:
                ax4.text(bar.get_x() + bar.get_width()/2, 1, 'N/A', ha='center', va='bottom', fontstyle='italic')

        plt.tight_layout()
        plt.show()
        print("✅ Strategic dashboard generated")

    def generate_strategic_report(self):
        """Generate comprehensive strategic report"""
        print("\n" + "="*80)
        print("🎯 SUGARCANE STRATEGIC SOCIAL MEDIA FORECAST REPORT")
        print("="*80)

        # Executive Summary
        print("\n📊 EXECUTIVE SUMMARY")
        print("-" * 40)

        total_current = sum([self.models[ct]['last_value'] for ct in self.forecasts.keys()])
        total_forecast = sum([self.forecasts[ct]['total_forecast'] for ct in self.forecasts.keys()])

        print(f"Forecast Period: {FORECAST_PERIODS} months")
        print(f"Content Types Analyzed: {len(self.forecasts)}")
        print(f"Current Monthly Engagement: {total_current:,.0f}")
        print(f"Predicted {FORECAST_PERIODS}-Month Total: {total_forecast:,.0f}")
        print(f"Average Monthly Projection: {total_forecast/FORECAST_PERIODS:,.0f}")

        # Detailed forecasts by content type
        print("\n📈 DETAILED FORECASTS BY CONTENT TYPE")
        print("-" * 50)

        for ctype, forecast_data in self.forecasts.items():
            print(f"\n🎬 {ctype.upper()}")
            model_info = forecast_data['model_info']
            print(f"  Model: ARIMA{model_info['order']}")
            if model_info.get('seasonal_order'):
                print(f"  Seasonal: {model_info['seasonal_order']}")

            if hasattr(self, 'validation_results') and ctype in self.validation_results:
                val_smape = self.validation_results[ctype]['smape']
                print(f"  Validation Accuracy: {val_smape:.1f}% sMAPE")

            print(f"  Next Month: {forecast_data['forecast'][0]:,.0f} engagement")
            print(f"  Quarter 1 Total: {forecast_data['forecast'][:3].sum():,.0f}")
            print(f"  Quarter 2 Total: {forecast_data['forecast'][3:6].sum():,.0f}")
            print(f"  {FORECAST_PERIODS}-Month Total: {forecast_data['total_forecast']:,.0f}")
            if is_finite_pos(forecast_data['growth_rate']):
                print(f"  Growth Trajectory: {forecast_data['growth_rate']:+.1f}%")
            else:
                print("  Growth Trajectory: N/A (insufficient or zero last value)")

            q1_lower = forecast_data['lower_ci'][:3].sum()
            q1_upper = forecast_data['upper_ci'][:3].sum()
            print(f"  Q1 Range: {q1_lower:,.0f} - {q1_upper:,.0f} ({CONFIDENCE_LEVEL*100:.0f}% confidence)")

        # Strategic recommendations (simple heuristics)
        print("\n🎯 STRATEGIC RECOMMENDATIONS")
        print("-" * 40)
        if len(self.forecasts) > 1:
            best_type = max(self.forecasts.keys(), key=lambda x: self.forecasts[x]['total_forecast'])
            worst_type = min(self.forecasts.keys(), key=lambda x: self.forecasts[x]['total_forecast'])

            print(f"1. CONTENT FOCUS: Prioritize {best_type} content")
            print(f"   - Highest projected engagement: {self.forecasts[best_type]['total_forecast']:,.0f}")
            if is_finite_pos(self.forecasts[best_type]['growth_rate']):
                print(f"   - Growth trajectory: {self.forecasts[best_type]['growth_rate']:+.1f}%")

            print(f"\n2. OPTIMIZATION OPPORTUNITY: Improve {worst_type} performance")
            print(f"   - Currently lowest projected: {self.forecasts[worst_type]['total_forecast']:,.0f}")
            if is_finite_pos(self.forecasts[worst_type]['growth_rate']):
                print(f"   - Growth trajectory: {self.forecasts[worst_type]['growth_rate']:+.1f}%")

            growth_types = [(ct, data['growth_rate']) for ct, data in self.forecasts.items()]
            growth_types.sort(key=lambda x: (-1 if not is_finite_pos(x[1]) else 0, x[1]), reverse=True)
            print(f"\n3. GROWTH RANKING:")
            for i, (ctype, growth) in enumerate(growth_types, 1):
                if is_finite_pos(growth):
                    status = "📈" if growth > 0 else "📉"
                    print(f"   {i}. {ctype}: {growth:+.1f}% {status}")
                else:
                    print(f"   {i}. {ctype}: N/A")

            print(f"\n4. RESOURCE ALLOCATION (Based on {FORECAST_PERIODS}-month forecasts):")
            total_all = sum(data['total_forecast'] for data in self.forecasts.values())
            for ctype, data in self.forecasts.items():
                percentage = (data['total_forecast'] / total_all) * 100 if total_all > 0 else 0
                print(f"   {ctype}: {percentage:.1f}% of resources")
        else:
            print("1. Single content type analysis - focus on optimizing current strategy")
            single_type = list(self.forecasts.keys())[0]
            growth = self.forecasts[single_type]['growth_rate']
            if is_finite_pos(growth) and growth > 0:
                print(f"   - Positive growth trajectory: {growth:.1f}% — continue current strategy")
            else:
                print("   - Declining/uncertain trajectory — adjust content strategy")

        print("\n" + "="*80)
        print("✅ Strategic forecast complete. Use for budget planning and resource allocation.")
        print("📧 Contact: Sugarcane Artist Management Strategic Analytics Team")
        print("="*80)

# -----------------------
# Main execution
# -----------------------
def run_strategic_forecasting():
    """Main function to run complete strategic forecasting analysis"""
    print("🎯 SUGARCANE STRATEGIC SOCIAL MEDIA FORECASTING")
    print("Long-term planning and resource allocation insights")
    print("FIXED VERSION - Resolves pandas frequency issues / nonnegative forecasts / sMAPE")
    print("="*70)

    forecaster = SugarcaneStrategicForecasting()

    if not forecaster.load_and_aggregate_data():
        print("❌ Failed to load data. Exiting.")
        return None

    forecaster.analyze_time_series_properties()

    if not forecaster.build_arima_models():
        print("❌ Failed to build models. Exiting.")
        return None

    forecaster.validate_models()

    if not forecaster.generate_forecasts():
        print("❌ Failed to generate forecasts. Exiting.")
        return None

    try:
        forecaster.create_strategic_dashboard()
    except Exception as e:
        print(f"⚠️  Dashboard creation failed: {e}")
        print("Continuing with report generation...")

    forecaster.generate_strategic_report()

    print("\n🎉 FORECASTING COMPLETE!")
    print("💡 Key Fixes in this Version:")
    print("   - Clips negatives (engagement cannot be < 0)")
    print("   - sMAPE for validation (stable near zero)")
    print("   - Safe growth rates (no -inf)")
    print("   - Chart labels guarded (no posx/posy spam)")
    print("   - No pmdarima dependency; manual ARIMA selection")
    return forecaster

if __name__ == "__main__":
    strategic_system = run_strategic_forecasting()
