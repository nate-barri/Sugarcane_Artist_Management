######################################################################
# AGGREGATE CATALOG GROWTH PREDICTOR (ENHANCED METRICS + HIST/FC)
# Predicts total 6-month growth for ALL existing videos combined
# Added: Show forecast visuals + separate aggregate dashboard panels
# Note: No equations were altered; only plotting/display organization changed.
######################################################################


import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')


# Database connection (unchanged)
db_params = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host': 'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port': '5432',
    'sslmode': 'require'
}




def _show_and_pause():
    """Show current figure non-blocking for a short time then close.
    Adjust pause duration here if you'd like plots to stay visible longer."""
    plt.show(block=True)


    plt.close()




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
            print(f"✗ Error loading data: {e}")
            return False


    def _prepare_features(self):
        """Feature engineering (kept unchanged)."""
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
        """Content type classifier (unchanged)."""
        t = str(title).lower()
        if 'official music video' in t:
            return 'music_video'
        if 'lyric' in t or 'lyrics' in t:
            return 'lyrics'
        if 'live' in t:
            return 'live'
        return 'other'


    def calculate_comprehensive_metrics(self, y_true, y_pred, dataset_name=""):
        """Calculate R², MAE, RMSE, MAPE, MASE, Median APE (unchanged)."""
        mask = y_true > 0
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]


        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))


        if len(y_true_filtered) > 0:
            mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
            ape = np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered) * 100
            median_ape = np.median(ape)
        else:
            mape = np.nan
            median_ape = np.nan


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


    def train_model(self):
        """Train Gradient Boosting on mature videos (kept unchanged)."""
        sep = '=' * 70
        print(f"\n{sep}")
        print("TRAINING MODEL FOR CATALOG PREDICTIONS")
        print(sep)


        mature_df = self.df[self.df['months_since_publish'] >= 6].copy()
        print(f"\n📊 Training Set:")
        print(f"   Videos (6+ months):          {len(mature_df)}")


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


        # Predictions (log space)
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)


        # Convert back from log space
        y_train_actual = np.expm1(y_train)
        y_test_actual = np.expm1(y_test)
        y_pred_train_actual = np.expm1(y_pred_train)
        y_pred_test_actual = np.expm1(y_pred_test)


        # Calculate comprehensive metrics (unchanged)
        train_metrics = self.calculate_comprehensive_metrics(
            y_train_actual.values, y_pred_train_actual, "Training"
        )
        test_metrics = self.calculate_comprehensive_metrics(
            y_test_actual.values, y_pred_test_actual, "Test"
        )


        # Store metrics
        self.metrics['train'] = train_metrics
        self.metrics['test'] = test_metrics


        # Print short notice (we won't plot the large metrics dashboard)
        print(f"\n✓ Model trained successfully (metrics computed)")


        return test_metrics['r2'], test_metrics['mae']


    def predict_all_videos(self):
        """Generate predictions for all existing videos (kept unchanged)."""
        print(f"\n{'='*70}")
        print("GENERATING PREDICTIONS FOR EXISTING CATALOG")
        print('='*70)


        predictions = []
        for idx, video in self.df.iterrows():
            if video['impressions'] <= 0:
                continue


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
                'title': video.get('video_title', f'video_{idx}'),
                'type': video['content_type'],
                'actual': int(video['views']),
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
        Forecast 6-month aggregate growth for existing catalog and display:
         - Four aggregate dashboard panels, each shown separately
         - Historical, Forecast-only, Combined Historical+Forecast (shown separately)
         - Print textual summary block (MODEL PERFORMANCE METRICS / 6-MONTH GROWTH PROJECTION)
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


        # Calculate 6-month growth by segment (same logic)
        mature_growth_rate = 0.065
        mature_growth = mature_videos['actual'].sum() * mature_growth_rate


        growing_growth = 0
        for idx, video in growing_videos.iterrows():
            predicted_mature = video['predicted_mature']
            current_views = video['actual']
            remaining_growth = max(0, predicted_mature - current_views)
            growing_growth += remaining_growth * 0.7


        new_growth = 0
        for idx, video in new_videos.iterrows():
            predicted_mature = video['predicted_mature']
            current_views = video['actual']
            remaining_growth = max(0, predicted_mature - current_views)
            new_growth += remaining_growth * 0.85


        total_6mo_growth = mature_growth + growing_growth + new_growth
        projected_6mo_total = total_current_views + total_6mo_growth


        # Scenario intervals
        conservative_growth = total_6mo_growth * 0.7
        optimistic_growth = total_6mo_growth * 1.3


        # ---------- DISPLAY: FOUR PANELS SEPARATELY ----------
        # Panel A: Cumulative growth projection (single figure)
        try:
            months = np.arange(0, 7)
            baseline_line = np.linspace(total_current_views, projected_6mo_total, 7)
            conservative_line = np.linspace(total_current_views, total_current_views + conservative_growth, 7)
            optimistic_line = np.linspace(total_current_views, total_current_views + optimistic_growth, 7)


            plt.figure(figsize=(10, 6))
            plt.plot(months, baseline_line, 'g-', linewidth=3, marker='o', markersize=6, label='Baseline Projection')
            plt.fill_between(months, conservative_line, optimistic_line, alpha=0.2, color='green', label='Confidence Range')
            plt.axhline(total_current_views, color='gray', linestyle='--', label='Current Total')
            plt.title('6-Month Catalog Growth Trajectory', fontsize=14, fontweight='bold')
            plt.xlabel('Months from Now')
            plt.ylabel('Total Catalog Views')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'
            ))
            plt.tight_layout()
            _show_and_pause()
        except Exception as e:
            print(f"✗ Could not generate panel A: {e}")


        # Panel B: Growth by video age segment (single figure)
        try:
            segments = ['Mature\n(6+ mo)', 'Growing\n(3-6 mo)', 'New\n(<3 mo)']
            current_views = [mature_videos['actual'].sum(), growing_videos['actual'].sum(), new_videos['actual'].sum()]


            mature_proj = current_views[0] * (1 + mature_growth_rate)
            growing_proj = current_views[1] + growing_growth  # already computed as additional
            new_proj = current_views[2] + new_growth


            projected_views = [mature_proj, growing_proj, new_proj]


            x = np.arange(len(segments))
            width = 0.35


            plt.figure(figsize=(10, 6))
            plt.bar(x - width/2, current_views, width, label='Current', color='steelblue', edgecolor='k')
            plt.bar(x + width/2, projected_views, width, label='Projected (6mo)', color='orange', edgecolor='k')
            plt.xticks(x, segments)
            plt.ylabel('Total Views')
            plt.title('Growth by Video Age Segment', fontsize=14, fontweight='bold')
            for i, v in enumerate(current_views):
                label = f'{v/1e6:.1f}M' if v >= 1e6 else f'{v/1e3:.0f}K'
                plt.text(i - width/2, v + max(1, v*0.01), label, ha='center', va='bottom', fontsize=9)
            for i, v in enumerate(projected_views):
                label = f'{int(v)/1e6:.1f}M' if v >= 1e6 else f'{v/1e3:.0f}K'
                plt.text(i + width/2, v + max(1, v*0.01), label, ha='center', va='bottom', fontsize=9)
            plt.legend()
            plt.grid(alpha=0.3, axis='y')
            plt.tight_layout()
            _show_and_pause()
        except Exception as e:
            print(f"✗ Could not generate panel B: {e}")


        # Panel C: 6-Month Growth Sources (pie) (single figure)
        try:
            mature_growth_calc = mature_growth
            growing_growth_calc = growing_growth
            new_growth_calc = new_growth
            growth_sources = [mature_growth_calc, growing_growth_calc, new_growth_calc]


            labels = [
                f'Mature Catalog\n{mature_growth_calc:,.0f}',
                f'Growing Videos\n{growing_growth_calc:,.0f}',
                f'New Videos\n{new_growth_calc:,.0f}'
            ]


            plt.figure(figsize=(8, 8))
            # if all zero, show a placeholder
            if sum(growth_sources) == 0:
                plt.text(0.5, 0.5, 'No projected growth', ha='center', va='center', fontsize=14)
            else:
                plt.pie(growth_sources, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title('6-Month Growth Sources', fontsize=14, fontweight='bold')
            plt.tight_layout()
            _show_and_pause()
        except Exception as e:
            print(f"✗ Could not generate panel C: {e}")


        # Panel D: Monthly Growth Breakdown (single figure)
        try:
            monthly_growth = total_6mo_growth / 6 if total_6mo_growth != 0 else 0
            months_labels = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6']
            monthly_vals = [monthly_growth] * 6
            cumulative_monthly = np.cumsum(monthly_vals)


            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(months_labels, monthly_vals, color='coral', edgecolor='k', alpha=0.7)
            ax.set_ylabel('Monthly Growth (views)', fontsize=11)
            ax.set_title('Monthly Growth Breakdown', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3, axis='y')


            ax_twin = ax.twinx()
            ax_twin.plot(months_labels, cumulative_monthly, 'g-o', linewidth=2.5, markersize=8, label='Cumulative')
            ax_twin.set_ylabel('Cumulative Growth (views)', fontsize=11, color='green')
            ax_twin.legend(loc='upper left')


            for bar in bars:
                h = bar.get_height()
                lbl = f'{h/1e3:.0f}K' if h >= 1e3 else f'{h:.0f}'
                ax.text(bar.get_x() + bar.get_width()/2., h + max(1, h*0.01), lbl, ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            _show_and_pause()
        except Exception as e:
            print(f"✗ Could not generate panel D: {e}")


        # ---------- ALSO: Historical-only, Forecast-only, Combined ----------


        # HISTORICAL ONLY
        try:
            months_back = 6
            x_hist = np.arange(-months_back, 1)
            hist = []
            for i in range(0, months_back + 1):
                frac = i / months_back
                hist_value = total_current_views - total_6mo_growth * (1 - frac)
                hist.append(hist_value)


            plt.figure(figsize=(10, 5))
            plt.plot(x_hist, hist, marker='o', linewidth=3, label='Historical (backcast)', color='tab:blue')
            plt.axvline(0, color='gray', linestyle='--')
            plt.title('Historical 6-Month (Backcast)', fontsize=14, fontweight='bold')
            plt.xlabel('Months (negative = past)')
            plt.ylabel('Total Catalog Views')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            _show_and_pause()
        except Exception as e:
            print(f"✗ Could not generate historical-only plot: {e}")


        # FORECAST ONLY
        try:
            months_forward = 6
            x_fore = np.arange(0, months_forward + 1)
            forecast_line = np.linspace(total_current_views, projected_6mo_total, months_forward + 1)
            lower = np.linspace(total_current_views, total_current_views + conservative_growth, months_forward + 1)
            upper = np.linspace(total_current_views, total_current_views + optimistic_growth, months_forward + 1)


            plt.figure(figsize=(10, 5))
            plt.plot(x_fore, forecast_line, marker='o', linewidth=3, label='Baseline Projection', color='tab:orange')
            plt.fill_between(x_fore, lower, upper, alpha=0.18, label='Confidence range (70%-130%)')
            plt.axvline(0, color='gray', linestyle='--')
            plt.title('Predicted 6-Month Forecast', fontsize=14, fontweight='bold')
            plt.xlabel('Months (positive = future)')
            plt.ylabel('Total Catalog Views')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            _show_and_pause()
        except Exception as e:
            print(f"✗ Could not generate forecast-only plot: {e}")


        # COMBINED HISTORICAL + FORECAST (no arrows)
        try:
            months_back = 6
            months_forward = 6
            x_hist = np.arange(-months_back, 1)
            hist = []
            for i in range(0, months_back + 1):
                frac = i / months_back
                hist_value = total_current_views - total_6mo_growth * (1 - frac)
                hist.append(hist_value)


            x_fore = np.arange(1, months_forward + 1)
            forecast = []
            for i in range(1, months_forward + 1):
                frac = i / months_forward
                forecast_value = total_current_views + total_6mo_growth * frac
                forecast.append(forecast_value)


            plt.figure(figsize=(12, 6))
            plt.plot(x_hist, hist, marker='o', linewidth=3, label='Historical (backcast)', color='tab:blue')
            plt.plot(x_fore, forecast, marker='o', linewidth=3, label='Forecast (baseline 6mo)', color='tab:orange')


            cons_line = np.linspace(total_current_views, total_current_views + conservative_growth, months_forward + 1)[1:]
            opt_line = np.linspace(total_current_views, total_current_views + optimistic_growth, months_forward + 1)[1:]
            plt.fill_between(x_fore, cons_line, opt_line, alpha=0.18, label='Confidence range (70%-130%)')


            plt.axvline(0, color='gray', linestyle='--', linewidth=1)
            plt.text(0, plt.gca().get_ylim()[1]*0.98, ' Now ', ha='center', va='top', backgroundcolor='white')


            plt.xlabel('Months (negative = past, positive = future)')
            plt.ylabel('Total Catalog Views')
            xticks = list(x_hist) + list(x_fore)
            xticklabels = [f"-{months_back - i}m" if i < months_back else "Now" for i in range(0, months_back + 1)] + [f"+{i}m" for i in range(1, months_forward + 1)]
            plt.xticks(xticks, xticklabels, rotation=45)


            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'
            ))


            # Text annotations (no arrows)
            plt.text(1, total_current_views * 1.01, f"Current: {total_current_views:,}", fontsize=9)
            plt.text(months_forward, projected_6mo_total * 1.01, f"Projected (6m): {projected_6mo_total:,.0f}", fontsize=9)


            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            _show_and_pause()


            print(f"\n✓ Historical + Forecast plot generated")
        except Exception as e:
            print(f"\n✗ Could not generate historical+forecast plot: {e}")
            import traceback
            traceback.print_exc()


        # ---------- PRINT TEXT SUMMARY BLOCK (exact format requested) ----------
        try:
            print("\nMODEL PERFORMANCE METRICS")
            print(" 6-MONTH GROWTH PROJECTION:\n")
            print(f"{'Segment':<28} {'Current Views':>15} {'Growth':>12} {'% Growth':>10}")
            print(f"{'-'*70}")
            # Print rows
            print(f"{'Mature catalog':<28} {mature_videos['actual'].sum():>15,} {int(mature_growth):>12,} { (mature_growth/(mature_videos['actual'].sum() if mature_videos['actual'].sum()>0 else 1))*100:>9.1f}%")
            print(f"{'Growing videos':<28} {growing_videos['actual'].sum():>15,} {int(growing_growth):>12,} { (growing_growth/(growing_videos['actual'].sum() if growing_videos['actual'].sum()>0 else 1))*100:>9.1f}%")
            print(f"{'New videos':<28} {new_videos['actual'].sum():>15,} {int(new_growth):>12,} { (new_growth/(new_videos['actual'].sum() if new_videos['actual'].sum()>0 else 1))*100:>9.1f}%")
            print(f"{'-'*70}")
            print(f"{'TOTAL':<28} {total_current_views:>15,} {int(total_6mo_growth):>12,} { (total_6mo_growth/(total_current_views if total_current_views>0 else 1))*100:>9.1f}%")


            print("\nSUMMARY:")
            print(f"   Current catalog:     {total_current_views:>12,} views")
            print(f"   6-month projection:  {projected_6mo_total:>12,} views")
            print(f"   Expected growth:     {int(total_6mo_growth):>12,} views ({(total_6mo_growth/(total_current_views if total_current_views>0 else 1))*100:.1f}%)")
            print(f"   Monthly growth rate: {( (total_6mo_growth/(total_current_views if total_current_views>0 else 1))*100 / 6 ):>12.2f}% per month")
        except Exception as e:
            print(f"✗ Could not print summary: {e}")


        # Return numbers as before
        return {
            'current_total': int(total_current_views),
            'projected_6mo_total': int(projected_6mo_total),
            'total_growth': int(total_6mo_growth),
            'growth_pct': (total_6mo_growth/(total_current_views if total_current_views>0 else 1))*100,
            'mature_growth': int(mature_growth),
            'growing_growth': int(growing_growth),
            'new_growth': int(new_growth)
        }


    def plot_metrics_comparison(self):
        """Deprecated for display — metrics computed but not plotted here."""
        print("⚠ Metrics computed (not displayed as combined dashboard in this version).")
        return


    def run_pipeline(self):
        """Execute complete aggregate prediction pipeline."""
        header = '#' * 70
        print(f"\n{header}")
        print('AGGREGATE CATALOG GROWTH PREDICTOR')
        print('Existing Videos Only - No New Releases')
        print(header)


        if not self.load_data():
            return False


        # Train model to compute metrics (kept)
        self.train_model()


        # Predict all videos
        pred_df = self.predict_all_videos()


        # Forecast aggregate growth and show all requested plots & summary
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


        # --- Print Evaluation Metrics ---
        print("\nMODEL PERFORMANCE METRICS")
        print("=" * 70)


        for dataset, metrics in self.metrics.items():
            print(f"\n{dataset.upper()} SET:")
            print(f"{'-'*70}")
            print(f"{'R2':<15}: {metrics['r2']:.4f}")
            print(f"{'MAE':<15}: {metrics['mae']:,.4f}")
            print(f"{'RMSE':<15}: {metrics['rmse']:,.4f}")
            print(f"{'MAPE':<15}: {metrics['mape']:.2f}%")
            print(f"{'Median APE':<15}: {metrics['median_ape']:.2f}%")
            print(f"{'MASE':<15}: {metrics['mase']:.4f}")
            print(f"{'-'*70}")


        return True






if __name__ == '__main__':
    predictor = AggregateCatalogPredictor(db_params)
    predictor.run_pipeline()



