# youtube_ml_pipeline_final.py - Production-Ready Small Dataset ML
import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import warnings
from scipy import stats


from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                           mean_absolute_percentage_error, median_absolute_error)


warnings.filterwarnings('ignore')


# Database connection
db_params = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host': 'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port': '5432',
    'sslmode': 'require'
}


class ModelEvaluator:
    """Production-ready model evaluation for small datasets"""
   
    def __init__(self):
        self.metrics = {}
        self.predictions = {}
       
    def calculate_robust_metrics(self, y_true, y_pred, model_name="Model"):
        """Calculate metrics robust to outliers and small samples"""
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
       
        if len(y_true_clean) == 0:
            return {"error": "No valid predictions"}
       
        # Absolute errors
        mae = float(mean_absolute_error(y_true_clean, y_pred_clean))
        mad = float(median_absolute_error(y_true_clean, y_pred_clean))
       
        # Percentage errors (with outlier protection)
        median_actual = float(np.median(y_true_clean))
        mean_actual = float(np.mean(y_true_clean))
       
        # For percentage metrics, clip extreme values
        percentage_errors = np.clip(
            np.abs((y_true_clean - y_pred_clean) / np.maximum(np.abs(y_true_clean), 1)),
            0, 2  # Cap at 200%
        )
        median_ape = float(np.median(percentage_errors) * 100)
        mean_ape = float(np.mean(percentage_errors) * 100)
       
        # SMAPE (symmetric, bounded 0-200%)
        smape = float(np.mean(
            2 * np.abs(y_pred_clean - y_true_clean) /
            (np.abs(y_true_clean) + np.abs(y_pred_clean) + 1e-10)
        ) * 100)
       
        # R² and correlation
        r2 = float(r2_score(y_true_clean, y_pred_clean))
        correlation = float(np.corrcoef(y_true_clean, y_pred_clean)[0, 1])
       
        # Adjusted R² for small samples
        n = len(y_true_clean)
        p = 5  # approximate features
        adj_r2 = float(1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2)
       
        # Bias
        mbe = float(np.mean(y_pred_clean - y_true_clean))
       
        # Direction accuracy (for ranking purposes)
        if len(y_true_clean) > 1:
            median_true = np.median(y_true_clean)
            direction_accuracy = np.mean(
                ((y_true_clean >= median_true) == (y_pred_clean >= median_true))
            )
        else:
            direction_accuracy = 0
       
        metrics = {
            'model_name': model_name,
            'n': len(y_true_clean),
           
            # Primary metrics
            'R²': r2,
            'Adj_R²': adj_r2,
            'Correlation': correlation,
           
            # Absolute errors
            'MAE': mae,
            'MAD': mad,
            'MBE': mbe,
           
            # Percentage errors (robust)
            'Median_APE': median_ape,
            'Mean_APE': mean_ape,
            'SMAPE': smape,
           
            # Distribution comparison
            'Actual_Median': median_actual,
            'Predicted_Median': float(np.median(y_pred_clean)),
            'Actual_Mean': mean_actual,
            'Predicted_Mean': float(np.mean(y_pred_clean)),
           
            # Practical metric
            'Direction_Accuracy': float(direction_accuracy * 100)
        }
       
        self.metrics[model_name] = metrics
        self.predictions[model_name] = {'y_true': y_true_clean, 'y_pred': y_pred_clean}
       
        return metrics
   
    def cross_validate_model(self, model, X, y, cv=5, use_log=False, model_name="Model"):
        """Perform cross-validation with detailed tracking"""
        if len(X) < cv:
            cv = max(3, len(X) // 3)
       
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
       
        all_y_true = []
        all_y_pred = []
        fold_r2 = []
        fold_mae = []
       
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
           
            if use_log:
                y_train_log = np.log1p(y_train)
                model.fit(X_train, y_train_log)
                y_pred_log = model.predict(X_test)
                y_pred = np.expm1(y_pred_log)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
           
            # Clip predictions to reasonable range
            if use_log:
                y_pred = np.clip(y_pred, 0, np.percentile(y, 99))
           
            fold_r2.append(r2_score(y_test, y_pred))
            fold_mae.append(mean_absolute_error(y_test, y_pred))
           
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
       
        return {
            'y_true': np.array(all_y_true),
            'y_pred': np.array(all_y_pred),
            'cv_r2_scores': fold_r2,
            'cv_mae_scores': fold_mae,
            'cv_r2_mean': np.mean(fold_r2),
            'cv_r2_std': np.std(fold_r2),
            'cv_mae_mean': np.mean(fold_mae),
            'cv_mae_std': np.std(fold_mae)
        }
   
    def print_report(self):
        """Print comprehensive evaluation report"""
        if not self.metrics:
            print("No metrics available")
            return
       
        print("\n" + "="*90)
        print("MODEL EVALUATION REPORT - SMALL DATASET OPTIMIZED")
        print("="*90)
       
        for model_name, m in self.metrics.items():
            print(f"\n{'─'*90}")
            print(f"  {model_name.upper()}")
            print(f"{'─'*90}")
            print(f"  Sample Size: {m['n']}")
           
            print(f"\n  📊 PREDICTIVE POWER:")
            print(f"     R² Score:           {m['R²']:7.4f}  {'✓ Good' if m['R²'] > 0.3 else '✗ Weak'}")
            print(f"     Adjusted R²:        {m['Adj_R²']:7.4f}")
            print(f"     Correlation:        {m['Correlation']:7.4f}  {'✓ Strong' if abs(m['Correlation']) > 0.6 else '✗ Weak'}")
            print(f"     Direction Accuracy: {m['Direction_Accuracy']:6.2f}%  (above/below median)")
           
            print(f"\n  📏 ERROR METRICS (Robust to Outliers):")
            print(f"     Median APE:         {m['Median_APE']:6.2f}%")
            print(f"     Mean APE:           {m['Mean_APE']:6.2f}%")
            print(f"     SMAPE:              {m['SMAPE']:6.2f}%")
            print(f"     MAD (Median AE):    {m['MAD']:,.0f}")
           
            print(f"\n  🎯 DISTRIBUTION COMPARISON:")
            print(f"     Actual Median:      {m['Actual_Median']:>12,.0f}")
            print(f"     Predicted Median:   {m['Predicted_Median']:>12,.0f}")
            print(f"     Bias (MBE):         {m['MBE']:>12,.0f}")
           
            # Overall assessment
            r2, corr, mape = m['Adj_R²'], m['Correlation'], m['Median_APE']
           
            if r2 > 0.4 and corr > 0.6 and mape < 40:
                status = "🟢 GOOD - Reliable for predictions"
            elif r2 > 0.2 and corr > 0.4 and mape < 60:
                status = "🟡 FAIR - Use with caution, good for trends"
            elif r2 > 0 and corr > 0.3:
                status = "🟠 WEAK - Better than random, good for ranking"
            else:
                status = "🔴 POOR - Not suitable for predictions"
           
            print(f"\n  📈 OVERALL: {status}")
       
        print("\n" + "="*90)
   
    def plot_evaluation(self):
        """Create separate evaluation plots for each model"""
        if not self.predictions:
            print("No predictions available for plotting")
            return
   
        models = list(self.predictions.keys())
       
        for model_name in models:
            pred_data = self.predictions[model_name]
            y_true = pred_data['y_true']
            y_pred = pred_data['y_pred']
           
            # Determine if this is a view prediction model
            is_view_prediction = 'View Prediction' in model_name
           
            # 1. Predictions vs Actuals
            fig1 = plt.figure(figsize=(8, 6))
            plt.scatter(y_true, y_pred, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
           
            # Perfect prediction line
            min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='')
           
            # Add median lines
            plt.axvline(np.median(y_true), color='blue', linestyle=':', alpha=0.5, label='Actual Median')
            plt.axhline(np.median(y_pred), color='orange', linestyle=':', alpha=0.5, label='Pred Median')
           
            plt.xlabel('Actual Values', fontweight='bold')
            plt.ylabel('Predicted Values', fontweight='bold')
            title = f'{model_name} - Predictions vs Actuals'
            if is_view_prediction:
                title += ' (View Prediction)'
            plt.title(title, fontweight='bold')
            plt.legend(loc='best', fontsize=8)
            plt.grid(True, alpha=0.3)
           
            # Stats box
            r2 = r2_score(y_true, y_pred)
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            plt.text(0.05, 0.95, f'R²={r2:.3f}\nρ={corr:.3f}',
                    transform=plt.gca().transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
           
            plt.tight_layout()
            plt.show()
           
            # 2. Residuals
            fig2 = plt.figure(figsize=(8, 6))
            residuals = y_true - y_pred
            plt.scatter(y_pred, residuals, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
            plt.axhline(0, color='red', linestyle='--', lw=2)
            plt.axhline(np.std(residuals), color='orange', linestyle=':', alpha=0.7)
            plt.axhline(-np.std(residuals), color='orange', linestyle=':', alpha=0.7)
            plt.xlabel('Predicted Values', fontweight='bold')
            plt.ylabel('Residuals', fontweight='bold')
            title = f'{model_name} - Residual Plot'
            if is_view_prediction:
                title += ' (View Prediction)'
            plt.title(title, fontweight='bold')
            plt.grid(True, alpha=0.3)
           
            plt.tight_layout()
            plt.show()
           
            # 3. Distribution Comparison
            fig3 = plt.figure(figsize=(8, 6))
            bins = min(15, len(y_true) // 3)
            plt.hist(y_true, bins=bins, alpha=0.5, label='Actual',
                    color='blue', edgecolor='black', density=True)
            plt.hist(y_pred, bins=bins, alpha=0.5, label='Predicted',
                    color='red', edgecolor='black', density=True)
            plt.xlabel('Values', fontweight='bold')
            plt.ylabel('Density', fontweight='bold')
            title = f'{model_name} - Distribution Comparison'
            if is_view_prediction:
                title += ' (View Prediction)'
            plt.title(title, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
           
            plt.tight_layout()
            plt.show()
           
            # 4. Ranking Comparison (Quintiles)
            fig4 = plt.figure(figsize=(8, 6))
           
            # Create quintiles
            true_quintiles = pd.qcut(y_true, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
            pred_quintiles = pd.qcut(y_pred, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
           
            # Confusion matrix for quintiles
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_quintiles, pred_quintiles, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
           
            im = plt.imshow(cm, cmap='Blues', aspect='auto')
            plt.xticks(range(5), ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            plt.yticks(range(5), ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            plt.xlabel('Predicted Quintile', fontweight='bold')
            plt.ylabel('Actual Quintile', fontweight='bold')
            title = f'{model_name} - Ranking Accuracy (Quintile Confusion Matrix)'
            if is_view_prediction:
                title += ' (View Prediction)'
            plt.title(title, fontweight='bold')
           
            # Add text annotations
            for i in range(5):
                for j in range(5):
                    plt.text(j, i, cm[i, j], ha="center", va="center",
                            color="black" if cm[i, j] < cm.max()/2 else "white")
           
            plt.colorbar(im)
            plt.tight_layout()
            plt.show()




class YouTubeMLPipeline:
    def __init__(self, db_params):
        self.db_params = db_params
        self.df = None
        self.scalers = {}
        self.encoders = {}
        self.models = {}
        self.evaluator = ModelEvaluator()
       
    def load_and_prepare_data(self):
        """Load and prepare data with outlier handling"""
        try:
            conn = psycopg2.connect(**self.db_params)
            query = "SELECT * FROM yt_video_etl"
            self.df = pd.read_sql(query, conn)
            conn.close()
            print(f"✓ Data loaded: {len(self.df)} records")
           
            self._prepare_features()
            self._analyze_dataset()
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
   
    def _prepare_features(self):
        """Feature engineering optimized for small datasets"""
        # Date parsing
        self.df['publish_date'] = pd.to_datetime(
            self.df['publish_year'].astype(str) + '-' +
            self.df['publish_month'].astype(str) + '-' +
            self.df['publish_day'].astype(str)
        )
       
        # Numeric conversions with outlier capping
        numeric_cols = ['duration', 'impressions', 'impressions_ctr',
                       'avg_views_per_viewer', 'subscribers_gained', 'subscribers_lost']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
                # Cap extreme outliers at 99th percentile
                if col in ['impressions', 'views']:
                    p99 = self.df[col].quantile(0.99)
                    self.df[f'{col}_capped'] = np.minimum(self.df[col], p99)
       
        # Engagement features (avoid division by zero)
        self.df['total_engagement'] = (self.df['likes'] + self.df['shares'] +
                                       self.df['comments_added'])
        self.df['engagement_rate'] = self.df['total_engagement'] / np.maximum(self.df['views'], 1)
       
        # Log-scale features for better modeling
        self.df['log_views'] = np.log1p(self.df['views'])
        self.df['log_impressions'] = np.log1p(self.df['impressions'])
       
        # Temporal features
        self.df['duration_numeric'] = pd.to_numeric(self.df['duration'], errors='coerce')
        self.df['duration_minutes'] = self.df['duration_numeric'] / 60
        self.df['day_of_week'] = self.df['publish_date'].dt.dayofweek
        self.df['month'] = self.df['publish_date'].dt.month
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        self.df['quarter'] = self.df['publish_date'].dt.quarter
        self.df['days_since_publish'] = (datetime.now() - self.df['publish_date']).dt.days
       
        # Content classification
        self.df['content_type'] = self.df['video_title'].apply(self._classify_content)
       
        # Performance tiers (useful for classification)
        self.df['view_tier'] = pd.qcut(self.df['views'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
        self.df['engagement_tier'] = pd.qcut(self.df['engagement_rate'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
       
        print(f"✓ Features engineered: {self.df.shape}")
   
    def _classify_content(self, title):
        """Classify content type"""
        title_lower = str(title).lower()
        if 'official music video' in title_lower or 'music video' in title_lower:
            return 'music_video'
        elif 'lyric' in title_lower:
            return 'lyrics'
        elif 'live' in title_lower:
            return 'live'
        else:
            return 'other'
   
    def _analyze_dataset(self):
        """Analyze dataset characteristics"""
        print(f"\n{'='*70}")
        print("DATASET ANALYSIS")
        print(f"{'='*70}")
        print(f"Total videos: {len(self.df)}")
        print(f"Date range: {self.df['publish_date'].min().date()} to {self.df['publish_date'].max().date()}")
        print(f"\nViews distribution:")
        print(f"  Min: {self.df['views'].min():,}")
        print(f"  25th percentile: {self.df['views'].quantile(0.25):,}")
        print(f"  Median: {self.df['views'].median():,}")
        print(f"  75th percentile: {self.df['views'].quantile(0.75):,}")
        print(f"  95th percentile: {self.df['views'].quantile(0.95):,}")
        print(f"  Max: {self.df['views'].max():,}")
        print(f"\nEngagement rate:")
        print(f"  Mean: {self.df['engagement_rate'].mean():.4f}")
        print(f"  Median: {self.df['engagement_rate'].median():.4f}")
        print(f"\nCategories: {self.df['category'].nunique()}")
        print(f"Content types: {self.df['content_type'].value_counts().to_dict()}")
   
    def _prepare_model_data(self, feature_cols, target_col):
        """Prepare data for modeling"""
        categorical_cols = [col for col in feature_cols if col in ['category', 'content_type', 'view_tier', 'engagement_tier']]
        numerical_cols = [col for col in feature_cols if col not in categorical_cols]
       
        X_encoded = self.df[numerical_cols].copy()
       
        # One-hot encode categorical features for better small dataset performance
        for col in categorical_cols:
            if col in feature_cols:
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                X_encoded = pd.concat([X_encoded, dummies], axis=1)
       
        X = X_encoded.values
        y = self.df[target_col].values
       
        return X, y, list(X_encoded.columns)
   
    def build_view_prediction_ensemble(self):
        """Build ensemble of simple models for view prediction"""
        print(f"\n{'='*70}")
        print("VIEW PREDICTION - ENSEMBLE APPROACH")
        print(f"{'='*70}")
       
        # Use log-transformed features to handle skew
        feature_cols = ['log_impressions', 'impressions_ctr', 'duration_minutes',
                       'day_of_week', 'is_weekend', 'content_type']
        target_col = 'views'
       
        X, y, feature_names = self._prepare_model_data(feature_cols, target_col)
       
        print(f"Features: {len(feature_names)}")
        print(f"Samples: {len(X)}")
       
        # Create more granular stratification bins to avoid extreme outliers in one split
        # Use log-transformed views for better stratification
        y_log_for_strat = np.log1p(y)
        n_bins = min(5, len(y) // 10)  # More bins for better distribution
        try:
            strata = pd.qcut(y_log_for_strat, q=n_bins, labels=False, duplicates='drop')
        except:
            strata = pd.qcut(y, q=3, labels=False, duplicates='drop')
       
        # Split data with improved stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=strata
        )
       
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_scaled = scaler.transform(X)
       
        # Use log-transformed target
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)
        y_log = np.log1p(y)
       
        print(f"\nTraining 3 models in ensemble...")
        print(f"Train set: min={y_train.min():,.0f}, median={np.median(y_train):,.0f}, max={y_train.max():,.0f}")
        print(f"Test set: min={y_test.min():,.0f}, median={np.median(y_test):,.0f}, max={y_test.max():,.0f}")
       
        # Model 1: Ridge (linear, robust to multicollinearity)
        model1 = Ridge(alpha=5.0, random_state=42)
        model1.fit(X_train_scaled, y_train_log)
       
        # Model 2: Huber (robust to outliers)
        model2 = HuberRegressor(epsilon=1.35, max_iter=300)
        model2.fit(X_train_scaled, y_train_log)
       
        # Model 3: Random Forest with more capacity for outliers
        model3 = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        )
        model3.fit(X_train_scaled, y_train_log)
       
        # Ensemble predictions with weighted average (RF gets most weight for outliers)
        y_pred1_log = model1.predict(X_test_scaled)
        y_pred2_log = model2.predict(X_test_scaled)
        y_pred3_log = model3.predict(X_test_scaled)
       
        # Weighted ensemble: RF 60%, Huber 25%, Ridge 15%
        y_pred_log_ensemble = 0.6 * y_pred3_log + 0.25 * y_pred2_log + 0.15 * y_pred1_log
        y_pred_ensemble = np.expm1(y_pred_log_ensemble)
       
        # More lenient clipping - allow predictions up to 150% of training max
        max_reasonable = np.percentile(y_train, 99) * 2.5
        y_pred_ensemble = np.clip(y_pred_ensemble, 0, max_reasonable)
       
        print(f"Max allowed prediction: {max_reasonable:,.0f}")
       
        # Evaluate holdout
        print("\n1️⃣  HOLDOUT EVALUATION:")
        metrics_holdout = self.evaluator.calculate_robust_metrics(
            y_test, y_pred_ensemble, "View Prediction (Holdout)"
        )
       
        # Cross-validation
        print("\n2️⃣  CROSS-VALIDATION:")
        cv_results = self.evaluator.cross_validate_model(
            model2, X_scaled, y_log, cv=5, use_log=True, model_name="View Prediction"
        )
       
        print(f"   CV R² (mean ± std): {cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}")
        print(f"   CV MAE (mean ± std): {cv_results['cv_mae_mean']:,.0f} ± {cv_results['cv_mae_std']:,.0f}")
       
        metrics_cv = self.evaluator.calculate_robust_metrics(
            cv_results['y_true'], cv_results['y_pred'], "View Prediction (CV)"
        )
       
        # Feature importance (from RF)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model3.feature_importances_
        }).sort_values('importance', ascending=False)
       
        print(f"\n📊 TOP FEATURES:")
        for idx, row in importance_df.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
       
        # Save model
        self.models['view_prediction'] = {
            'ensemble': [model1, model2, model3],
            'scaler': scaler,
            'feature_names': feature_names,
            'use_log': True
        }
       
        return metrics_holdout, metrics_cv
   
    def build_engagement_predictor(self):
        """Build engagement rate predictor - using total engagement instead"""
        print(f"\n{'='*70}")
        print("ENGAGEMENT PREDICTION (Total Engagement)")
        print(f"{'='*70}")
       
        # Use total_engagement instead of rate (more predictable with absolute numbers)
        feature_cols = ['log_views', 'log_impressions', 'impressions_ctr',
                       'duration_minutes', 'day_of_week', 'is_weekend', 'content_type']
        target_col = 'total_engagement'
       
        X, y, feature_names = self._prepare_model_data(feature_cols, target_col)
       
        # Remove invalid targets (negative or extreme outliers)
        valid_mask = np.isfinite(y) & (y >= 0) & (y <= np.percentile(y, 99))
        X = X[valid_mask]
        y = y[valid_mask]
       
        print(f"Valid samples: {len(X)}")
        print(f"Target range: {y.min():.0f} to {y.max():.0f} (median: {np.median(y):.0f})")
       
        # Split with stratification
        try:
            strata = pd.qcut(y, q=3, labels=False, duplicates='drop')
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=strata
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )
       
        # Scale
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_scaled = scaler.transform(X)
       
        # Use log-transformed target for better predictions
        y_train_log = np.log1p(y_train)
        y_log = np.log1p(y)
       
        # Ensemble of models
        model1 = Ridge(alpha=5.0, random_state=42)
        model1.fit(X_train_scaled, y_train_log)
       
        model2 = HuberRegressor(epsilon=1.35, max_iter=300)
        model2.fit(X_train_scaled, y_train_log)
       
        model3 = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42
        )
        model3.fit(X_train_scaled, y_train_log)
       
        # Weighted ensemble prediction
        y_pred1_log = model1.predict(X_test_scaled)
        y_pred2_log = model2.predict(X_test_scaled)
        y_pred3_log = model3.predict(X_test_scaled)
       
        y_pred_log = 0.5 * y_pred3_log + 0.3 * y_pred2_log + 0.2 * y_pred1_log
        y_pred = np.expm1(y_pred_log)
       
        # Clip predictions to reasonable range
        y_pred = np.clip(y_pred, 0, np.percentile(y_train, 99) * 1.5)
       
        # Evaluate holdout
        print("\n1️⃣  HOLDOUT EVALUATION:")
        metrics_holdout = self.evaluator.calculate_robust_metrics(
            y_test, y_pred, "Engagement Prediction (Holdout)"
        )
       
        # Cross-validation
        print("\n2️⃣  CROSS-VALIDATION:")
        cv_results = self.evaluator.cross_validate_model(
            model2, X_scaled, y_log, cv=5, use_log=True, model_name="Engagement Prediction"
        )
       
        print(f"   CV R² (mean ± std): {cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}")
        print(f"   CV MAE (mean ± std): {cv_results['cv_mae_mean']:,.0f} ± {cv_results['cv_mae_std']:,.0f}")
       
        metrics_cv = self.evaluator.calculate_robust_metrics(
            cv_results['y_true'], cv_results['y_pred'], "Engagement Prediction (CV)"
        )
       
        # Save model ensemble
        self.models['engagement_prediction'] = {
            'ensemble': [model1, model2, model3],
            'scaler': scaler,
            'feature_names': feature_names,
            'use_log': True,
            'target_type': 'total_engagement'
        }
       
        return metrics_holdout, metrics_cv
   
    def predict_new_video(self, video_features):
        """Predict views and engagement for a new video"""
        if 'view_prediction' not in self.models:
            print("❌ Models not trained yet!")
            return None
       
        # Prepare features for view prediction
        feature_dict_views = {
            'log_impressions': np.log1p(video_features.get('impressions', 0)),
            'impressions_ctr': video_features.get('impressions_ctr', 0),
            'duration_minutes': video_features.get('duration_minutes', 0),
            'day_of_week': video_features.get('day_of_week', 0),
            'is_weekend': video_features.get('is_weekend', 0),
            'content_type': video_features.get('content_type', 'other')
        }
       
        # View prediction
        view_model_info = self.models['view_prediction']
        X_view = self._encode_features(feature_dict_views, view_model_info['feature_names'])
        X_view_scaled = view_model_info['scaler'].transform([X_view])
       
        # Ensemble prediction with weights
        models = view_model_info['ensemble']
        y_pred1_log = models[0].predict(X_view_scaled)[0]
        y_pred2_log = models[1].predict(X_view_scaled)[0]
        y_pred3_log = models[2].predict(X_view_scaled)[0]
       
        # Weighted average: RF 60%, Huber 25%, Ridge 15%
        avg_pred_log = 0.6 * y_pred3_log + 0.25 * y_pred2_log + 0.15 * y_pred1_log
        predicted_views = np.expm1(avg_pred_log)
       
        # Cap to reasonable range (more lenient)
        max_views_in_training = self.df['views'].quantile(0.99) * 2.5
        predicted_views = np.clip(predicted_views, 0, max_views_in_training)
       
        # Engagement prediction (total engagement)
        eng_model_info = self.models.get('engagement_prediction')
        if eng_model_info:
            feature_dict_eng = {
                'log_views': np.log1p(predicted_views),
                'log_impressions': np.log1p(video_features.get('impressions', 0)),
                'impressions_ctr': video_features.get('impressions_ctr', 0),
                'duration_minutes': video_features.get('duration_minutes', 0),
                'day_of_week': video_features.get('day_of_week', 0),
                'is_weekend': video_features.get('is_weekend', 0),
                'content_type': video_features.get('content_type', 'other')
            }
           
            X_eng = self._encode_features(feature_dict_eng, eng_model_info['feature_names'])
            X_eng_scaled = eng_model_info['scaler'].transform([X_eng])
           
            # Ensemble prediction
            models_eng = eng_model_info['ensemble']
            e_pred1_log = models_eng[0].predict(X_eng_scaled)[0]
            e_pred2_log = models_eng[1].predict(X_eng_scaled)[0]
            e_pred3_log = models_eng[2].predict(X_eng_scaled)[0]
           
            avg_eng_log = 0.5 * e_pred3_log + 0.3 * e_pred2_log + 0.2 * e_pred1_log
            predicted_total_engagement = np.expm1(avg_eng_log)
            predicted_total_engagement = max(0, predicted_total_engagement)
           
            # Convert to engagement rate
            predicted_engagement_rate = predicted_total_engagement / max(predicted_views, 1)
            predicted_engagement_rate = min(predicted_engagement_rate, 0.5)  # Cap at 50%
        else:
            predicted_engagement_rate = self.df['engagement_rate'].median()
            predicted_total_engagement = predicted_views * predicted_engagement_rate
       
        return {
            'predicted_views': int(predicted_views),
            'predicted_engagement_rate': float(predicted_engagement_rate),
            'predicted_total_engagement': int(predicted_total_engagement),
            'predicted_likes': int(predicted_total_engagement * 0.6),
            'predicted_comments': int(predicted_total_engagement * 0.3),
            'predicted_shares': int(predicted_total_engagement * 0.1)
        }
   
    def _encode_features(self, feature_dict, expected_features):
        """Encode features to match training format"""
        encoded = []
       
        for feature in expected_features:
            if feature in feature_dict:
                encoded.append(feature_dict[feature])
            elif feature.startswith('content_type_'):
                content_type = feature_dict.get('content_type', 'other')
                expected_type = feature.replace('content_type_', '')
                encoded.append(1 if content_type == expected_type else 0)
            else:
                encoded.append(0)
       
        return np.array(encoded)
   
    def save_models(self, filename_prefix='youtube_ml_model'):
        """Save trained models to disk"""
        if not self.models:
            print("❌ No models to save!")
            return False
       
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{filename_prefix}_{timestamp}.pkl"
           
            save_data = {
                'models': self.models,
                'feature_engineering_params': {
                    'content_types': self.df['content_type'].unique().tolist()
                },
                'training_date': datetime.now().isoformat(),
                'dataset_size': len(self.df)
            }
           
            joblib.dump(save_data, filename)
            print(f"✅ Models saved to: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            return False
   
    def load_models(self, filename):
        """Load trained models from disk"""
        try:
            loaded_data = joblib.load(filename)
            self.models = loaded_data['models']
            print(f"✅ Models loaded from: {filename}")
            print(f"   Training date: {loaded_data['training_date']}")
            print(f"   Dataset size: {loaded_data['dataset_size']}")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
   
    def generate_insights(self):
        """Generate actionable insights from the data"""
        print(f"\n{'='*70}")
        print("ACTIONABLE INSIGHTS")
        print(f"{'='*70}")
       
        # 1. Best performing content types
        print("\n1️⃣  CONTENT TYPE PERFORMANCE:")
        content_perf = self.df.groupby('content_type').agg({
            'views': ['mean', 'median', 'count'],
            'engagement_rate': ['mean', 'median']
        }).round(2)
       
        # Flatten column names
        content_perf.columns = ['views_mean', 'views_median', 'count', 'eng_mean', 'eng_median']
        content_perf = content_perf.sort_values('views_median', ascending=False)
       
        # Format for display
        for col in ['views_mean', 'views_median']:
            content_perf[col] = content_perf[col].apply(lambda x: f"{int(x):,}")
        for col in ['eng_mean', 'eng_median']:
            content_perf[col] = content_perf[col].apply(lambda x: f"{x:.4f}")
       
        print(content_perf.to_string())
       
        # 2. Best days to publish
        print("\n\n2️⃣  BEST DAYS TO PUBLISH:")
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_perf = self.df.groupby('day_of_week').agg({
            'views': 'median',
            'engagement_rate': 'median',
            'video_title': 'count'
        })
        day_perf.columns = ['median_views', 'median_engagement', 'count']
        day_perf['day_name'] = [day_names[i] for i in day_perf.index]
        day_perf = day_perf.sort_values('median_views', ascending=False)
       
        # Format
        day_perf['median_views'] = day_perf['median_views'].apply(lambda x: f"{int(x):,}")
        day_perf['median_engagement'] = day_perf['median_engagement'].apply(lambda x: f"{x:.4f}")
       
        print(day_perf[['day_name', 'median_views', 'median_engagement', 'count']].to_string())
       
        # 3. Optimal video duration
        print("\n\n3️⃣  OPTIMAL VIDEO DURATION:")
        self.df['duration_category'] = pd.cut(
            self.df['duration_minutes'],
            bins=[0, 3, 5, 10, float('inf')],
            labels=['Short (<3min)', 'Medium (3-5min)', 'Long (5-10min)', 'Very Long (>10min)']
        )
        duration_perf = self.df.groupby('duration_category').agg({
            'views': 'median',
            'engagement_rate': 'median',
            'video_title': 'count'
        })
        duration_perf.columns = ['Median Views', 'Median Engagement', 'Count']
       
        # Format
        duration_perf['Median Views'] = duration_perf['Median Views'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
        duration_perf['Median Engagement'] = duration_perf['Median Engagement'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
       
        print(duration_perf.to_string())
       
        # 4. CTR impact
        print("\n\n4️⃣  CTR IMPACT ON VIEWS:")
        self.df['ctr_category'] = pd.qcut(
            self.df['impressions_ctr'],
            q=4,
            labels=['Low CTR', 'Medium-Low CTR', 'Medium-High CTR', 'High CTR'],
            duplicates='drop'
        )
        ctr_perf = self.df.groupby('ctr_category').agg({
            'views': 'median',
            'impressions_ctr': 'mean',
            'video_title': 'count'
        })
        ctr_perf.columns = ['Median Views', 'Avg CTR (%)', 'Count']
       
        # Format
        ctr_perf['Median Views'] = ctr_perf['Median Views'].apply(lambda x: f"{int(x):,}")
        ctr_perf['Avg CTR (%)'] = ctr_perf['Avg CTR (%)'].apply(lambda x: f"{x:.2f}%")
       
        print(ctr_perf.to_string())
       
        # 5. Top performing videos
        print("\n\n5️⃣  TOP 5 PERFORMING VIDEOS:")
        top_videos = self.df.nlargest(5, 'views')[
            ['video_title', 'views', 'engagement_rate', 'impressions_ctr', 'content_type']
        ].copy()
        top_videos['views'] = top_videos['views'].apply(lambda x: f"{x:,.0f}")
        top_videos['engagement_rate'] = top_videos['engagement_rate'].apply(lambda x: f"{x:.4f}")
        top_videos['impressions_ctr'] = top_videos['impressions_ctr'].apply(lambda x: f"{x:.2f}%")
        print(top_videos.to_string(index=False))
       
        # 6. Key recommendations
        print(f"\n\n6️⃣  KEY RECOMMENDATIONS:")
       
        # Best content type
        content_perf_raw = self.df.groupby('content_type')['views'].median().sort_values(ascending=False)
        best_content = content_perf_raw.index[0]
        print(f"   • Focus on '{best_content}' content - highest median views ({content_perf_raw.iloc[0]:,.0f})")
       
        # Best day
        day_perf_raw = self.df.groupby('day_of_week')['views'].median().sort_values(ascending=False)
        best_day_idx = day_perf_raw.index[0]
        best_day = day_names[best_day_idx]
        print(f"   • Publish on {best_day}s for maximum views ({day_perf_raw.iloc[0]:,.0f} median)")
       
        # Duration sweet spot
        duration_perf_raw = self.df.groupby('duration_category')['views'].median().sort_values(ascending=False)
        best_duration = duration_perf_raw.index[0]
        print(f"   • Optimal duration: {best_duration} ({duration_perf_raw.iloc[0]:,.0f} median views)")
       
        # CTR importance
        ctr_perf_raw = self.df.groupby('ctr_category')['views'].median()
        if 'High CTR' in ctr_perf_raw.index and 'Low CTR' in ctr_perf_raw.index:
            high_ctr_median = ctr_perf_raw['High CTR']
            low_ctr_median = ctr_perf_raw['Low CTR']
            if low_ctr_median > 0:
                ctr_multiplier = high_ctr_median / low_ctr_median
                print(f"   • Improve CTR - High CTR videos get {ctr_multiplier:.1f}x more views")
       
        # Engagement insights
        avg_engagement = self.df['engagement_rate'].mean()
        print(f"   • Average engagement rate: {avg_engagement:.4f} ({avg_engagement*100:.2f}%)")
        print(f"   • Target engagement: {avg_engagement*1.5:.4f} for top performance")
       
        print(f"\n{'='*70}")
   
    def run_full_pipeline(self):
        """Run complete ML pipeline"""
        print(f"\n{'#'*70}")
        print("YOUTUBE ML PIPELINE - PRODUCTION RUN")
        print(f"{'#'*70}")
       
        # Step 1: Load data
        if not self.load_and_prepare_data():
            return False
       
        # Step 2: Build models
        print("\n" + "="*70)
        print("BUILDING MODELS")
        print("="*70)
       
        view_holdout, view_cv = self.build_view_prediction_ensemble()
        eng_holdout, eng_cv = self.build_engagement_predictor()
       
        # Step 3: Print comprehensive report
        self.evaluator.print_report()
       
        # Step 4: Generate visualizations
        print("\n📊 Generating evaluation plots...")
        self.evaluator.plot_evaluation()
       
        # Step 5: Generate insights
        self.generate_insights()
       
        # Step 6: Example prediction
        print(f"\n{'='*70}")
        print("EXAMPLE PREDICTION")
        print(f"{'='*70}")
       
        example_video = {
            'impressions': 10000,
            'impressions_ctr': 5.0,
            'duration_minutes': 4.5,
            'day_of_week': 2,  # Wednesday
            'is_weekend': 0,
            'content_type': 'music_video'
        }
       
        print("\nInput features:")
        for key, val in example_video.items():
            print(f"  {key}: {val}")
       
        prediction = self.predict_new_video(example_video)
       
        if prediction:
            print("\nPredicted outcomes:")
            print(f"  Views: {prediction['predicted_views']:,}")
            print(f"  Total Engagement: {prediction['predicted_total_engagement']:,}")
            print(f"  Engagement Rate: {prediction['predicted_engagement_rate']:.4f} ({prediction['predicted_engagement_rate']*100:.2f}%)")
            print(f"  └─ Likes: {prediction['predicted_likes']:,}")
            print(f"  └─ Comments: {prediction['predicted_comments']:,}")
            print(f"  └─ Shares: {prediction['predicted_shares']:,}")
       
        # Step 7: Save models
        print(f"\n{'='*70}")
        self.save_models()
       
        print(f"\n{'#'*70}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'#'*70}\n")
       
        return True




# Main execution
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = YouTubeMLPipeline(db_params)
   
    # Run full pipeline
    success = pipeline.run_full_pipeline()
   
    if success:
        print("\n✅ All tasks completed successfully!")
        print("\nNext steps:")
        print("  1. Review the model evaluation metrics")
        print("  2. Analyze the insights for content strategy")
        print("  3. Use predict_new_video() for new content planning")
        print("  4. Retrain periodically with new data")
    else:
        print("\n❌ Pipeline failed. Check error messages above.")

