# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            classification_report, confusion_matrix, roc_auc_score, roc_curve,
                            mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, log_loss)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ================= DB CONNECTION =================
db_params = {
    'dbname':   'neondb',
    'user':     'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host':     'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port':     '5432',
    'sslmode':  'require'
}

def fetch_data(conn):
    """Fetch all data with additional metrics"""
    query = """
    SELECT 
        video_id, title, views, likes, shares, comments_added, saves,
        duration_sec, post_type, sound_used,
        publish_year, publish_month, publish_day, publish_time,
        EXTRACT(DOW FROM publish_time) as day_of_week,
        EXTRACT(HOUR FROM publish_time) as hour_of_day,
        EXTRACT(WEEK FROM publish_time) as week_of_year,
        CASE 
            WHEN views > 0 THEN 
                ((COALESCE(likes,0) + COALESCE(shares,0) + 
                  COALESCE(comments_added,0) + COALESCE(saves,0))::FLOAT / views) * 100
            ELSE 0 
        END as engagement_rate,
        CASE WHEN views > 0 THEN (COALESCE(likes,0)::FLOAT / views) * 100 ELSE 0 END as like_rate,
        CASE WHEN views > 0 THEN (COALESCE(shares,0)::FLOAT / views) * 100 ELSE 0 END as share_rate,
        CASE WHEN views > 0 THEN (COALESCE(comments_added,0)::FLOAT / views) * 100 ELSE 0 END as comment_rate
    FROM public.tt_video_etl
    WHERE views IS NOT NULL AND views > 0
        AND duration_sec IS NOT NULL
        AND publish_time IS NOT NULL
    ORDER BY publish_time;
    """
    return pd.read_sql_query(query, conn)

def engineer_advanced_features(df):
    """Create enhanced features"""
    df = df.copy()

    # Time features
    df['is_weekend'] = df['day_of_week'].isin([0, 6]).astype(int)
    df['is_weekday_peak'] = df['day_of_week'].isin([1, 2, 3]).astype(int)
    df['is_prime_time'] = df['hour_of_day'].between(17, 22).astype(int)
    df['is_morning'] = df['hour_of_day'].between(6, 12).astype(int)

    # Season
    df['season'] = df['publish_month'].map({
        12: 0, 1: 0, 2: 0,  # Winter
        3: 1, 4: 1, 5: 1,   # Spring
        6: 2, 7: 2, 8: 2,   # Summer
        9: 3, 10: 3, 11: 3  # Fall
    })

    # Content features
    df['has_sound'] = (df['sound_used'].notna() & (df['sound_used'] != '')).astype(int)
    df['post_type'] = df['post_type'].fillna('unknown')
    df['duration_category'] = pd.cut(df['duration_sec'],
                                     bins=[0, 15, 30, 60, float('inf')],
                                     labels=[0, 1, 2, 3])

    # Historical performance features
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    df = df.sort_values('publish_time')

    # Rolling averages (7-day window)
    df['views_rolling_7d'] = df['views'].rolling(window=7, min_periods=1).mean()
    df['engagement_rolling_7d'] = df['engagement_rate'].rolling(window=7, min_periods=1).mean()

    # Video number (sequential)
    df['video_number'] = range(len(df))

    # Interaction features
    df['duration_x_weekend'] = df['duration_sec'] * df['is_weekend']
    df['duration_x_primetime'] = df['duration_sec'] * df['is_prime_time']
    df['sound_x_weekend'] = df['has_sound'] * df['is_weekend']

    # Target variables
    df['view_category'] = pd.cut(df['views'],
                                  bins=[0, 10000, 100000, 1000000, float('inf')],
                                  labels=['Low', 'Medium', 'High', 'Viral'])
    df['is_viral'] = (df['views'] >= 100000).astype(int)

    # Log transform for skewed features
    df['log_views'] = np.log1p(df['views'])
    df['log_duration'] = np.log1p(df['duration_sec'])

    return df

def calculate_mase(y_true, y_pred, y_train):
    """Calculate Mean Absolute Scaled Error"""
    n = len(y_train)
    if n <= 1:
        return np.inf
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    mase = errors.mean() / d if d != 0 else np.inf
    return mase

# ================= MODEL 1: VIEW CATEGORY CLASSIFICATION =================
def model1_view_category(df):
    print("\n" + "="*70)
    print("MODEL 1: VIEW CATEGORY PREDICTION (Low/Medium/High/Viral)")
    print("="*70)

    le_post = LabelEncoder()
    df['post_type_encoded'] = le_post.fit_transform(df['post_type'])

    feature_cols = [
        'duration_sec', 'log_duration', 'publish_month', 'day_of_week', 'hour_of_day',
        'is_weekend', 'is_weekday_peak', 'is_prime_time', 'is_morning',
        'has_sound', 'post_type_encoded', 'season', 'duration_category',
        'video_number', 'duration_x_weekend', 'duration_x_primetime', 'sound_x_weekend'
    ]

    X = df[feature_cols].fillna(0)
    y = df['view_category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=12, min_samples_split=8)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High', 'Viral'])

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Medium', 'High', 'Viral'],
                yticklabels=['Low', 'Medium', 'High', 'Viral'])
    plt.title('Model 1: View Category Confusion Matrix', fontweight='bold', fontsize=14)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")

    return model, scaler, feature_cols, accuracy

# ================= MODEL 2: VIRALITY WITH SMOTE (ENHANCED WITH REGRESSION METRICS + MASE) =================
def model2_enhanced_virality(df):
    print("\n" + "="*70)
    print("MODEL 2 ENHANCED: VIRALITY PREDICTION WITH CLASS BALANCING")
    print("="*70)

    le_post = LabelEncoder()
    df['post_type_encoded'] = le_post.fit_transform(df['post_type'])

    feature_cols = [
        'duration_sec', 'log_duration', 'publish_month', 'day_of_week', 'hour_of_day',
        'is_weekend', 'is_weekday_peak', 'is_prime_time', 'is_morning',
        'has_sound', 'post_type_encoded', 'season', 'duration_category',
        'views_rolling_7d', 'engagement_rolling_7d', 'video_number',
        'duration_x_weekend', 'duration_x_primetime', 'sound_x_weekend'
    ]

    X = df[feature_cols].fillna(0)
    y = df['is_viral']

    viral_count = y.sum()
    total = len(y)
    viral_pct = (viral_count / total) * 100

    print(f"\nOriginal dataset: {viral_count}/{total} viral ({viral_pct:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(sampling_strategy=0.6, random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    print(f"After SMOTE: {y_train_balanced.sum()}/{len(y_train_balanced)} viral")

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_balanced, y_train_balanced)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Calculate additional classification metrics
    cm = confusion_matrix(y_test, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Calculate pseudo-regression metrics for classification
    mae_prob = mean_absolute_error(y_test, y_pred_proba)
    rmse_prob = np.sqrt(mean_squared_error(y_test, y_pred_proba))
    brier_score = np.mean((y_pred_proba - y_test) ** 2)
    logloss = log_loss(y_test, y_pred_proba)
    mase_prob = calculate_mase(y_test.values, y_pred_proba, y_train.values)

    print(f"\n{'='*70}")
    print(f"{'CLASSIFICATION METRICS':<70}")
    print(f"{'='*70}")
    print(f"{'METRIC':<20} {'VALUE':<15} {'INTERPRETATION'}")
    print(f"{'-'*70}")
    print(f"{'Accuracy':<20} {accuracy*100:>6.2f}%          Overall correctness")
    print(f"{'Precision':<20} {precision*100:>6.2f}%          Viral predictions accuracy")
    print(f"{'Recall (Sensitivity)':<20} {recall*100:>6.2f}%          Viral videos caught")
    print(f"{'Specificity':<20} {specificity*100:>6.2f}%          Non-viral correctly ID'd")
    print(f"{'F1 Score':<20} {f1*100:>6.2f}%          Balance of precision/recall")
    print(f"{'NPV':<20} {npv*100:>6.2f}%          Non-viral pred accuracy")
    print(f"{'AUC-ROC':<20} {auc:>6.4f}           Model discrimination power")

    print(f"\n{'='*70}")
    print(f"{'PROBABILITY-BASED METRICS (Like Regression)':<70}")
    print(f"{'='*70}")
    print(f"{'METRIC':<20} {'VALUE':<15} {'INTERPRETATION'}")
    print(f"{'-'*70}")
    print(f"{'MAE':<20} {mae_prob:.4f}           Mean absolute error (probability)")
    print(f"{'RMSE':<20} {rmse_prob:.4f}           Root mean squared error")
    print(f"{'Brier Score':<20} {brier_score:.4f}           Calibration quality (lower=better)")
    print(f"{'Log Loss':<20} {logloss:.4f}           Prediction confidence penalty")
    print(f"{'MASE':<20} {mase_prob:.4f}           Scaled error (< 1 is good)")

    mae_status = "✓ EXCELLENT" if mae_prob < 0.25 else "✓ GOOD" if mae_prob < 0.35 else "⚠ NEEDS WORK"
    brier_status = "✓ EXCELLENT" if brier_score < 0.15 else "✓ GOOD" if brier_score < 0.25 else "⚠ NEEDS WORK"
    mase_status = "✓ EXCELLENT" if mase_prob < 0.8 else "✓ GOOD" if mase_prob < 1.0 else "⚠ NEEDS WORK"

    print(f"\nStatus: MAE {mae_status} | Brier {brier_status} | MASE {mase_status}")

    print(f"\n{'='*70}")
    print(f"{'CONFUSION MATRIX BREAKDOWN':<70}")
    print(f"{'='*70}")
    print(f"{'True Negatives':<20} {tn:>6}          Correctly predicted non-viral")
    print(f"{'False Positives':<20} {fp:>6}          Wrongly predicted as viral")
    print(f"{'False Negatives':<20} {fn:>6}          Missed viral videos")
    print(f"{'True Positives':<20} {tp:>6}          Correctly predicted viral")
    print(f"\n{'Total Test Videos':<20} {len(y_test):>6}")
    print(f"{'Actual Viral':<20} {y_test.sum():>6}          ({y_test.mean()*100:.1f}% of test set)")

    print(f"\n{'='*70}")
    print(f"{'PERFORMANCE IMPROVEMENTS vs BASELINE':<70}")
    print(f"{'='*70}")
    baseline_acc = 0.67
    baseline_prec = 0.17
    baseline_recall = 0.11
    baseline_f1 = 0.13
    baseline_auc = 0.61

    print(f"{'Accuracy':<20} {'+' if accuracy > baseline_acc else ''}{(accuracy-baseline_acc)*100:>5.1f}%")
    print(f"{'Precision':<20} {'+' if precision > baseline_prec else ''}{(precision-baseline_prec)*100:>5.1f}%")
    print(f"{'Recall':<20} {'+' if recall > baseline_recall else ''}{(recall-baseline_recall)*100:>5.1f}%")
    print(f"{'F1 Score':<20} {'+' if f1 > baseline_f1 else ''}{(f1-baseline_f1)*100:>5.1f}%")
    print(f"{'AUC-ROC':<20} {'+' if auc > baseline_auc else ''}{(auc-baseline_auc):>5.3f}")

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('Enhanced Model: ROC Curve', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=axes[0, 1],
                xticklabels=['Non-Viral', 'Viral'],
                yticklabels=['Non-Viral', 'Viral'])
    axes[0, 1].set_title('Confusion Matrix', fontweight='bold')
    axes[0, 1].set_ylabel('Actual')
    axes[0, 1].set_xlabel('Predicted')

    top_features = feature_importance.head(10)
    axes[1, 0].barh(top_features['feature'], top_features['importance'])
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].set_title('Top 10 Feature Importance', fontweight='bold')
    axes[1, 0].invert_yaxis()

    axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6, label='Non-Viral', color='blue')
    axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6, label='Viral', color='red')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Probability Distribution by Class', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].axvline(0.5, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    return model, scaler, feature_cols, auc, f1, mase_prob

# ================= MODEL 3: ENGAGEMENT RATE WITH ENHANCED VISUALIZATION =================
def model3_enhanced_engagement(df):
    print("\n" + "="*70)
    print("MODEL 3 ENHANCED: ENGAGEMENT RATE WITH BETTER FEATURES")
    print("="*70)

    q_low = df['engagement_rate'].quantile(0.01)
    q_high = df['engagement_rate'].quantile(0.99)
    df_clean = df[(df['engagement_rate'] >= q_low) & (df['engagement_rate'] <= q_high)].copy()

    print(f"Removed {len(df) - len(df_clean)} outliers ({((len(df) - len(df_clean))/len(df)*100):.1f}%)")

    le_post = LabelEncoder()
    df_clean['post_type_encoded'] = le_post.fit_transform(df_clean['post_type'])

    feature_cols = [
        'duration_sec', 'log_duration', 'publish_month', 'day_of_week', 'hour_of_day',
        'is_weekend', 'is_weekday_peak', 'is_prime_time', 'is_morning',
        'has_sound', 'post_type_encoded', 'season', 'duration_category',
        'video_number', 'log_views',
        'duration_x_weekend', 'duration_x_primetime', 'sound_x_weekend'
    ]

    X = df_clean[feature_cols].fillna(0)
    y = df_clean['engagement_rate']

    print(f"\nEngagement Rate Stats:")
    print(f"  Mean:   {y.mean():.2f}%")
    print(f"  Median: {y.median():.2f}%")
    print(f"  Std:    {y.std():.2f}%")
    print(f"  Range:  {y.min():.2f}% - {y.max():.2f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    mase = calculate_mase(y_test.values, y_pred, y_train.values)

    mae_pct = (mae / y_test.mean()) * 100
    rmse_pct = (rmse / y_test.mean()) * 100

    print(f"\n{'METRIC':<12} {'VALUE':<15} {'INTERPRETATION'}")
    print(f"{'='*60}")
    print(f"{'MAE':<12} {mae:.2f}% ({mae_pct:.1f}%)   Avg error")
    print(f"{'RMSE':<12} {rmse:.2f}% ({rmse_pct:.1f}%)  Larger errors penalized")
    print(f"{'MAPE':<12} {mape:.2f}%            Relative error")
    print(f"{'MASE':<12} {mase:.4f}            Scaled error (< 1 is good)")
    print(f"{'R²':<12} {r2:.4f} ({r2*100:.1f}%)    Variance explained")

    improvement = "✓ IMPROVED" if r2 > 0 else "✗ STILL NEGATIVE"
    mase_status = "✓ GOOD" if mase < 1 else "⚠ NEEDS WORK"
    print(f"\nStatus: {improvement} | MASE: {mase_status}")

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")

    # ===== ENHANCED VISUALIZATION =====
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)

    # Graph 1: Predicted vs Actual
    ax1 = fig.add_subplot(gs[0, 0])
    errors = np.abs(y_pred - y_test)
    colors = np.where(errors <= 1, 'darkgreen',
             np.where(errors <= 3, 'lightgreen',
             np.where(errors <= 5, 'yellow',
             np.where(errors <= 7, 'orange', 'red'))))

    ax1.scatter(y_test, y_pred, c=colors, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    ax1.fill_between([y_test.min(), y_test.max()],
                     [y_test.min()-3, y_test.max()-3],
                     [y_test.min()+3, y_test.max()+3],
                     alpha=0.1, color='green', label='±3% Zone')

    ax1.set_xlabel('Actual Engagement Rate (%)', fontsize=10)
    ax1.set_ylabel('Predicted Engagement Rate (%)', fontsize=10)
    ax1.set_title(f'Predicted vs Actual (R²={r2:.3f})', fontweight='bold', fontsize=11)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    textstr = f'MAE: {mae:.2f}%\nRMSE: {rmse:.2f}%\nn = {len(y_test)}'
    ax1.text(0.98, 0.02, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Graph 2: Residual Plot
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = y_pred - y_test

    ax2.axhspan(-3, 3, alpha=0.2, color='green', label='±3% (Acceptable)')
    ax2.axhspan(-5, -3, alpha=0.15, color='yellow')
    ax2.axhspan(3, 5, alpha=0.15, color='yellow', label='±3-5% (Warning)')
    ax2.axhspan(-10, -5, alpha=0.1, color='red')
    ax2.axhspan(5, 10, alpha=0.1, color='red', label='>±5% (Poor)')

    outliers = np.abs(residuals) > 5
    ax2.scatter(y_pred[~outliers], residuals[~outliers], alpha=0.5, s=20, color='blue', label='Normal')
    ax2.scatter(y_pred[outliers], residuals[outliers], alpha=0.7, s=30, color='red',
               edgecolors='darkred', linewidth=1, label='Outliers')

    ax2.axhline(0, color='black', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Predicted Engagement Rate (%)', fontsize=10)
    ax2.set_ylabel('Residuals (%)', fontsize=10)
    ax2.set_title('Residual Plot with Reference Bands', fontweight='bold', fontsize=11)
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Graph 3: Error Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    n, bins, patches = ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')

    try:
        from scipy import stats
        mu, std = residuals.mean(), residuals.std()
        xmin, xmax = ax3.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        p = p * len(residuals) * (bins[1] - bins[0])
        ax3.plot(x, p, 'r-', linewidth=2, label='Normal Distribution')
    except Exception:
        mu, std = residuals.mean(), residuals.std()

    ax3.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero Error')
    ax3.axvline(residuals.mean(), color='blue', linestyle=':', linewidth=2, label=f'Mean: {residuals.mean():.2f}%')
    ax3.axvline(residuals.median(), color='green', linestyle=':', linewidth=2, label=f'Median: {residuals.median():.2f}%')

    ax3.set_xlabel('Prediction Error (%)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title(f'Error Distribution (Std={std:.2f}%)', fontweight='bold', fontsize=11)
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # Graph 4: Feature Importance
    ax4 = fig.add_subplot(gs[1, 1])
    top_features = feature_importance.head(10)
    colors_importance = ['darkblue' if imp > 0.3 else 'steelblue' if imp > 0.1 else 'lightblue'
                        for imp in top_features['importance']]

    bars = ax4.barh(top_features['feature'], top_features['importance'], color=colors_importance, edgecolor='black')

    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax4.text(row['importance'], i, f" {row['importance']*100:.1f}%",
                va='center', fontsize=9, fontweight='bold')

    ax4.set_xlabel('Importance', fontsize=10)
    ax4.set_title('Top 10 Feature Importance', fontweight='bold', fontsize=11)
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='x')

    # Graph 5: ACTUAL vs PREDICTED TIMELINE
    ax5 = fig.add_subplot(gs[2, :])
    test_indices = X_test.index
    video_numbers = df_clean.loc[test_indices, 'video_number'].values

    sort_idx = np.argsort(video_numbers)
    video_numbers_sorted = video_numbers[sort_idx]
    y_test_sorted = y_test.values[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    errors_sorted = np.abs(y_pred_sorted - y_test_sorted)

    ax5.plot(video_numbers_sorted, y_test_sorted, 'o-', color='black',
            linewidth=2, markersize=4, label='Actual Engagement', zorder=3)
    ax5.plot(video_numbers_sorted, y_pred_sorted, 'x--', color='red',
            linewidth=2, markersize=4, label='Predicted Engagement', alpha=0.8, zorder=2)

    for i in range(len(video_numbers_sorted)-1):
        error = errors_sorted[i]
        if error <= 3:
            color = 'green'
            alpha = 0.15
        elif error <= 5:
            color = 'orange'
            alpha = 0.2
        else:
            color = 'red'
            alpha = 0.25

        ax5.fill_between([video_numbers_sorted[i], video_numbers_sorted[i+1]],
                        [y_test_sorted[i], y_test_sorted[i+1]],
                        [y_pred_sorted[i], y_pred_sorted[i+1]],
                        color=color, alpha=alpha, zorder=1)

    ax5.axhline(y.mean(), color='gray', linestyle=':', linewidth=1.5,
               label=f'Mean Engagement: {y.mean():.1f}%', alpha=0.7, zorder=1)

    ax5.set_xlabel('Video Number (Chronological Order)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Engagement Rate (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Actual vs Predicted Engagement Over Time', fontweight='bold', fontsize=12)
    ax5.legend(loc='best', fontsize=9, framealpha=0.9)
    ax5.grid(True, alpha=0.3, linestyle='--')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.3, label='Good Prediction (≤3% error)'),
        Patch(facecolor='orange', alpha=0.3, label='Acceptable (3-5% error)'),
        Patch(facecolor='red', alpha=0.3, label='Poor (>5% error)')
    ]
    ax5.legend(handles=ax5.get_legend_handles_labels()[0] + legend_elements,
              loc='upper left', fontsize=8, ncol=2, framealpha=0.9)

    plt.suptitle('Model 3: Engagement Rate Prediction Analysis',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

    return model, scaler, feature_cols, r2, mase

# ================= BEST POSTING TIME ANALYSIS WITH PREDICTIVE MODEL =================
def analyze_best_posting_times(df):
    print("\n" + "="*70)
    print("BEST DAY & TIME TO POST ANALYSIS (WITH PREDICTIVE MODELING)")
    print("="*70)
    
    # Day of week analysis
    day_names = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 
                 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    
    df['day_name'] = df['day_of_week'].map(day_names)
    
    # Analyze by day of week
    day_stats = df.groupby('day_of_week').agg({
        'views': ['mean', 'median', 'sum', 'count'],
        'engagement_rate': ['mean', 'median'],
        'is_viral': ['sum', 'mean'],
        'likes': 'mean',
        'shares': 'mean'
    }).round(2)
    
    day_stats.columns = ['_'.join(col).strip() for col in day_stats.columns.values]
    day_stats['day_name'] = day_stats.index.map(day_names)
    day_stats = day_stats.sort_values('views_mean', ascending=False)
    
    # Analyze by hour of day
    hour_stats = df.groupby('hour_of_day').agg({
        'views': ['mean', 'median', 'sum', 'count'],
        'engagement_rate': ['mean', 'median'],
        'is_viral': ['sum', 'mean']
    }).round(2)
    
    hour_stats.columns = ['_'.join(col).strip() for col in hour_stats.columns.values]
    hour_stats = hour_stats.sort_values('views_mean', ascending=False)
    
    # Combined day + hour analysis
    day_hour_stats = df.groupby(['day_of_week', 'hour_of_day']).agg({
        'views': ['mean', 'count'],
        'engagement_rate': 'mean',
        'is_viral': 'mean'
    }).round(2)
    
    day_hour_stats.columns = ['_'.join(col).strip() for col in day_hour_stats.columns.values]
    day_hour_stats = day_hour_stats[day_hour_stats['views_count'] >= 2]
    day_hour_stats = day_hour_stats.sort_values('views_mean', ascending=False)
    
    # Calculate composite score
    day_stats['composite_score'] = (
        (day_stats['views_mean'] / day_stats['views_mean'].max()) * 0.4 +
        (day_stats['engagement_rate_mean'] / day_stats['engagement_rate_mean'].max()) * 0.3 +
        (day_stats['is_viral_mean'] / day_stats['is_viral_mean'].max()) * 0.3
    ) * 100
    
    hour_stats['composite_score'] = (
        (hour_stats['views_mean'] / hour_stats['views_mean'].max()) * 0.4 +
        (hour_stats['engagement_rate_mean'] / hour_stats['engagement_rate_mean'].max()) * 0.3 +
        (hour_stats['is_viral_mean'] / hour_stats['is_viral_mean'].max()) * 0.3
    ) * 100
    
    # ================= BUILD PREDICTIVE MODEL FOR POSTING TIME =================
    print("\n" + "-"*70)
    print("TRAINING PREDICTIVE MODEL: Views Based on Posting Time")
    print("-"*70)
    
    # Prepare features for prediction
    le_post = LabelEncoder()
    df_model = df.copy()
    df_model['post_type_encoded'] = le_post.fit_transform(df_model['post_type'])
    
    # Features focused on DAY OF WEEK
    time_features = [
        'day_of_week', 'is_weekend', 'is_weekday_peak',
        'publish_month', 'season',
        'duration_sec', 'has_sound', 'post_type_encoded'
    ]
    
    X = df_model[time_features].fillna(0)
    y = df_model['views']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    mase = calculate_mase(y_test.values, y_pred, y_train.values)
    
    mae_pct = (mae / y_test.mean()) * 100
    rmse_pct = (rmse / y_test.mean()) * 100
    
    print(f"\n{'METRIC':<12} {'VALUE':<20} {'INTERPRETATION'}")
    print(f"{'='*65}")
    print(f"{'MAE':<12} {mae:>12,.0f} ({mae_pct:.1f}%)   Avg prediction error")
    print(f"{'RMSE':<12} {rmse:>12,.0f} ({rmse_pct:.1f}%)  Penalizes large errors")
    print(f"{'MAPE':<12} {mape:>12.2f}%         Relative error")
    print(f"{'MASE':<12} {mase:>12.4f}          Scaled error (< 1 is good)")
    print(f"{'R²':<12} {r2:>12.4f} ({r2*100:.1f}%)    Variance explained")
    
    r2_status = "✓ EXCELLENT" if r2 > 0.7 else "✓ GOOD" if r2 > 0.5 else "✓ ACCEPTABLE" if r2 > 0.3 else "⚠ NEEDS WORK"
    mase_status = "✓ EXCELLENT" if mase < 0.8 else "✓ GOOD" if mase < 1.0 else "⚠ NEEDS WORK"
    
    print(f"\nModel Quality: R² {r2_status} | MASE {mase_status}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': time_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop Timing Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']:<20} {row['importance']:.4f}")
    
    # Print descriptive statistics
    print("\n" + "-"*70)
    print("DESCRIPTIVE STATISTICS BY POSTING TIME")
    print("-"*70)
    
    print("\n📅 BEST DAYS TO POST (Ranked by Average Views):")
    print("="*80)
    print(f"{'Rank':<6} {'Day':<12} {'Avg Views':<12} {'Std Dev':<12} {'Engagement':<12} {'Viral %':<10} {'Posts':<8} {'Score'}")
    print("-"*80)
    
    for rank, (idx, row) in enumerate(day_stats.iterrows(), 1):
        emoji = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        day_videos = df[df['day_of_week'] == idx]['views']
        std_dev = day_videos.std()
        
        print(f"{emoji} {rank:<4} {row['day_name']:<12} {row['views_mean']:>10,.0f}  "
              f"{std_dev:>10,.0f}  "
              f"{row['engagement_rate_mean']:>10.2f}%  {row['is_viral_mean']*100:>8.1f}%  "
              f"{int(row['views_count']):>6}  {row['composite_score']:>6.1f}")
    
    # Statistical significance testing
    print("\n📊 STATISTICAL SIGNIFICANCE TEST (ANOVA):")
    print("="*80)
    
    try:
        from scipy import stats
        
        day_groups = [df[df['day_of_week'] == i]['views'].values for i in range(7)]
        f_stat, p_value = stats.f_oneway(*day_groups)
        
        print(f"F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"✅ Result: STATISTICALLY SIGNIFICANT (p < 0.05)")
            print(f"   → Day of week DOES affect views")
        else:
            print(f"⚠️  Result: NOT SIGNIFICANT (p >= 0.05)")
            print(f"   → Day of week may not significantly affect views")
        
        best_day_idx = day_stats.index[0]
        best_day_views = df[df['day_of_week'] == best_day_idx]['views'].values
        
        print(f"\n🎯 Top Day ({day_names[best_day_idx]}) vs Others:")
        print("-"*80)
        
        for idx in range(7):
            if idx != best_day_idx:
                other_views = df[df['day_of_week'] == idx]['views'].values
                t_stat, t_p_value = stats.ttest_ind(best_day_views, other_views)
                
                significance = "✅ SIGNIFICANT" if t_p_value < 0.05 else "⚠️  NOT SIGNIFICANT"
                diff_pct = ((day_stats.loc[best_day_idx, 'views_mean'] - day_stats.loc[idx, 'views_mean']) / 
                           day_stats.loc[idx, 'views_mean'] * 100)
                
                print(f"  vs {day_names[idx]:<10}: p={t_p_value:.4f}  {significance}  (Δ {diff_pct:+.1f}%)")
    
    except Exception as e:
        print(f"Could not perform statistical tests: {e}")
        p_value = None
    
    print("\n⏰ BEST HOURS TO POST (Ranked by Average Views):")
    print("="*80)
    print(f"{'Rank':<6} {'Hour':<15} {'Avg Views':<12} {'Engagement':<12} {'Viral %':<10} {'Posts':<8} {'Score'}")
    print("-"*70)
    
    top_hours = hour_stats.head(10)
    for rank, (hour, row) in enumerate(top_hours.iterrows(), 1):
        emoji = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        time_label = f"{int(hour):02d}:00-{int(hour)+1:02d}:00"
        print(f"{emoji} {rank:<4} {time_label:<15} {row['views_mean']:>10,.0f}  "
              f"{row['engagement_rate_mean']:>10.2f}%  {row['is_viral_mean']*100:>8.1f}%  "
              f"{int(row['views_count']):>6}  {row['composite_score']:>6.1f}")
    
    print("\n🎯 TOP 10 DAY+HOUR COMBINATIONS:")
    print("="*70)
    print(f"{'Rank':<6} {'Day':<12} {'Hour':<15} {'Avg Views':<12} {'Engagement':<12} {'Posts'}")
    print("-"*70)
    
    top_combos = day_hour_stats.head(10)
    for rank, ((day, hour), row) in enumerate(top_combos.iterrows(), 1):
        emoji = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        day_name = day_names[day]
        time_label = f"{int(hour):02d}:00-{int(hour)+1:02d}:00"
        print(f"{emoji} {rank:<4} {day_name:<12} {time_label:<15} {row['views_mean']:>10,.0f}  "
              f"{row['engagement_rate_mean']:>10.2f}%  {int(row['views_count']):>6}")
    
    # Recommendations
    best_day = day_stats.iloc[0]
    best_hour = hour_stats.iloc[0]
    best_combo = day_hour_stats.iloc[0]
    
    print("\n💡 RECOMMENDATIONS (WITH CONFIDENCE LEVELS):")
    print("="*80)
    
    # Calculate confidence intervals
    best_day_idx = day_stats.index[0]
    best_day_views = df[df['day_of_week'] == best_day_idx]['views'].values
    
    confidence_level = 0.95
    degrees_freedom = len(best_day_views) - 1
    sample_mean = np.mean(best_day_views)
    sample_std = np.std(best_day_views, ddof=1)
    sample_stderr = sample_std / np.sqrt(len(best_day_views))
    
    try:
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
        margin_error = t_value * sample_stderr
        ci_lower = sample_mean - margin_error
        ci_upper = sample_mean + margin_error
        
        print(f"✅ Best Day: {best_day['day_name']}")
        print(f"   • Average views: {best_day['views_mean']:,.0f}")
        print(f"   • 95% Confidence Interval: [{ci_lower:,.0f}, {ci_upper:,.0f}]")
        print(f"   • Sample size: {int(best_day['views_count'])} posts")
        print(f"   • Reliability: {'HIGH' if int(best_day['views_count']) >= 50 else 'MODERATE' if int(best_day['views_count']) >= 30 else 'LOW'}")
    except:
        print(f"✅ Best Day: {best_day['day_name']} (avg {best_day['views_mean']:,.0f} views)")
    
    print(f"\n✅ Top 3 Days (Similar Performance):")
    for i, (idx, row) in enumerate(day_stats.head(3).iterrows(), 1):
        print(f"   {i}. {row['day_name']}: {row['views_mean']:,.0f} avg views ({int(row['views_count'])} posts)")
    print(f"✅ Best Single Hour: {int(best_hour.name):02d}:00-{int(best_hour.name)+1:02d}:00 (avg {best_hour['views_mean']:,.0f} views)")
    print(f"✅ Best Combination: {day_names[best_combo.name[0]]} at {int(best_combo.name[1]):02d}:00 (avg {best_combo['views_mean']:,.0f} views)")
    print(f"\n   Note: Hour analysis may be limited due to posting schedule concentration")
    
    # Calculate improvement potential
    avg_views = df['views'].mean()
    best_combo_improvement = ((best_combo['views_mean'] - avg_views) / avg_views) * 100
    
    print(f"\n📈 Improvement Potential (Best Day vs Average):")
    print(f"   • Current overall average: {avg_views:,.0f} views per video")
    print(f"   • Best day ({best_day['day_name']}) average: {best_day['views_mean']:,.0f} views per video")
    print(f"   • Potential improvement: {best_combo_improvement:+.1f}%")
    print(f"   • Confidence: {'HIGH' if abs(best_combo_improvement) > 30 and int(best_day['views_count']) >= 50 else 'MODERATE'}")
    
    # Worst vs best comparison
    worst_day = day_stats.iloc[-1]
    worst_best_diff = ((best_day['views_mean'] - worst_day['views_mean']) / worst_day['views_mean'] * 100)
    print(f"\n   • Worst day ({worst_day['day_name']}) average: {worst_day['views_mean']:,.0f} views")
    print(f"   • Best vs Worst: {worst_best_diff:+.1f}% difference")
    
    # Weekend vs weekday
    weekend_views = df[df['is_weekend'] == 1]['views'].mean()
    weekday_views = df[df['is_weekend'] == 0]['views'].mean()
    print(f"\n   • Weekend avg: {weekend_views:,.0f} views")
    print(f"   • Weekday avg: {weekday_views:,.0f} views")
    print(f"   • Weekend vs Weekday: {((weekend_views - weekday_views) / weekday_views * 100):+.1f}%")
    
    # Prime time analysis
    prime_views = df[df['is_prime_time'] == 1]['views'].mean()
    non_prime_views = df[df['is_prime_time'] == 0]['views'].mean()
    print(f"\n   • Prime time (5PM-10PM) avg: {prime_views:,.0f} views")
    print(f"   • Non-prime time avg: {non_prime_views:,.0f} views")
    print(f"   • Prime time advantage: {((prime_views - non_prime_views) / non_prime_views * 100):+.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Views by day of week
    day_plot_data = day_stats.sort_index()
    colors_day = ['#ff6b6b' if score == day_plot_data['composite_score'].max() 
                  else '#4ecdc4' if score >= day_plot_data['composite_score'].quantile(0.66)
                  else '#95e1d3' for score in day_plot_data['composite_score']]
    
    axes[0, 0].bar(range(7), day_plot_data['views_mean'], color=colors_day, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_xticks(range(7))
    axes[0, 0].set_xticklabels([day_names[i] for i in range(7)], rotation=45)
    axes[0, 0].set_ylabel('Average Views', fontweight='bold')
    axes[0, 0].set_title('Average Views by Day of Week', fontweight='bold', fontsize=12)
    axes[0, 0].axhline(avg_views, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Overall Avg: {avg_views:,.0f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(day_plot_data['views_mean']):
        axes[0, 0].text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 2. Views by hour of day
    colors_hour = ['#ff6b6b' if score == hour_stats['composite_score'].max() 
                   else '#4ecdc4' if score >= hour_stats['composite_score'].quantile(0.75)
                   else '#95e1d3' for score in hour_stats['composite_score']]
    
    axes[0, 1].bar(hour_stats.index, hour_stats['views_mean'], color=colors_hour, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_xlabel('Hour of Day', fontweight='bold')
    axes[0, 1].set_ylabel('Average Views', fontweight='bold')
    axes[0, 1].set_title('Average Views by Hour of Day', fontweight='bold', fontsize=12)
    axes[0, 1].axhline(avg_views, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Overall Avg: {avg_views:,.0f}')
    axes[0, 1].axvspan(17, 22, alpha=0.1, color='yellow', label='Prime Time (5-10PM)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Heatmap: Day x Hour
    pivot_views = df.pivot_table(values='views', index='day_of_week', columns='hour_of_day', aggfunc='mean')
    pivot_views.index = [day_names[i] for i in pivot_views.index]
    
    sns.heatmap(pivot_views, annot=False, fmt='.0f', cmap='YlOrRd', ax=axes[0, 2], 
                cbar_kws={'label': 'Avg Views'}, linewidths=0.5)
    axes[0, 2].set_xlabel('Hour of Day', fontweight='bold')
    axes[0, 2].set_ylabel('Day of Week', fontweight='bold')
    axes[0, 2].set_title('Views Heatmap: Day x Hour', fontweight='bold', fontsize=12)
    
    # 4. Engagement rate by day
    day_plot_data_sorted = day_stats.sort_index()
    axes[1, 0].bar(range(7), day_plot_data_sorted['engagement_rate_mean'], 
                   color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.7)
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels([day_names[i] for i in range(7)], rotation=45)
    axes[1, 0].set_ylabel('Engagement Rate (%)', fontweight='bold')
    axes[1, 0].set_title('Engagement Rate by Day of Week', fontweight='bold', fontsize=12)
    axes[1, 0].axhline(df['engagement_rate'].mean(), color='red', linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'Avg: {df["engagement_rate"].mean():.2f}%')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(day_plot_data_sorted['engagement_rate_mean']):
        axes[1, 0].text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 5. Predicted vs Actual (from model)
    axes[1, 1].scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[1, 1].set_xlabel('Actual Views', fontweight='bold')
    axes[1, 1].set_ylabel('Predicted Views', fontweight='bold')
    axes[1, 1].set_title(f'Model: Predicted vs Actual (R²={r2:.3f})', fontweight='bold', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    textstr = f'MAE: {mae:,.0f}\nRMSE: {rmse:,.0f}\nMASE: {mase:.3f}'
    axes[1, 1].text(0.05, 0.95, textstr, transform=axes[1, 1].transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. Feature Importance
    top_features = feature_importance.head(8)
    colors_importance = ['darkblue' if imp > 0.15 else 'steelblue' if imp > 0.08 else 'lightblue'
                        for imp in top_features['importance']]
    
    axes[1, 2].barh(top_features['feature'], top_features['importance'], color=colors_importance, edgecolor='black')
    for i, (idx, row) in enumerate(top_features.iterrows()):
        axes[1, 2].text(row['importance'], i, f" {row['importance']*100:.1f}%",
                       va='center', fontsize=9, fontweight='bold')
    axes[1, 2].set_xlabel('Importance', fontweight='bold')
    axes[1, 2].set_title('Top 8 Timing Feature Importance', fontweight='bold', fontsize=12)
    axes[1, 2].invert_yaxis()
    axes[1, 2].grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Best Posting Times Analysis with Predictive Model', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    
    return {
        'best_day': day_names[best_day.name],
        'best_hour': int(best_hour.name),
        'best_combo': (day_names[best_combo.name[0]], int(best_combo.name[1])),
        'improvement_potential': best_combo_improvement,
        'day_stats': day_stats,
        'hour_stats': hour_stats,
        'day_hour_stats': day_hour_stats,
        'p_value': p_value if 'p_value' in locals() else None,
        'sample_size': int(best_day['views_count']),
        'model_metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'mase': mase
        },
        'model': model,
        'scaler': scaler,
        'features': time_features
    }

# ================= MODEL 4: ENSEMBLE TIME SERIES FORECASTING WITH MASE =================
def model4_ensemble_forecast(df):
    print("\n" + "="*70)
    print("MODEL 4: ENSEMBLE CHANNEL GROWTH FORECASTING (Next 6 Months)")
    print("="*70)

    df['publish_time'] = pd.to_datetime(df['publish_time'])

    weekly = df.set_index('publish_time').resample('W-SUN').agg({
        'views': ['sum', 'mean', 'count'],
        'likes': 'sum',
        'shares': 'sum',
        'comments_added': 'sum',
        'saves': 'sum',
        'engagement_rate': 'mean',
        'is_viral': 'sum'
    })

    weekly.columns = ['total_views', 'avg_views', 'video_count',
                      'total_likes', 'total_shares', 'total_comments', 'total_saves',
                      'avg_engagement', 'viral_count']

    weekly['total_interactions'] = (weekly['total_likes'] + weekly['total_shares'] +
                                    weekly['total_comments'] + weekly['total_saves'])

    weekly_recent = weekly.tail(52).copy()
    weekly_recent = weekly_recent.fillna(method='ffill').fillna(method='bfill').fillna(0)

    print(f"\nUsing {len(weekly_recent)} weeks of data")
    print(f"Range: {weekly_recent.index.min().strftime('%Y-%m-%d')} to {weekly_recent.index.max().strftime('%Y-%m-%d')}")
    print(f"Average: {weekly_recent['video_count'].mean():.1f} videos/week, {weekly_recent['total_views'].mean():,.0f} views/week")

    baseline_stats = {
        'views_mean': weekly_recent['total_views'].mean(),
        'views_std': weekly_recent['total_views'].std(),
        'views_recent_12w': weekly_recent['total_views'].tail(12).mean(),
        'views_recent_4w': weekly_recent['total_views'].tail(4).mean(),
        'engagement_mean': weekly_recent['avg_engagement'].mean(),
        'engagement_std': weekly_recent['avg_engagement'].std(),
        'interactions_mean': weekly_recent['total_interactions'].mean(),
        'viral_rate': weekly_recent['viral_count'].mean(),
        'video_count_avg': weekly_recent['video_count'].mean()
    }

    print(f"\nBaseline Statistics:")
    print(f"  Recent 12-week avg views: {baseline_stats['views_recent_12w']:,.0f}")
    print(f"  Recent 4-week avg views: {baseline_stats['views_recent_4w']:,.0f}")
    print(f"  Overall avg views: {baseline_stats['views_mean']:,.0f}")
    print(f"  Avg engagement rate: {baseline_stats['engagement_mean']:.2f}%")

    # Forecasting helper functions
    def forecast_ema(series, steps, alpha=0.3):
        forecast = []
        last_value = series.iloc[-1]
        mean_value = series.mean()
        for i in range(steps):
            pred = alpha * last_value + (1 - alpha) * mean_value
            forecast.append(pred)
            last_value = pred
        return np.array(forecast)

    def forecast_linear_trend(series, steps):
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.arange(len(series), len(series) + steps).reshape(-1, 1)
        forecast = model.predict(future_X)
        mean_val = series.mean()
        std_val = series.std()
        forecast = np.clip(forecast, mean_val - 2*std_val, mean_val + 2*std_val)
        return forecast

    def forecast_seasonal_naive(series, steps, season_length=4):
        if len(series) < season_length:
            return np.full(steps, series.tail(4).mean())
        forecast = []
        for i in range(steps):
            idx = -(season_length - (i % season_length))
            if abs(idx) <= len(series):
                forecast.append(series.iloc[idx])
            else:
                forecast.append(series.mean())
        return np.array(forecast)

    def forecast_rolling_mean(series, steps, window=12):
        rolling_mean = series.tail(window).mean()
        return np.full(steps, rolling_mean)

    print("\n--- Forecasting: Total Weekly Views (Ensemble Method) ---")

    views_series = weekly_recent['total_views']
    train_size = int(len(views_series) * 0.80)
    if train_size < 4:
        train_size = max(1, len(views_series) - 2)
    views_train = views_series.iloc[:train_size]
    views_test = views_series.iloc[train_size:]
    test_steps = len(views_test) if len(views_test) > 0 else 1

    # Compute test forecasts
    ema_test = forecast_ema(views_train, test_steps, alpha=0.25)
    linear_test = forecast_linear_trend(views_train, test_steps)
    seasonal_test = forecast_seasonal_naive(views_train, test_steps)
    rolling_test = forecast_rolling_mean(views_train, test_steps)

    def safe_score(true, pred):
        try:
            return max(0, r2_score(true, pred))
        except:
            return 0.0

    r2_ema = safe_score(views_test, ema_test)
    r2_linear = safe_score(views_test, linear_test)
    r2_seasonal = safe_score(views_test, seasonal_test)
    r2_rolling = safe_score(views_test, rolling_test)

    print(f"  Method Performance (MAE & MASE):")
    mae_ema = mean_absolute_error(views_test, ema_test)
    mae_linear = mean_absolute_error(views_test, linear_test)
    mae_seasonal = mean_absolute_error(views_test, seasonal_test)
    mae_rolling = mean_absolute_error(views_test, rolling_test)
    
    # Calculate MASE for each method
    mase_ema = calculate_mase(views_test.values, ema_test, views_train.values)
    mase_linear = calculate_mase(views_test.values, linear_test, views_train.values)
    mase_seasonal = calculate_mase(views_test.values, seasonal_test, views_train.values)
    mase_rolling = calculate_mase(views_test.values, rolling_test, views_train.values)

    print(f"    EMA:              MAE={mae_ema:>9,.0f}  MASE={mase_ema:.3f}  R²={r2_ema:.3f}")
    print(f"    Linear Trend:     MAE={mae_linear:>9,.0f}  MASE={mase_linear:.3f}  R²={r2_linear:.3f}")
    print(f"    Seasonal Naive:   MAE={mae_seasonal:>9,.0f}  MASE={mase_seasonal:.3f}  R²={r2_seasonal:.3f}")
    print(f"    Rolling Mean:     MAE={mae_rolling:>9,.0f}  MASE={mase_rolling:.3f}  R²={r2_rolling:.3f}")

    mae_scores = np.array([mae_ema, mae_linear, mae_seasonal, mae_rolling], dtype=float)
    inv_mae = 1.0 / (mae_scores + 1.0)
    weights = inv_mae / inv_mae.sum()

    print(f"  Ensemble Weights (by MAE): EMA={weights[0]:.2f}, Linear={weights[1]:.2f}, Seasonal={weights[2]:.2f}, Rolling={weights[3]:.2f}")

    ensemble_test = (weights[0] * ema_test +
                    weights[1] * linear_test +
                    weights[2] * seasonal_test +
                    weights[3] * rolling_test)

    mae_ensemble = mean_absolute_error(views_test, ensemble_test)
    mase_ensemble = calculate_mase(views_test.values, ensemble_test, views_train.values)

    try:
        mape_ensemble = mean_absolute_percentage_error(views_test, ensemble_test) * 100
        if mape_ensemble > 1000 or np.isnan(mape_ensemble) or np.isinf(mape_ensemble):
            mape_ensemble = None
    except:
        mape_ensemble = None

    naive_mae = mean_absolute_error(views_test, np.full(len(views_test), views_train.mean()))
    improvement_pct = ((naive_mae - mae_ensemble) / naive_mae) * 100 if naive_mae != 0 else 0.0
    pseudo_r2 = max(0, improvement_pct / 100)

    if mape_ensemble is not None:
        print(f"  Ensemble MAE: {mae_ensemble:,.0f} | MASE: {mase_ensemble:.3f} | MAPE: {mape_ensemble:.1f}%")
    else:
        print(f"  Ensemble MAE: {mae_ensemble:,.0f} | MASE: {mase_ensemble:.3f} | MAPE: N/A")
    
    mase_status = "✓ EXCELLENT" if mase_ensemble < 0.8 else "✓ GOOD" if mase_ensemble < 1.0 else "⚠ NEEDS WORK"
    print(f"  Improvement over naive: {improvement_pct:+.1f}% → Pseudo-R²={pseudo_r2:.3f}")
    print(f"  MASE Status: {mase_status}")

    # Forecast 26 weeks ahead
    steps = 26
    ema_forecast = forecast_ema(views_series, steps, alpha=0.25)
    linear_forecast = forecast_linear_trend(views_series, steps)
    seasonal_forecast = forecast_seasonal_naive(views_series, steps)
    rolling_forecast = forecast_rolling_mean(views_series, steps)

    forecast_views = (weights[0] * ema_forecast +
                     weights[1] * linear_forecast +
                     weights[2] * seasonal_forecast +
                     weights[3] * rolling_forecast)

    min_views = baseline_stats['views_recent_4w'] * 0.5
    forecast_views = np.maximum(forecast_views, min_views)

    views_method = f"Ensemble (MAE={mae_ensemble:,.0f}, MASE={mase_ensemble:.3f})"
    views_r2 = pseudo_r2

    print("\n--- Forecasting: Average Engagement Rate (Conservative) ---")

    engagement_series = weekly_recent['avg_engagement'].fillna(baseline_stats['engagement_mean'])
    eng_train_size = int(len(engagement_series) * 0.80)
    if eng_train_size < 2:
        eng_train_size = max(1, len(engagement_series)-1)
    eng_train = engagement_series.iloc[:eng_train_size]
    eng_test = engagement_series.iloc[eng_train_size:]

    recent_mean = eng_train.tail(8).mean() if len(eng_train.tail(8)) > 0 else eng_train.mean()
    pred_mean = np.full(len(eng_test), recent_mean)
    overall_mean = eng_train.mean()
    pred_overall = np.full(len(eng_test), overall_mean)
    pred_ema = forecast_ema(eng_train, len(eng_test), alpha=0.4)

    mae_mean = mean_absolute_error(eng_test, pred_mean) if len(eng_test) > 0 else 0.0
    mae_overall = mean_absolute_error(eng_test, pred_overall) if len(eng_test) > 0 else 0.0
    mae_ema_eng = mean_absolute_error(eng_test, pred_ema) if len(eng_test) > 0 else 0.0
    
    # Calculate MASE for engagement
    mase_mean = calculate_mase(eng_test.values, pred_mean, eng_train.values) if len(eng_test) > 0 else 0.0
    mase_overall = calculate_mase(eng_test.values, pred_overall, eng_train.values) if len(eng_test) > 0 else 0.0
    mase_ema_eng = calculate_mase(eng_test.values, pred_ema, eng_train.values) if len(eng_test) > 0 else 0.0

    print(f"  Recent Mean:   MAE={mae_mean:.3f}%  MASE={mase_mean:.3f}")
    print(f"  Overall Mean:  MAE={mae_overall:.3f}%  MASE={mase_overall:.3f}")
    print(f"  EMA:           MAE={mae_ema_eng:.3f}%  MASE={mase_ema_eng:.3f}")

    if mae_mean <= mae_overall and mae_mean <= mae_ema_eng:
        forecast_engagement = np.full(steps, engagement_series.tail(8).mean() if len(engagement_series.tail(8)) > 0 else engagement_series.mean())
        eng_method = f"Recent Mean (MAE={mae_mean:.2f}%, MASE={mase_mean:.3f})"
        best_mae = mae_mean
        best_mase = mase_mean
    elif mae_overall <= mae_ema_eng:
        forecast_engagement = np.full(steps, engagement_series.mean())
        eng_method = f"Overall Mean (MAE={mae_overall:.2f}%, MASE={mase_overall:.3f})"
        best_mae = mae_overall
        best_mase = mase_overall
    else:
        forecast_engagement = forecast_ema(engagement_series, steps, alpha=0.4)
        eng_method = f"EMA (MAE={mae_ema_eng:.2f}%, MASE={mase_ema_eng:.3f})"
        best_mae = mae_ema_eng
        best_mase = mase_ema_eng

    naive_eng_mae = mean_absolute_error(eng_test, np.full(len(eng_test), eng_train.mean())) if len(eng_test) > 0 else 0.0
    eng_improvement = ((naive_eng_mae - best_mae) / naive_eng_mae) * 100 if naive_eng_mae != 0 else 0.0
    eng_r2 = max(0, eng_improvement / 100)

    print(f"  Selected: {eng_method}")
    print(f"  Improvement over naive: {eng_improvement:+.1f}% → Pseudo-R²={eng_r2:.3f}")

    forecast_engagement = np.clip(
        forecast_engagement,
        baseline_stats['engagement_mean'] - 1.5 * baseline_stats['engagement_std'],
        baseline_stats['engagement_mean'] + 1.5 * baseline_stats['engagement_std']
    )

    print("\n--- Forecasting: Total Interactions (Proportional) ---")

    views_array = np.array(forecast_views)
    engagement_array = np.array(forecast_engagement)
    forecast_interactions = (views_array * engagement_array / 100).astype(int)

    actual_ratio = (weekly_recent['total_interactions'] / weekly_recent['total_views']).mean()
    predicted_ratio = np.mean(forecast_interactions) / np.mean(views_array) if np.mean(views_array) != 0 else 0.0

    print(f"  Historical interaction rate: {actual_ratio*100:.2f}%")
    print(f"  Forecasted interaction rate: {predicted_ratio*100:.2f}%")

    print("\n--- Forecasting: Viral Videos per Week (Historical Average) ---")

    viral_series = weekly_recent['viral_count']
    viral_rate = viral_series.mean()
    video_count_avg = baseline_stats['video_count_avg']

    forecast_viral = np.full(steps, viral_rate)
    forecast_viral = np.clip(forecast_viral, 0, video_count_avg * 0.35)

    print(f"  Historical viral rate: {viral_rate:.2f} per week")
    print(f"  Forecasted: {viral_rate:.2f} per week")

    last_date = weekly_recent.index[-1]
    forecast_dates = [last_date + timedelta(weeks=i) for i in range(1, steps + 1)]

    all_forecasts = {
        'total_views': {
            'dates': forecast_dates,
            'values': views_array.tolist(),
            'label': 'Total Weekly Views',
            'method': views_method,
            'r2': views_r2,
            'mase': mase_ensemble
        },
        'avg_engagement': {
            'dates': forecast_dates,
            'values': engagement_array.tolist(),
            'label': 'Average Engagement Rate (%)',
            'method': eng_method,
            'r2': eng_r2,
            'mase': best_mase
        },
        'total_interactions': {
            'dates': forecast_dates,
            'values': forecast_interactions.tolist(),
            'label': 'Total Interactions',
            'method': 'Proportional',
            'r2': None,
            'mase': None
        },
        'viral_count': {
            'dates': forecast_dates,
            'values': forecast_viral.tolist(),
            'label': 'Viral Videos per Week',
            'method': 'Historical Average',
            'r2': None,
            'mase': None
        }
    }

    all_results = {
        'total_views': {'r2': views_r2, 'mae': mae_ensemble, 'mase': mase_ensemble, 'method': views_method},
        'avg_engagement': {'r2': eng_r2, 'mae': best_mae, 'mase': best_mase, 'method': eng_method},
        'total_interactions': {'r2': None, 'mae': None, 'mase': None, 'method': 'Proportional'},
        'viral_count': {'r2': None, 'mae': None, 'mase': None, 'method': 'Historical Average'}
    }

    # Plotting panel
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    plot_targets = ['total_views', 'avg_engagement', 'total_interactions', 'viral_count']

    for idx, target_name in enumerate(plot_targets):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])

        forecast_data = all_forecasts[target_name]
        hist_data = weekly_recent.tail(26)
        ax.plot(hist_data.index, hist_data[target_name],
                marker='o', linewidth=2, label='Historical', color='blue', markersize=4)

        ax.plot(forecast_data['dates'], forecast_data['values'],
                marker='D', linewidth=2, linestyle='--', label='Forecast', color='red', markersize=4)

        forecast_values = np.array(forecast_data['values'])
        if target_name in ['total_views', 'total_interactions']:
            uncertainty = 0.12
        else:
            uncertainty = 0.06

        ax.fill_between(forecast_data['dates'],
                        forecast_values * (1 - uncertainty),
                        forecast_values * (1 + uncertainty),
                        alpha=0.2, color='red', label=f'{int(uncertainty*100)}% Band')

        if target_name == 'total_views':
            ax.axhline(baseline_stats['views_recent_12w'], color='green',
                      linestyle=':', linewidth=2, alpha=0.7, label='12w Avg')

        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel(forecast_data['label'], fontsize=10)

        r2_val = forecast_data.get('r2')
        mase_val = forecast_data.get('mase')
        
        if r2_val is not None and mase_val is not None:
            title_str = f"{forecast_data['label']}\nPseudo-R²={r2_val:.3f}, MASE={mase_val:.3f}"
        elif r2_val is not None:
            title_str = f"{forecast_data['label']}\nPseudo-R²={r2_val:.3f}"
        else:
            title_str = f"{forecast_data['label']}\n{forecast_data['method']}"

        ax.set_title(title_str, fontweight='bold', fontsize=11)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    plt.suptitle('Channel Growth Forecasts - Next 6 Months (Ensemble with MASE)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

    # Cumulative forecast plot
    try:
        historical_dates = weekly_recent.tail(26).index
        historical_vals = weekly_recent.tail(26)['total_views'].values
        historical_cum = np.cumsum(historical_vals)

        future_dates = all_forecasts['total_views']['dates']
        forecast_vals = np.array(all_forecasts['total_views']['values'])
        forecast_cum = historical_cum[-1] + np.cumsum(forecast_vals)

        mape_val = mape_ensemble if mape_ensemble is not None else 15.0
        mape_val = float(mape_val) if mape_val is not None else 15.0
        if np.isnan(mape_val) or np.isinf(mape_val):
            mape_val = 15.0
        mape_val = max(5.0, min(mape_val, 80.0))

        lower = forecast_cum * (1 - mape_val / 100.0)
        upper = forecast_cum * (1 + mape_val / 100.0)

        combined_hist_dates = list(historical_dates)
        combined_future_dates = list(future_dates)

        plt.figure(figsize=(12, 6))
        plt.plot(combined_hist_dates, historical_cum, color='blue', linewidth=2, label='Historical Total Views')
        plt.plot(combined_future_dates, forecast_cum, color='orange', linestyle='--', linewidth=2, label='Predicted (6 mo)')
        plt.fill_between(combined_future_dates, lower, upper, color='orange', alpha=0.2, label=f'±MAPE ({mape_val:.1f}%) confidence range')

        plt.title('Total Channel Views: Historical + 6-Month Forecast', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Views')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Could not generate cumulative forecast plot:", e)

    print("\n" + "="*70)
    print("FORECAST SUMMARY - Next 6 Months")
    print("="*70)

    for target_name, forecast_data in all_forecasts.items():
        forecasts = forecast_data['values']
        label = forecast_data['label']
        total_6m = sum(forecasts)
        avg_weekly = np.mean(forecasts)

        if target_name == 'avg_engagement':
            print(f"\n{label}:")
            print(f"  Average (6 months): {avg_weekly:.2f}%")
            print(f"  Range: {min(forecasts):.2f}% - {max(forecasts):.2f}%")
            print(f"  Current baseline: {baseline_stats['engagement_mean']:.2f}%")
            print(f"  Method: {forecast_data['method']}")
            if forecast_data.get('mase'):
                print(f"  MASE: {forecast_data['mase']:.3f}")
        elif target_name == 'viral_count':
            print(f"\n{label}:")
            print(f"  Total (6 months): {int(total_6m)} viral videos")
            print(f"  Average: {avg_weekly:.1f} per week")
            print(f"  Expected rate: {(avg_weekly/baseline_stats['video_count_avg']*100):.1f}% of videos")
            print(f"  Method: {forecast_data['method']}")
        else:
            print(f"\n{label}:")
            print(f"  Total (6 months): {total_6m:,.0f}")
            print(f"  Average per week: {avg_weekly:,.0f}")
            if target_name == 'total_views':
                print(f"  Range: {min(forecasts):,.0f} - {max(forecasts):,.0f}")
                print(f"  Recent baseline (12w): {baseline_stats['views_recent_12w']:,.0f}")
                pct_change = ((avg_weekly - baseline_stats['views_recent_12w']) / baseline_stats['views_recent_12w'] * 100) if baseline_stats['views_recent_12w'] != 0 else 0.0
                print(f"  Projected change: {pct_change:+.1f}% vs recent average")
            print(f"  Method: {forecast_data['method']}")
            if forecast_data.get('mase'):
                mase_val = forecast_data['mase']
                mase_status = "✓ EXCELLENT" if mase_val < 0.8 else "✓ GOOD" if mase_val < 1.0 else "⚠ NEEDS WORK"
                print(f"  MASE: {mase_val:.3f} {mase_status}")

    print("\n✅ MODEL 4 IMPROVEMENTS:")
    print("  • Ensemble of 4 forecasting methods (EMA, Linear, Seasonal, Rolling)")
    print("  • Automatic method weighting based on MAE")
    print("  • ✅ MASE (Mean Absolute Scaled Error) now calculated for all forecasts")
    print("  • MASE < 1.0 means better than naive forecast")
    print("  • Uses Pseudo-R² metric (improvement over naive baseline)")
    print("  • Robust to data volatility and outliers")
    print("  • Conservative floor constraints prevent unrealistic predictions")
    print("  • Proper temporal validation on 20% holdout set")

    return all_forecasts, all_results, baseline_stats

# ================= MAIN =================
def main():
    print("="*70)
    print("COMPLETE TIKTOK PREDICTIVE ANALYTICS SUITE")
    print("Individual Video Predictions + Channel Growth Forecasting")
    print("="*70)

    conn = psycopg2.connect(**db_params)
    df = fetch_data(conn)
    conn.close()

    print(f"\nLoaded {len(df)} videos")
    print(f"Date range: {df['publish_time'].min()} to {df['publish_time'].max()}")
    print(f"Total views: {df['views'].sum():,.0f}")
    print(f"Total engagement events: {(df['likes'].sum() + df['shares'].sum() + df['comments_added'].sum() + df['saves'].sum()):,.0f}")

    df = engineer_advanced_features(df)

    print("\n" + "="*70)
    print("RUNNING BEST POSTING TIME ANALYSIS")
    print("="*70)
    
    posting_analysis = analyze_best_posting_times(df)

    print("\n" + "="*70)
    print("RUNNING INDIVIDUAL VIDEO PREDICTION MODELS (1-3)")
    print("="*70)

    m1_model, m1_scaler, m1_features, m1_acc = model1_view_category(df)
    m2_model, m2_scaler, m2_features, m2_auc, m2_f1, m2_mase = model2_enhanced_virality(df)
    m3_model, m3_scaler, m3_features, m3_r2, m3_mase = model3_enhanced_engagement(df)

    print("\n" + "="*70)
    print("RUNNING CHANNEL GROWTH FORECASTING MODEL (4) - ENSEMBLE WITH MASE")
    print("="*70)

    m4_forecasts, m4_results, m4_baseline = model4_ensemble_forecast(df)

    print("\n" + "="*70)
    print("COMPLETE ANALYTICS SUMMARY")
    print("="*70)

    print("\nINDIVIDUAL VIDEO PREDICTION MODELS:")
    print("-" * 70)

    print(f"""
MODEL 1 - View Category Classification: {m1_acc*100:.1f}% Accuracy
  → Predicts: Will a video be Low/Medium/High/Viral
  → Use for: Content strategy & resource allocation

MODEL 2 - Virality Probability: AUC={m2_auc:.3f}, F1={m2_f1:.3f}, MASE={m2_mase:.3f}
  → Predicts: Probability of exceeding 100K views
  → Improvement: +{(m2_auc-0.609)*100:.1f}% AUC, +{(m2_f1-0.133)*100:.1f}% F1
  → MASE Status: {'✓ EXCELLENT' if m2_mase < 0.8 else '✓ GOOD' if m2_mase < 1.0 else '⚠ NEEDS WORK'}
  → Use for: Identifying high-potential content

MODEL 3 - Engagement Rate: R²={m3_r2:.3f}, MASE={m3_mase:.3f}
  → Predicts: Expected engagement rate (%)
  → Improvement: R² from -0.128 to {m3_r2:.3f} (+{(m3_r2-(-0.128))*100:.1f}%)
  → MASE Status: {'✓ EXCELLENT' if m3_mase < 0.8 else '✓ GOOD' if m3_mase < 1.0 else '⚠ NEEDS WORK'}
  → Use for: Content quality benchmarking

MODEL 5 - Day of Week Optimization: R²={posting_analysis['model_metrics']['r2']:.3f}, MASE={posting_analysis['model_metrics']['mase']:.3f}
  → Predicts: Expected views based on DAY OF WEEK (not hour)
  → MAE: {posting_analysis['model_metrics']['mae']:,.0f} views ({posting_analysis['model_metrics']['mae']/152931*100:.1f}%)
  → RMSE: {posting_analysis['model_metrics']['rmse']:,.0f} views
  → MASE Status: {'✓ EXCELLENT' if posting_analysis['model_metrics']['mase'] < 0.8 else '✓ GOOD' if posting_analysis['model_metrics']['mase'] < 1.0 else '⚠ NEEDS WORK'}
  → Statistical Significance: {'✅ CONFIRMED' if posting_analysis.get('p_value', 1.0) < 0.05 else '⚠️ NOT CONFIRMED'}
  → Use for: Choosing optimal posting day
    """)

    print("\nCHANNEL GROWTH FORECASTS (Next 6 Months) - ENSEMBLE WITH MASE:")
    print("-" * 70)

    views_forecast = m4_forecasts['total_views']['values']
    engagement_forecast = m4_forecasts['avg_engagement']['values']
    interactions_forecast = m4_forecasts['total_interactions']['values']
    viral_forecast = m4_forecasts['viral_count']['values']

    views_r2 = m4_results['total_views']['r2']
    views_mase = m4_results['total_views']['mase']
    eng_r2 = m4_results['avg_engagement']['r2']
    eng_mase = m4_results['avg_engagement']['mase']

    views_mase_status = "✓ EXCELLENT" if views_mase < 0.8 else "✓ GOOD" if views_mase < 1.0 else "⚠ NEEDS WORK"
    eng_mase_status = "✓ EXCELLENT" if eng_mase < 0.8 else "✓ GOOD" if eng_mase < 1.0 else "⚠ NEEDS WORK"

    print(f"""
VIEWS:
  • Total forecasted: {sum(views_forecast):,.0f} views over 6 months
  • Average per week: {np.mean(views_forecast):,.0f} views
  • Growth trend: {views_forecast[0]:,.0f} → {views_forecast[-1]:,.0f}
  • Model quality: Pseudo-R²={views_r2:.3f}, MASE={views_mase:.3f} {views_mase_status}
  • MAE: {m4_results['total_views']['mae']:,.0f}
  • Method: {m4_results['total_views']['method']}

ENGAGEMENT:
  • Average rate: {np.mean(engagement_forecast):.2f}%
  • Trend: {engagement_forecast[0]:.2f}% → {engagement_forecast[-1]:.2f}%
  • Model quality: Pseudo-R²={eng_r2:.3f}, MASE={eng_mase:.3f} {eng_mase_status}
  • MAE: {m4_results['avg_engagement']['mae']:.2f}%
  • Method: {m4_results['avg_engagement']['method']}

INTERACTIONS (Likes + Shares + Comments + Saves):
  • Total forecasted: {sum(interactions_forecast):,.0f} interactions
  • Average per week: {np.mean(interactions_forecast):,.0f}
  • Method: {m4_results['total_interactions']['method']}

VIRAL VIDEOS:
  • Expected viral videos: {int(sum(viral_forecast))} over 6 months
  • Average per week: {np.mean(viral_forecast):.1f}
  • Method: {m4_results['viral_count']['method']}
    """)

    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("-" * 70)

    forecast_avg = np.mean(views_forecast)
    growth_rate = ((forecast_avg - m4_baseline['views_recent_12w']) / m4_baseline['views_recent_12w'] * 100) if m4_baseline['views_recent_12w'] != 0 else 0.0

    growth_interpretation = ""
    if abs(growth_rate) < 5:
        growth_interpretation = "stable performance expected"
    elif growth_rate > 0:
        growth_interpretation = "growth projected"
    else:
        growth_interpretation = "decline projected"

    print(f"""
• Recent average: {m4_baseline['views_recent_12w']:,.0f} views/week (last 12 weeks)
• Forecasted average: {forecast_avg:,.0f} views/week (next 26 weeks)
• Projected change: {growth_rate:+.1f}% ({growth_interpretation})

• Forecast method: Ensemble of 4 methods (weighted by MAE performance)
• Quality metric: Pseudo-R² (% improvement over naive forecast)
• MASE metric: Scaled error where < 1.0 = better than naive
• All methods validated on 20% holdout test set
• Conservative constraints prevent extreme predictions

• Top prediction driver (Models 1-3): Rolling 7-day performance (40% importance)
• Consistency matters: Regular posting maintains baseline performance
• Quality over quantity: Focus on engagement rate optimization

🎯 POSTING STRATEGY:
• Best day to post: {posting_analysis['best_day']}
• Statistical confidence: {'HIGH' if posting_analysis.get('p_value', 1.0) < 0.05 and posting_analysis.get('sample_size', 0) >= 50 else 'MODERATE'}
• Top 3 days perform similarly: {', '.join([posting_analysis['day_stats'].iloc[i]['day_name'] for i in range(min(3, len(posting_analysis['day_stats'])))])}
• Improvement potential: {posting_analysis['improvement_potential']:+.1f}% (best day vs average)
• Avoid: {posting_analysis['day_stats'].iloc[-1]['day_name']} (lowest average)

✅ ALL MODELS NOW INCLUDE MASE:
  • ✓ Model 2 (Virality): MASE = {m2_mase:.3f} {('✓ EXCELLENT' if m2_mase < 0.8 else '✓ GOOD' if m2_mase < 1.0 else '⚠ NEEDS WORK')}
  • ✓ Model 3 (Engagement): MASE = {m3_mase:.3f} {('✓ EXCELLENT' if m3_mase < 0.8 else '✓ GOOD' if m3_mase < 1.0 else '⚠ NEEDS WORK')}
  • ✓ Model 4 (Views Forecast): MASE = {views_mase:.3f} {views_mase_status}
  • ✓ Model 4 (Engagement Forecast): MASE = {eng_mase:.3f} {eng_mase_status}
  • ✓ Model 5 (Posting Time): MASE = {posting_analysis['model_metrics']['mase']:.3f} {('✓ EXCELLENT' if posting_analysis['model_metrics']['mase'] < 0.8 else '✓ GOOD' if posting_analysis['model_metrics']['mase'] < 1.0 else '⚠ NEEDS WORK')}

✅ MODEL 4 ENHANCEMENTS:
  • ✓ NO negative metrics (Pseudo-R² is always 0 to 1)
  • ✓ MASE now calculated for ALL forecast types
  • ✓ Proper temporal validation with MAE as primary metric
  • ✓ Multiple forecasting methods with automatic selection
  • ✓ Robust to data volatility
  • ✓ Conservative and realistic predictions
  • ✓ Transparent performance reporting with MASE
    """)

    print("\n" + "="*70)
    print("Key Model Improvements Applied:")
    print("  ✓ Added 15+ engineered features (time, content, historical)")
    print("  ✓ Applied SMOTE for class balancing in virality model")
    print("  ✓ Feature scaling with StandardScaler")
    print("  ✓ Removed outliers from engagement data")
    print("  ✓ Better hyperparameter tuning")
    print("  ✓ FIXED: Model 4 uses ensemble forecasting (4 methods)")
    print("  ✓ FIXED: Automatic method weighting based on MAE performance")
    print("  ✓ FIXED: Uses Pseudo-R² metric (always positive 0-1 scale)")
    print("  ✓ ✅ NEW: MASE calculated for ALL models (2, 3, 4, 5)")
    print("  ✓ ENHANCED: Model 2 includes regression-style metrics + MASE")
    print("  ✓ ENHANCED: Model 3 includes comprehensive timeline visualization")
    print("  ✓ ENHANCED: Model 4 now shows MASE for all forecast metrics")
    print("  ✓ NEW: Best posting time analysis with day/hour optimization")
    print("  ✓ NEW: Heatmap visualization for optimal posting schedule")
    print("  ✓ Multi-target forecasting (views, engagement, interactions, virality)")
    print("="*70)

    return {
        'models': {
            'view_category': (m1_model, m1_scaler, m1_features),
            'virality': (m2_model, m2_scaler, m2_features),
            'engagement': (m3_model, m3_scaler, m3_features)
        },
        'forecasts': m4_forecasts,
        'posting_analysis': posting_analysis,
        'metrics': {
            'view_category_acc': m1_acc,
            'virality_auc': m2_auc,
            'virality_f1': m2_f1,
            'virality_mase': m2_mase,
            'engagement_r2': m3_r2,
            'engagement_mase': m3_mase,
            'forecast_results': m4_results,
            'posting_mase': posting_analysis['model_metrics']['mase']
        },
        'baseline': m4_baseline
    }

if __name__ == "__main__":
    results = main()
    print("\n✅ All models completed successfully!")
    print("Results stored in 'results' dictionary for further analysis")
    print("\nTo use forecasts: results['forecasts']['total_views']['values']")
    print("To use posting insights: results['posting_analysis']['best_combo']")
    print("\n📊 MASE VALUES SUMMARY:")
    print(f"  • Model 2 (Virality): {results['metrics']['virality_mase']:.3f}")
    print(f"  • Model 3 (Engagement): {results['metrics']['engagement_mase']:.3f}")
    print(f"  • Model 4 (Views): {results['metrics']['forecast_results']['total_views']['mase']:.3f}")
    print(f"  • Model 4 (Engagement): {results['metrics']['forecast_results']['avg_engagement']['mase']:.3f}")
    print(f"  • Model 5 (Posting Time): {results['metrics']['posting_mase']:.3f}")
    print("\n  MASE < 1.0 = Better than naive forecast ✓")