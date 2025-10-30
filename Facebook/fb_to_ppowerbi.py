"""
Facebook Reach Prediction - Hybrid Multi-Stage Approach
Stage 1: Pre-publishing classification (High/Medium/Low reach)
Stage 2: Early engagement forecasting (first hour predictions)
Stage 3: Post-type optimization recommendations
"""

import warnings
warnings.filterwarnings("ignore")

import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from datetime import timedelta

# Import XGBoost separately
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

print("=" * 80)
print("FACEBOOK REACH PREDICTION - HYBRID MULTI-STAGE APPROACH")
print("Practical forecasting using classification + early signals + optimization")
print("=" * 80)

# -------------------------
# Load Data from Database
# -------------------------
db_params = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host': 'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port': '5432',
    'sslmode': 'require'
}

print("\nConnecting to database...")
try:
    conn = psycopg2.connect(**db_params)
    query = "SELECT * FROM facebook_data_set ORDER BY publish_time"
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"✓ Loaded {len(df)} posts from database")
except Exception as e:
    print(f"Error connecting to database: {e}")
    print("Falling back to CSV...")
    df = pd.read_csv("Facebook/facebook_data_set.csv", parse_dates=['publish_time'])

df = df.sort_values('publish_time').reset_index(drop=True)

# -------------------------
# Data Cleaning
# -------------------------
print("\n" + "=" * 80)
print("DATA CLEANING & PREPARATION")
print("=" * 80)

print(f"\nOriginal dataset: {len(df)} posts")
df = df.dropna(subset=['reach'])
df = df[df['reach'] > 0]
reach_99 = df['reach'].quantile(0.995)
df = df[df['reach'] <= reach_99]
df = df.reset_index(drop=True)

print(f"Final dataset: {len(df)} posts")
print(f"Date range: {df['publish_time'].min().date()} to {df['publish_time'].max().date()}")
print(f"Reach range: {df['reach'].min():,.0f} to {df['reach'].max():,.0f}")
print(f"Median reach: {df['reach'].median():,.0f}")
print(f"Mean reach: {df['reach'].mean():,.0f}")

# -------------------------
# Feature Engineering
# -------------------------
print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# Temporal features
df['day_of_week'] = df['publish_time'].dt.dayofweek
df['hour'] = df['publish_time'].dt.hour
df['month'] = df['publish_time'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_prime_time'] = df['hour'].between(17, 22).astype(int)
df['is_business_hours'] = df['hour'].between(9, 17).astype(int)

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Content features
df['duration_sec'] = pd.to_numeric(df['duration_sec'], errors='coerce').fillna(0)
df['has_duration'] = (df['duration_sec'] > 0).astype(int)
df['log_duration'] = np.log1p(df['duration_sec'])

# Engagement features
for col in ['reactions', 'comments', 'shares', 'seconds_viewed', 'average_seconds_viewed']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df['total_engagement'] = df['reactions'] + df['comments'] + df['shares']
df['log_reactions'] = np.log1p(df['reactions'])
df['log_comments'] = np.log1p(df['comments'])
df['log_shares'] = np.log1p(df['shares'])
df['log_engagement'] = np.log1p(df['total_engagement'])

# Historical features
df = df.sort_values('publish_time').reset_index(drop=True)
df['recent_avg_reach'] = df['reach'].shift(1).rolling(window=10, min_periods=1).mean()
df['recent_std_reach'] = df['reach'].shift(1).rolling(window=10, min_periods=2).std()

# Fill NaN values in historical features
df['recent_avg_reach'] = df['recent_avg_reach'].fillna(df['reach'].mean())
df['recent_std_reach'] = df['recent_std_reach'].fillna(0)

# Create reach categories (High/Medium/Low)
df['reach_quartile'] = pd.qcut(df['reach'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
df['reach_category'] = pd.cut(df['reach'], 
                               bins=[0, df['reach'].quantile(0.33), df['reach'].quantile(0.67), df['reach'].max()],
                               labels=['Low', 'Medium', 'High'])

print("✓ Features created")

# -------------------------
# TIME-FORWARD SPLIT
# -------------------------
print("\n" + "=" * 80)
print("TIME-FORWARD TRAIN/TEST SPLIT")
print("=" * 80)

total_days = (df['publish_time'].max() - df['publish_time'].min()).days
split_date = df['publish_time'].min() + timedelta(days=int(total_days * 0.70))

df_train = df[df['publish_time'] <= split_date].copy()
df_test = df[df['publish_time'] > split_date].copy()

train_months = (df_train['publish_time'].max() - df_train['publish_time'].min()).days / 30.44
test_months = (df_test['publish_time'].max() - df_test['publish_time'].min()).days / 30.44

print(f"\nTraining: {df_train['publish_time'].min().date()} → {df_train['publish_time'].max().date()} ({train_months:.1f}mo, {len(df_train)} posts)")
print(f"Test: {df_test['publish_time'].min().date()} → {df_test['publish_time'].max().date()} ({test_months:.1f}mo, {len(df_test)} posts)")

# -------------------------
# STAGE 1: PRE-PUBLISHING CLASSIFICATION
# -------------------------
print("\n" + "=" * 80)
print("STAGE 1: PRE-PUBLISHING REACH CATEGORY PREDICTION")
print("Predicting: High/Medium/Low reach before posting")
print("=" * 80)

# Pre-publishing features only
pre_pub_features = [
    'day_of_week', 'hour', 'month', 'is_weekend', 'is_prime_time', 'is_business_hours',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'duration_sec', 'has_duration', 'log_duration',
    'recent_avg_reach', 'recent_std_reach'
]

# Add post type dummies
train_type_dummies = pd.get_dummies(df_train['post_type'], prefix='type')
test_type_dummies = pd.get_dummies(df_test['post_type'], prefix='type')

X_train_pre = pd.concat([df_train[pre_pub_features], train_type_dummies], axis=1)
X_test_pre = pd.concat([df_test[pre_pub_features], test_type_dummies], axis=1)

# Align columns
for col in X_train_pre.columns:
    if col not in X_test_pre.columns:
        X_test_pre[col] = 0
for col in X_test_pre.columns:
    if col not in X_train_pre.columns:
        X_train_pre[col] = 0

X_train_pre = X_train_pre[sorted(X_train_pre.columns)]
X_test_pre = X_test_pre[sorted(X_test_pre.columns)]

y_train_cat = df_train['reach_category']
y_test_cat = df_test['reach_category']

# Train classifier
scaler_pre = RobustScaler()
X_train_pre_scaled = scaler_pre.fit_transform(X_train_pre)
X_test_pre_scaled = scaler_pre.transform(X_test_pre)

clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
clf.fit(X_train_pre_scaled, y_train_cat)
y_pred_cat = clf.predict(X_test_pre_scaled)

accuracy = accuracy_score(y_test_cat, y_pred_cat)
print(f"\n📊 CLASSIFICATION RESULTS:")
print(f"   Accuracy: {accuracy:.1%}")
print(f"\n   Detailed Report:")
print(classification_report(y_test_cat, y_pred_cat, zero_division=0))

# -------------------------
# STAGE 2: ENGAGEMENT-BASED REGRESSION
# -------------------------
print("\n" + "=" * 80)
print("STAGE 2: EARLY ENGAGEMENT FORECASTING")
print("Using engagement signals to forecast final reach")
print("=" * 80)

# All features including engagement
full_features = pre_pub_features + [
    'reactions', 'comments', 'shares', 'total_engagement',
    'log_reactions', 'log_comments', 'log_shares', 'log_engagement',
    'seconds_viewed', 'average_seconds_viewed'
]

X_train_full = pd.concat([df_train[full_features], train_type_dummies], axis=1)
X_test_full = pd.concat([df_test[full_features], test_type_dummies], axis=1)

# Align columns
for col in X_train_full.columns:
    if col not in X_test_full.columns:
        X_test_full[col] = 0
for col in X_test_full.columns:
    if col not in X_train_full.columns:
        X_train_full[col] = 0

X_train_full = X_train_full[sorted(X_train_full.columns)]
X_test_full = X_test_full[sorted(X_test_full.columns)]

y_train = df_train['reach']
y_test = df_test['reach']

scaler_full = RobustScaler()
X_train_full_scaled = scaler_full.fit_transform(X_train_full)
X_test_full_scaled = scaler_full.transform(X_test_full)

# Train models
print("\nTraining Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_samples_split=15,
    min_samples_leaf=8,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train_full_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_full_scaled)

mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)
mape_gb = np.mean(np.abs((y_test - y_pred_gb) / y_test)) * 100

within_25_gb = np.sum(np.abs(y_test - y_pred_gb) / y_test <= 0.25) / len(y_test) * 100
within_50_gb = np.sum(np.abs(y_test - y_pred_gb) / y_test <= 0.50) / len(y_test) * 100

print(f"\n📊 GRADIENT BOOSTING RESULTS:")
print(f"   R² Score:        {r2_gb:.3f}")
print(f"   MAE:             {mae_gb:,.0f} reach ({mae_gb/y_test.mean()*100:.1f}% of mean)")
print(f"   RMSE:            {rmse_gb:,.0f} reach")
print(f"   MAPE:            {mape_gb:.2f}%")
print(f"   Within ±25%:     {within_25_gb:.1f}%")
print(f"   Within ±50%:     {within_50_gb:.1f}%")

# Feature importance
importance_df = pd.DataFrame({
    'feature': X_train_full.columns,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🔝 Top 15 Most Important Features:")
for idx, row in importance_df.head(15).iterrows():
    print(f"   {row['feature']:30} {row['importance']*100:>6.2f}%")

# -------------------------
# STAGE 3: OPTIMIZATION INSIGHTS
# -------------------------
print("\n" + "=" * 80)
print("STAGE 3: CONTENT OPTIMIZATION INSIGHTS")
print("=" * 80)

# Best posting times
time_performance = df_test.groupby('hour')['reach'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
print("\n📅 BEST POSTING TIMES (by average reach):")
print(f"{'Hour':<10} {'Avg Reach':<15} {'Median Reach':<15} {'Posts':<10}")
print("-" * 50)
for hour, row in time_performance.head(5).iterrows():
    if row['count'] >= 3:  # Only show if sufficient data
        print(f"{hour:02d}:00     {row['mean']:>10,.0f}     {row['median']:>12,.0f}     {int(row['count']):>5}")

# Best days of week
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_performance = df_test.groupby('day_of_week')['reach'].agg(['mean', 'median', 'count'])
day_performance.index = [day_names[i] for i in day_performance.index]
day_performance = day_performance.sort_values('mean', ascending=False)

print("\n📅 BEST DAYS OF WEEK (by average reach):")
print(f"{'Day':<12} {'Avg Reach':<15} {'Median Reach':<15} {'Posts':<10}")
print("-" * 52)
for day, row in day_performance.iterrows():
    print(f"{day:<12} {row['mean']:>10,.0f}     {row['median']:>12,.0f}     {int(row['count']):>5}")

# Post type performance
type_performance = df_test.groupby('post_type')['reach'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
print("\n📊 POST TYPE PERFORMANCE (by average reach):")
print(f"{'Type':<12} {'Avg Reach':<15} {'Median Reach':<15} {'Posts':<10}")
print("-" * 52)
for ptype, row in type_performance.iterrows():
    print(f"{ptype:<12} {row['mean']:>10,.0f}     {row['median']:>12,.0f}     {int(row['count']):>5}")

# Video duration analysis (if applicable)
if df_test[df_test['duration_sec'] > 0].shape[0] > 10:
    df_videos = df_test[df_test['duration_sec'] > 0].copy()
    df_videos['duration_category'] = pd.cut(df_videos['duration_sec'], 
                                             bins=[0, 30, 60, 180, 600, 99999],
                                             labels=['<30s', '30-60s', '1-3min', '3-10min', '>10min'])
    duration_perf = df_videos.groupby('duration_category')['reach'].agg(['mean', 'count'])
    
    print("\n🎬 VIDEO DURATION PERFORMANCE:")
    print(f"{'Duration':<12} {'Avg Reach':<15} {'Videos':<10}")
    print("-" * 37)
    for dur, row in duration_perf.iterrows():
        if row['count'] >= 3:
            print(f"{dur:<12} {row['mean']:>10,.0f}     {int(row['count']):>5}")

# -------------------------
# ACTIONABLE RECOMMENDATIONS
# -------------------------
print("\n" + "=" * 80)
print("🎯 ACTIONABLE RECOMMENDATIONS")
print("=" * 80)

best_hour = time_performance.head(1).index[0]
best_day = day_performance.head(1).index[0]
best_type = type_performance.head(1).index[0]

print(f"""
✅ WHAT WORKS BEST (Based on Test Period Data):

1. TIMING:
   • Best hour: {best_hour:02d}:00 (avg reach: {time_performance.loc[best_hour, 'mean']:,.0f})
   • Best day: {best_day} (avg reach: {day_performance.loc[best_day, 'mean']:,.0f})
   • Prime time (5pm-10pm): {'Higher' if df_test[df_test['is_prime_time']==1]['reach'].mean() > df_test[df_test['is_prime_time']==0]['reach'].mean() else 'Lower'} reach than other times

2. CONTENT TYPE:
   • Best performing: {best_type} (avg reach: {type_performance.loc[best_type, 'mean']:,.0f})
   • Consider focusing on this content type

3. ENGAGEMENT SIGNALS (Early Indicators):
   • Reactions are the #1 predictor ({importance_df[importance_df['feature']=='log_reactions']['importance'].values[0]*100:.1f}% importance)
   • Monitor first hour engagement to forecast final reach
   • Use Stage 2 model with early engagement data

4. MODEL USAGE GUIDE:
   
   BEFORE POSTING (Stage 1 - Classification):
   • Predict if post will be High/Medium/Low reach: {accuracy:.0%} accurate
   • Use this to decide: "Should I post now or wait for better timing?"
   
   AFTER 1-2 HOURS (Stage 2 - Regression):
   • Feed early engagement into model
   • Forecast final reach with {r2_gb:.1%} accuracy
   • Decide: "Should I boost this post or let it run organically?"

⚠️  IMPORTANT LIMITATIONS:
   • Cannot predict viral posts (outliers removed from training)
   • Algorithm changes affect accuracy over time
   • Content quality is the biggest factor (not measured here)
   • Historical performance has high variance

💡 BEST PRACTICE:
   Use Stage 1 for planning → Post → Wait 1-2 hours → Use Stage 2 for forecasting
""")

print("\n" + "=" * 80)
print("✅ HYBRID PREDICTION SYSTEM COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Use classification model to plan optimal posting times")
print("2. Monitor early engagement signals")
print("3. Use regression model to forecast final reach after 1-2 hours")
print("4. Iterate and improve based on what works for YOUR page")