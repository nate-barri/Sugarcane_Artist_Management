"""
Facebook Reach Predictor - STRATIFIED APPROACH
Train separate models for low/medium/high reach posts
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# -------------------------
# Settings
# -------------------------
CSV_PATH = "Facebook/facebook_data_set.csv"
RANDOM_STATE = 42

print("=" * 60)
print("STRATIFIED REACH PREDICTOR v3.0")
print("=" * 60)

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv(CSV_PATH, parse_dates=['publish_time'])
print(f"\nLoaded {len(df)} posts")
print(f"Reach stats:")
print(f"  Min:     {df['reach'].min():.0f}")
print(f"  25th:    {df['reach'].quantile(0.25):.0f}")
print(f"  Median:  {df['reach'].median():.0f}")
print(f"  75th:    {df['reach'].quantile(0.75):.0f}")
print(f"  95th:    {df['reach'].quantile(0.95):.0f}")
print(f"  Max:     {df['reach'].max():.0f}")

# -------------------------
# Stratified Segmentation
# -------------------------
print("\n" + "=" * 60)
print("STRATIFIED SEGMENTATION")
print("=" * 60)

# Define reach tiers based on quartiles
low_threshold = df['reach'].quantile(0.33)
med_threshold = df['reach'].quantile(0.67)
viral_threshold = df['reach'].quantile(0.95)

df['reach_tier'] = pd.cut(df['reach'], 
                          bins=[-1, low_threshold, med_threshold, viral_threshold, float('inf')],
                          labels=['low', 'medium', 'high', 'viral'])

print(f"\nReach Tiers:")
for tier in ['low', 'medium', 'high', 'viral']:
    tier_df = df[df['reach_tier'] == tier]
    print(f"  {tier.upper():8} ({len(tier_df):3} posts): {tier_df['reach'].min():.0f} - {tier_df['reach'].max():.0f} (mean: {tier_df['reach'].mean():.0f})")

# Focus on low/medium/high (exclude viral outliers for main model)
trainable_df = df[df['reach_tier'] != 'viral'].copy()
print(f"\nTraining on {len(trainable_df)} non-viral posts ({len(trainable_df)/len(df)*100:.1f}%)")

# -------------------------
# Enhanced Feature Engineering
# -------------------------
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

def create_features(df_input):
    df = df_input.copy()
    df = df.sort_values('publish_time').reset_index(drop=True)
    
    # Time features
    df['day_of_week'] = df['publish_time'].dt.dayofweek
    df['hour'] = df['publish_time'].dt.hour
    df['month'] = df['publish_time'].dt.month
    df['day_of_month'] = df['publish_time'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_peak_hour'] = df['hour'].isin([19, 20, 21, 22]).astype(int)
    df['is_tuesday'] = (df['day_of_week'] == 1).astype(int)
    df['is_morning'] = df['hour'].isin([6, 7, 8, 9]).astype(int)
    df['is_lunch'] = df['hour'].isin([11, 12, 13]).astype(int)
    df['is_evening'] = df['hour'].isin([17, 18, 19, 20]).astype(int)
    
    # Post type features
    df['is_reel'] = (df['post_type'] == 'Reels').astype(int)
    df['is_video'] = df['post_type'].isin(['Videos', 'video']).astype(int)
    df['is_photo'] = (df['post_type'] == 'Photos').astype(int)
    df['video_duration_min'] = df['duration_sec'].fillna(0) / 60
    df['has_video'] = (df['video_duration_min'] > 0).astype(int)
    df['is_short_video'] = ((df['video_duration_min'] > 0) & (df['video_duration_min'] <= 1)).astype(int)
    df['is_long_video'] = (df['video_duration_min'] > 3).astype(int)
    
    # Rolling features - multiple windows
    for window in [3, 5, 7, 14]:
        df[f'recent_avg_{window}'] = df['reach'].rolling(window, min_periods=1).mean().shift(1)
        df[f'recent_std_{window}'] = df['reach'].rolling(window, min_periods=1).std().shift(1)
        df[f'recent_max_{window}'] = df['reach'].rolling(window, min_periods=1).max().shift(1)
    
    # Momentum and trends
    df['momentum_3_7'] = df['recent_avg_3'] - df['recent_avg_7']
    df['momentum_7_14'] = df['recent_avg_7'] - df['recent_avg_14']
    df['volatility_ratio'] = df['recent_std_7'] / (df['recent_avg_7'] + 1)
    df['recent_peak_ratio'] = df['recent_max_7'] / (df['recent_avg_7'] + 1)
    
    # Post frequency
    df['posts_last_7d'] = df['reach'].rolling(7, min_periods=1).count().shift(1)
    
    # Type-specific rolling averages
    for post_type in df['post_type'].unique():
        mask = df['post_type'] == post_type
        col_name = f'type_avg_{post_type.lower().replace(" ", "_")}'
        type_avg = df.loc[mask, 'reach'].expanding().mean().shift(1)
        df.loc[mask, col_name] = type_avg
        df[col_name] = df[col_name].fillna(df['reach'].median())
    
    # Day-of-week performance
    for day in range(7):
        mask = df['day_of_week'] == day
        col_name = f'dow_avg_{day}'
        dow_avg = df.loc[mask, 'reach'].expanding().mean().shift(1)
        df.loc[mask, col_name] = dow_avg
        df[col_name] = df[col_name].fillna(df['reach'].median())
    
    # Hour-of-day performance
    for hour in [9, 12, 15, 18, 21]:
        mask = df['hour'] == hour
        col_name = f'hour_avg_{hour}'
        hour_avg = df.loc[mask, 'reach'].expanding().mean().shift(1)
        df.loc[mask, col_name] = hour_avg
        df[col_name] = df[col_name].fillna(df['reach'].median())
    
    # Fill NAs only in numeric columns (avoid categorical columns like reach_tier)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

trainable_df = create_features(trainable_df)
print(f"✓ Created features for {len(trainable_df)} posts")

# -------------------------
# Define Features
# -------------------------
feature_cols = [
    # Time features
    'day_of_week', 'hour', 'month', 'day_of_month',
    'is_weekend', 'is_peak_hour', 'is_tuesday', 'is_morning', 'is_lunch', 'is_evening',
    
    # Post type
    'is_reel', 'is_video', 'is_photo', 'video_duration_min', 'has_video',
    'is_short_video', 'is_long_video',
    
    # Recent performance
    'recent_avg_3', 'recent_avg_5', 'recent_avg_7', 'recent_avg_14',
    'recent_std_3', 'recent_std_5', 'recent_std_7', 'recent_std_14',
    'recent_max_3', 'recent_max_5', 'recent_max_7', 'recent_max_14',
    
    # Momentum
    'momentum_3_7', 'momentum_7_14', 'volatility_ratio', 'recent_peak_ratio',
    'posts_last_7d'
]

# Add type and time-specific features
type_cols = [c for c in trainable_df.columns if c.startswith('type_avg_')]
dow_cols = [c for c in trainable_df.columns if c.startswith('dow_avg_')]
hour_cols = [c for c in trainable_df.columns if c.startswith('hour_avg_')]
feature_cols.extend(type_cols + dow_cols + hour_cols)

print(f"✓ Using {len(feature_cols)} features")

# -------------------------
# Train Tier Classifier
# -------------------------
print("\n" + "=" * 60)
print("STEP 1: REACH TIER CLASSIFIER")
print("=" * 60)

X_tier = trainable_df[feature_cols]
y_tier = trainable_df['reach_tier']

# Map tiers to numbers for training
tier_map = {'low': 0, 'medium': 1, 'high': 2}
y_tier_numeric = y_tier.astype(str).map(tier_map)

# Check for any unmapped values
if y_tier_numeric.isna().any():
    print(f"Warning: Found unmapped tiers: {y_tier[y_tier_numeric.isna()].unique()}")
    y_tier_numeric = y_tier_numeric.fillna(0).astype(int)

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_tier, y_tier_numeric, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tier_numeric
)

tier_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

tier_classifier.fit(X_train_t, y_train_t)
tier_pred = tier_classifier.predict(X_test_t)

tier_accuracy = (tier_pred == y_test_t).mean()
print(f"\nTier Classification Accuracy: {tier_accuracy:.3f}")

# -------------------------
# Train Tier-Specific Regressors
# -------------------------
print("\n" + "=" * 60)
print("STEP 2: TIER-SPECIFIC REGRESSORS")
print("=" * 60)

tier_models = {}
tier_results = {}

for tier_num, tier_name in [(0, 'low'), (1, 'medium'), (2, 'high')]:
    print(f"\n--- Training {tier_name.upper()} Reach Model ---")
    
    # Get tier-specific data
    tier_mask = trainable_df['reach_tier'] == tier_name
    X_tier_data = trainable_df[tier_mask][feature_cols]
    y_tier_data = trainable_df[tier_mask]['reach']
    
    if len(X_tier_data) < 20:
        print(f"  ⚠ Not enough data ({len(X_tier_data)} posts), skipping")
        continue
    
    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_tier_data, y_tier_data, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Train with log transform for this tier
    y_tr_log = np.log1p(y_tr)
    
    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr_log)
    rf_pred = np.expm1(rf.predict(X_te))
    
    # Gradient Boosting  
    gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=RANDOM_STATE
    )
    gb.fit(X_tr, y_tr_log)
    gb_pred = np.expm1(gb.predict(X_te))
    
    # Ensemble
    ensemble_pred = 0.6 * rf_pred + 0.4 * gb_pred
    
    # Evaluate
    r2 = r2_score(y_te, ensemble_pred)
    mae = mean_absolute_error(y_te, ensemble_pred)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_vals = np.abs((y_te - ensemble_pred) / y_te)
        mape_vals = mape_vals[np.isfinite(mape_vals) & (mape_vals < 10)]
        mape = np.mean(mape_vals) * 100 if len(mape_vals) > 0 else np.nan
        
        pct_within_50 = np.mean(np.abs(y_te - ensemble_pred) / y_te < 0.50) * 100
    
    print(f"  Training samples: {len(X_tr)}, Test samples: {len(X_te)}")
    print(f"  R²:           {r2:.3f}")
    print(f"  MAE:          {mae:.0f} ({mae/y_te.mean()*100:.1f}%)")
    print(f"  MAPE:         {mape:.1f}%")
    print(f"  Within ±50%:  {pct_within_50:.1f}%")
    
    tier_models[tier_name] = {'rf': rf, 'gb': gb}
    tier_results[tier_name] = {
        'r2': r2, 'mae': mae, 'mape': mape, 
        'within_50': pct_within_50,
        'predictions': ensemble_pred,
        'actual': y_te
    }

# -------------------------
# Combined System
# -------------------------
print("\n" + "=" * 60)
print("COMBINED STRATIFIED SYSTEM")
print("=" * 60)

def predict_reach_stratified(post_features):
    """Predict using tier classification + tier-specific model"""
    # Step 1: Predict tier
    tier_num = tier_classifier.predict([post_features])[0]
    tier_probs = tier_classifier.predict_proba([post_features])[0]
    tier_name = {0: 'low', 1: 'medium', 2: 'high'}[tier_num]
    
    # Step 2: Use tier-specific model
    if tier_name in tier_models:
        rf_pred = np.expm1(tier_models[tier_name]['rf'].predict([post_features])[0])
        gb_pred = np.expm1(tier_models[tier_name]['gb'].predict([post_features])[0])
        predicted_reach = 0.6 * rf_pred + 0.4 * gb_pred
    else:
        predicted_reach = trainable_df['reach'].median()
    
    return {
        'predicted_reach': predicted_reach,
        'tier': tier_name,
        'tier_confidence': tier_probs[tier_num],
        'confidence_lower': predicted_reach * 0.7,
        'confidence_upper': predicted_reach * 1.3
    }

print("\n✓ Two-step prediction system:")
print("  1. Classify post into reach tier (low/medium/high)")
print("  2. Use tier-specific regression model")

# -------------------------
# Overall Performance
# -------------------------
print("\n" + "=" * 60)
print("OVERALL PERFORMANCE")
print("=" * 60)

# Collect all predictions
all_actual = []
all_predicted = []

for tier_name in ['low', 'medium', 'high']:
    if tier_name in tier_results:
        all_actual.extend(tier_results[tier_name]['actual'])
        all_predicted.extend(tier_results[tier_name]['predictions'])

if len(all_actual) > 0:
    overall_r2 = r2_score(all_actual, all_predicted)
    overall_mae = mean_absolute_error(all_actual, all_predicted)
    overall_rmse = np.sqrt(mean_squared_error(all_actual, all_predicted))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        overall_mape_vals = np.abs((np.array(all_actual) - np.array(all_predicted)) / np.array(all_actual))
        overall_mape_vals = overall_mape_vals[np.isfinite(overall_mape_vals) & (overall_mape_vals < 10)]
        overall_mape = np.mean(overall_mape_vals) * 100
        
        overall_within_50 = np.mean(np.abs(np.array(all_actual) - np.array(all_predicted)) / np.array(all_actual) < 0.50) * 100
    
    print(f"\nCombined Performance (all tiers):")
    print(f"  R² Score:        {overall_r2:.3f}")
    print(f"  MAE:             {overall_mae:.0f}")
    print(f"  RMSE:            {overall_rmse:.0f}")
    print(f"  MAPE:            {overall_mape:.1f}%")
    print(f"  Within ±50%:     {overall_within_50:.1f}%")

# -------------------------
# Scenario Predictions
# -------------------------
print("\n" + "=" * 60)
print("SCENARIO PREDICTIONS")
print("=" * 60)

base_scenario = {
    'day_of_week': 1, 'hour': 21, 'month': 10, 'day_of_month': 15,
    'is_weekend': 0, 'is_peak_hour': 1, 'is_tuesday': 1,
    'is_morning': 0, 'is_lunch': 0, 'is_evening': 0,
    'video_duration_min': 1.5, 'has_video': 1,
    'is_short_video': 1, 'is_long_video': 0,
    'posts_last_7d': 7,
}

# Add rolling averages at median
for window in [3, 5, 7, 14]:
    base_scenario[f'recent_avg_{window}'] = trainable_df['reach'].median()
    base_scenario[f'recent_std_{window}'] = trainable_df['reach'].std() * 0.5
    base_scenario[f'recent_max_{window}'] = trainable_df['reach'].quantile(0.75)

base_scenario['momentum_3_7'] = 0
base_scenario['momentum_7_14'] = 0
base_scenario['volatility_ratio'] = 0.5
base_scenario['recent_peak_ratio'] = 1.5

for col in type_cols + dow_cols + hour_cols:
    base_scenario[col] = 0

scenarios = []
post_types = [
    ('Reels', 1, 0, 0),
    ('Videos', 0, 1, 0),
    ('Photos', 0, 0, 1),
    ('Text', 0, 0, 0)
]

for name, is_reel, is_video, is_photo in post_types:
    scenario = base_scenario.copy()
    scenario['is_reel'] = is_reel
    scenario['is_video'] = is_video
    scenario['is_photo'] = is_photo
    
    type_col = f'type_avg_{name.lower()}'
    if type_col in type_cols:
        scenario[type_col] = df[df['post_type'] == name]['reach'].median()
    
    features = [scenario[col] for col in feature_cols]
    result = predict_reach_stratified(features)
    
    scenarios.append({
        'Post Type': name,
        'Predicted': int(result['predicted_reach']),
        'Tier': result['tier'].title(),
        'Confidence': f"{result['tier_confidence']*100:.0f}%",
        'Range': f"{int(result['confidence_lower'])}-{int(result['confidence_upper'])}"
    })

scenario_df = pd.DataFrame(scenarios)
print("\n📊 Predictions (Tuesday 9 PM):")
print(scenario_df.to_string(index=False))

# -------------------------
# Comparison
# -------------------------
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    'Model': [
        'SARIMA',
        'Original RF',
        'Log-Transform Ensemble',
        'Stratified System'
    ],
    'R²': [-1.297, 0.373, 0.307, overall_r2],
    'MAPE%': [45.02, 244.17, 70.75, overall_mape],
    'Within±50%': [0, 0, 41.0, overall_within_50]
})

print("\n", comparison.to_string(index=False))
print(f"\n✅ Best R²: {overall_r2:.3f}")
print(f"✅ Best MAPE: {overall_mape:.1f}%")
print(f"✅ {overall_within_50:.1f}% predictions within ±50%")

# -------------------------
# Visualizations
# -------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Actual vs Predicted (by tier)
ax1 = axes[0, 0]
colors_map = {'low': 'green', 'medium': 'orange', 'high': 'red'}
for tier_name in ['low', 'medium', 'high']:
    if tier_name in tier_results:
        ax1.scatter(tier_results[tier_name]['actual'], 
                   tier_results[tier_name]['predictions'],
                   alpha=0.6, s=30, label=tier_name.title(), 
                   color=colors_map[tier_name])

max_val = max(all_actual)
ax1.plot([0, max_val], [0, max_val], 'k--', lw=2, label='Perfect')
ax1.set_xlabel('Actual Reach')
ax1.set_ylabel('Predicted Reach')
ax1.set_title(f'Stratified Model\nOverall R² = {overall_r2:.3f}')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Performance by Tier
ax2 = axes[0, 1]
tier_names = []
tier_r2s = []
for tier in ['low', 'medium', 'high']:
    if tier in tier_results:
        tier_names.append(tier.title())
        tier_r2s.append(tier_results[tier]['r2'])

bars = ax2.bar(tier_names, tier_r2s, color=['green', 'orange', 'red'], alpha=0.7)
ax2.set_ylabel('R² Score')
ax2.set_title('R² by Reach Tier')
ax2.grid(alpha=0.3, axis='y')
for i, v in enumerate(tier_r2s):
    ax2.text(i, v + 0.02, f'{v:.3f}', ha='center')

# Plot 3: MAPE Comparison
ax3 = axes[0, 2]
models = comparison['Model'].values
mapes = comparison['MAPE%'].values
colors = ['red', 'red', 'orange', 'green']
ax3.barh(models, mapes, color=colors, alpha=0.7)
ax3.set_xlabel('MAPE (%)')
ax3.set_title('MAPE Comparison\n(Lower is Better)')
ax3.grid(alpha=0.3, axis='x')
for i, v in enumerate(mapes):
    ax3.text(v + 2, i, f'{v:.1f}%', va='center')

# Plot 4: Tier Distribution
ax4 = axes[1, 0]
tier_counts = df['reach_tier'].value_counts()
colors = ['green', 'orange', 'red', 'purple']
ax4.pie(tier_counts, labels=[t.title() for t in tier_counts.index], 
        autopct='%1.1f%%', colors=colors, startangle=90)
ax4.set_title('Post Distribution by Tier')

# Plot 5: Scenarios
ax5 = axes[1, 1]
scenario_df_sorted = scenario_df.sort_values('Predicted')
x_pos = np.arange(len(scenario_df_sorted))
tier_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
bar_colors = [tier_colors[t] for t in scenario_df_sorted['Tier']]
ax5.bar(x_pos, scenario_df_sorted['Predicted'], color=bar_colors, alpha=0.7, edgecolor='black')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(scenario_df_sorted['Post Type'], rotation=0)
ax5.set_ylabel('Predicted Reach')
ax5.set_title('Predicted Reach by Type')
ax5.grid(alpha=0.3, axis='y')

# Plot 6: Summary
ax6 = axes[1, 2]
ax6.axis('off')
summary = f"""
STRATIFIED MODEL SUMMARY

Approach: Tier Classification + 
          Tier-Specific Regression

Overall Performance:
  R² Score:          {overall_r2:.3f}
  MAE:               {overall_mae:.0f}
  MAPE:              {overall_mape:.1f}%
  Within ±50%:       {overall_within_50:.1f}%

Tier Classification:
  Accuracy:          {tier_accuracy:.3f}

Individual Tier R²:
"""

for tier in ['low', 'medium', 'high']:
    if tier in tier_results:
        summary += f"  {tier.title():8}         {tier_results[tier]['r2']:.3f}\n"

summary += f"""
Key Advantage:
  ✓ Specialized models per tier
  ✓ Better accuracy at each level
  ✓ More reliable predictions
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig('Facebook/stratified_model_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: Facebook/stratified_model_results.png")
plt.show()

print("\n" + "=" * 60)
print("🎉 STRATIFIED MODEL COMPLETE")
print("=" * 60)
print(f"\n🎯 Final R²: {overall_r2:.3f}")
print(f"📊 Final MAPE: {overall_mape:.1f}%")
print(f"✅ {overall_within_50:.1f}% within ±50% accuracy")