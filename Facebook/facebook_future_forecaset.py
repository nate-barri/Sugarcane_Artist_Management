"""
Facebook Reach Analysis - CONTENT & ENGAGEMENT CHARACTERIZATION
Identifies high-reach post characteristics and timing patterns
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# -------------------------
# Settings
# -------------------------
CSV_PATH = "Facebook/facebook_data_set.csv"

print("=" * 70)
print("FACEBOOK REACH ANALYSIS - HIGH-REACH CHARACTERIZATION")
print("What makes posts successful & when timing works")
print("=" * 70)

# -------------------------
# Load & Clean Data
# -------------------------
df = pd.read_csv(CSV_PATH, parse_dates=['publish_time'])
df = df.sort_values('publish_time').reset_index(drop=True)

print(f"\nOriginal dataset: {len(df)} posts")
df = df.dropna(subset=['reach'])
print(f"After removing NaN reach: {len(df)} posts")

# -------------------------
# Define High-Reach Threshold
# -------------------------
print("\n" + "=" * 70)
print("HIGH-REACH POST CLASSIFICATION")
print("=" * 70)

# High-reach = top 25% of posts
high_reach_threshold = df['reach'].quantile(0.75)
low_reach_threshold = df['reach'].quantile(0.25)

df['reach_level'] = 'Medium'
df.loc[df['reach'] >= high_reach_threshold, 'reach_level'] = 'High'
df.loc[df['reach'] <= low_reach_threshold, 'reach_level'] = 'Low'

high_reach_posts = df[df['reach_level'] == 'High']
low_reach_posts = df[df['reach_level'] == 'Low']

print(f"\nReach thresholds:")
print(f"   Low:    < {low_reach_threshold:,.0f}")
print(f"   Medium: {low_reach_threshold:,.0f} - {high_reach_threshold:,.0f}")
print(f"   High:   > {high_reach_threshold:,.0f}")

print(f"\nDistribution:")
print(f"   High reach posts: {len(high_reach_posts)} ({len(high_reach_posts)/len(df)*100:.1f}%)")
print(f"   Low reach posts:  {len(low_reach_posts)} ({len(low_reach_posts)/len(df)*100:.1f}%)")
print(f"   Mean reach (High): {high_reach_posts['reach'].mean():,.0f}")
print(f"   Mean reach (Low):  {low_reach_posts['reach'].mean():,.0f}")

# -------------------------
# 1. POST TYPE ANALYSIS
# -------------------------
print("\n" + "=" * 70)
print("1. POST TYPE CHARACTERISTICS")
print("=" * 70)

type_analysis = []
for post_type in df['post_type'].unique():
    type_df = df[df['post_type'] == post_type]
    if len(type_df) < 10:
        continue
    
    high_reach_pct = (type_df['reach_level'] == 'High').sum() / len(type_df) * 100
    mean_reach = type_df['reach'].mean()
    median_reach = type_df['reach'].median()
    
    type_analysis.append({
        'Post Type': post_type,
        'Total Posts': len(type_df),
        '% High-Reach': high_reach_pct,
        'Mean Reach': mean_reach,
        'Median Reach': median_reach
    })

type_df_analysis = pd.DataFrame(type_analysis).sort_values('% High-Reach', ascending=False)

print("\n📊 Post Type Performance:")
print(type_df_analysis.to_string(index=False))

print("\n🎯 FINDING: High-Reach Characteristics by Type")
for _, row in type_df_analysis.head(3).iterrows():
    print(f"\n   {row['Post Type'].upper()}:")
    print(f"      • {row['% High-Reach']:.1f}% of posts achieve high reach")
    print(f"      • Average reach: {row['Mean Reach']:,.0f}")
    print(f"      • Median reach: {row['Median Reach']:,.0f}")
    print(f"      → Use this format {row['% High-Reach']:.0f}% of the time")

# -------------------------
# 2. TIMING ANALYSIS
# -------------------------
print("\n" + "=" * 70)
print("2. OPTIMAL TIMING PATTERNS")
print("=" * 70)

df['day_of_week'] = df['publish_time'].dt.dayofweek
df['hour'] = df['publish_time'].dt.hour
df['day_name'] = df['publish_time'].dt.day_name()
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Day analysis
print("\n📅 BEST DAYS (for high reach):")
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_stats = []

for day in day_order:
    day_df = df[df['day_name'] == day]
    if len(day_df) == 0:
        continue
    
    high_reach_pct = (day_df['reach_level'] == 'High').sum() / len(day_df) * 100
    mean_reach = day_df['reach'].mean()
    
    day_stats.append({
        'Day': day,
        'Posts': len(day_df),
        '% High-Reach': high_reach_pct,
        'Avg Reach': mean_reach
    })

day_stats_df = pd.DataFrame(day_stats).sort_values('% High-Reach', ascending=False)
print(day_stats_df.to_string(index=False))

# Hour analysis
print("\n⏰ BEST HOURS (for high reach):")
hour_stats = []

for hour in range(24):
    hour_df = df[df['hour'] == hour]
    if len(hour_df) < 5:
        continue
    
    high_reach_pct = (hour_df['reach_level'] == 'High').sum() / len(hour_df) * 100
    mean_reach = hour_df['reach'].mean()
    
    hour_stats.append({
        'Hour': f"{hour:02d}:00",
        'Posts': len(hour_df),
        '% High-Reach': high_reach_pct,
        'Avg Reach': mean_reach
    })

hour_stats_df = pd.DataFrame(hour_stats).sort_values('% High-Reach', ascending=False)
print(hour_stats_df.head(10).to_string(index=False))

# Day + Hour combinations
print("\n🎯 BEST DAY + HOUR COMBINATIONS:")
combo_stats = []

for day in day_order:
    for hour in range(24):
        combo_df = df[(df['day_name'] == day) & (df['hour'] == hour)]
        if len(combo_df) < 3:
            continue
        
        high_reach_pct = (combo_df['reach_level'] == 'High').sum() / len(combo_df) * 100
        mean_reach = combo_df['reach'].mean()
        
        combo_stats.append({
            'Day+Hour': f"{day} {hour:02d}:00",
            'Posts': len(combo_df),
            '% High-Reach': high_reach_pct,
            'Avg Reach': mean_reach
        })

combo_stats_df = pd.DataFrame(combo_stats).sort_values('% High-Reach', ascending=False)
print(combo_stats_df.head(12).to_string(index=False))

# -------------------------
# 3. POST TYPE + TIMING INTERACTION
# -------------------------
print("\n" + "=" * 70)
print("3. BEST TIMING FOR EACH POST TYPE")
print("=" * 70)

for post_type in type_df_analysis['Post Type'].head(4):
    type_df = df[df['post_type'] == post_type]
    
    print(f"\n{post_type.upper()}:")
    
    # Best days for this type
    day_perf = []
    for day in day_order:
        day_type_df = type_df[type_df['day_name'] == day]
        if len(day_type_df) < 3:
            continue
        
        high_pct = (day_type_df['reach_level'] == 'High').sum() / len(day_type_df) * 100
        mean_reach = day_type_df['reach'].mean()
        
        day_perf.append((day, high_pct, mean_reach, len(day_type_df)))
    
    day_perf.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   Best days:")
    for day, high_pct, mean_reach, count in day_perf[:3]:
        print(f"      • {day:10} - {high_pct:.0f}% high-reach | Avg: {mean_reach:,.0f} | Posts: {count}")
    
    # Best hours for this type
    hour_perf = []
    for hour in range(24):
        hour_type_df = type_df[type_df['hour'] == hour]
        if len(hour_type_df) < 2:
            continue
        
        high_pct = (hour_type_df['reach_level'] == 'High').sum() / len(hour_type_df) * 100
        mean_reach = hour_type_df['reach'].mean()
        
        hour_perf.append((hour, high_pct, mean_reach, len(hour_type_df)))
    
    hour_perf.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   Best hours:")
    for hour, high_pct, mean_reach, count in hour_perf[:3]:
        print(f"      • {hour:02d}:00 - {high_pct:.0f}% high-reach | Avg: {mean_reach:,.0f} | Posts: {count}")

# -------------------------
# 4. REACH VARIABILITY BY CONTEXT
# -------------------------
print("\n" + "=" * 70)
print("4. CONSISTENCY ANALYSIS (When is reach predictable?)")
print("=" * 70)

# Calculate coefficient of variation for different segments
segments = []

for post_type in df['post_type'].unique():
    type_df = df[df['post_type'] == post_type]
    if len(type_df) < 10:
        continue
    
    cv = type_df['reach'].std() / type_df['reach'].mean()
    segments.append(('Post Type: ' + post_type, cv, len(type_df)))

for day in day_order:
    day_df = df[df['day_name'] == day]
    if len(day_df) < 10:
        continue
    
    cv = day_df['reach'].std() / day_df['reach'].mean()
    segments.append(('Day: ' + day, cv, len(day_df)))

segments.sort(key=lambda x: x[1])

print("\n📊 Consistency (Lower CV = More Predictable):")
print(f"\n{'Segment':<30} {'Variability (CV)':<20} {'Sample Size'}")
print("-" * 60)
for segment, cv, count in segments:
    predictability = "HIGH" if cv < 2.0 else "MEDIUM" if cv < 4.0 else "LOW"
    print(f"{segment:<30} {cv:.2f} ({predictability})        {count}")

# -------------------------
# 5. REGRESSION METRICS (Mean Reach Prediction)
# -------------------------
print("\n" + "=" * 70)
print("5. REGRESSION METRICS - PREDICTING MEAN REACH BY SEGMENT")
print("=" * 70)

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Build predictions based on segment means
segment_predictions = []

# Post type predictions
for post_type in df['post_type'].unique():
    type_df = df[df['post_type'] == post_type]
    if len(type_df) < 10:
        continue
    
    mean_reach = type_df['reach'].mean()
    actual_reaches = type_df['reach'].values
    predicted_reaches = np.full_like(actual_reaches, mean_reach, dtype=float)
    
    segment_predictions.extend(list(zip(actual_reaches, predicted_reaches)))

# Day predictions
for day in day_order:
    day_df = df[df['day_name'] == day]
    if len(day_df) < 10:
        continue
    
    mean_reach = day_df['reach'].mean()
    actual_reaches = day_df['reach'].values
    predicted_reaches = np.full_like(actual_reaches, mean_reach, dtype=float)
    
    segment_predictions.extend(list(zip(actual_reaches, predicted_reaches)))

# Hour predictions
for hour in range(24):
    hour_df = df[df['hour'] == hour]
    if len(hour_df) < 5:
        continue
    
    mean_reach = hour_df['reach'].mean()
    actual_reaches = hour_df['reach'].values
    predicted_reaches = np.full_like(actual_reaches, mean_reach, dtype=float)
    
    segment_predictions.extend(list(zip(actual_reaches, predicted_reaches)))

segment_predictions = np.array(segment_predictions)
y_actual = segment_predictions[:, 0]
y_predicted = segment_predictions[:, 1]

# Calculate metrics
mae = mean_absolute_error(y_actual, y_predicted)
rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
mad = median_absolute_error(y_actual, y_predicted)
r2 = r2_score(y_actual, y_predicted)

# MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_actual - y_predicted) / np.where(y_actual == 0, 1, y_actual))) * 100

# MASE (Mean Absolute Scaled Error) - using naive forecast baseline
naive_mae = np.mean(np.abs(y_actual - np.mean(y_actual)))
mase = mae / naive_mae if naive_mae != 0 else np.inf

print(f"\n📊 SEGMENT-BASED REACH PREDICTION METRICS:")
print(f"   R² Score:              {r2*100:.2f}%  (variance explained)")
print(f"   MAE:                   {mae:,.0f} reach")
print(f"   MAE (% of mean):       {mae/y_actual.mean()*100:.1f}%")
print(f"   RMSE:                  {rmse:,.0f} reach")
print(f"   RMSE (% of mean):      {rmse/y_actual.mean()*100:.1f}%")
print(f"   MAD:                   {mad:,.0f} reach")
print(f"   MAD (% of median):     {mad/np.median(y_actual)*100:.1f}%")
print(f"   MAPE:                  {mape:.2f}%")
print(f"   MASE:                  {mase:.4f}")

print(f"\n   Interpretation:")
if mase < 1:
    print(f"   ✓ Model is {(1-mase)*100:.1f}% better than naive baseline")
else:
    print(f"   ⚠ Model is {(mase-1)*100:.1f}% worse than naive baseline")

print(f"   ✓ Average prediction error: ±{mae:,.0f} reach ({mae/y_actual.mean()*100:.1f}%)")

# By segment type
print(f"\n📈 METRICS BY SEGMENT TYPE:")

# Post type
print(f"\n   POST TYPE PREDICTIONS:")
type_predictions = []
for post_type in df['post_type'].unique():
    type_df = df[df['post_type'] == post_type]
    if len(type_df) < 10:
        continue
    
    mean_reach = type_df['reach'].mean()
    actual_reaches = type_df['reach'].values
    predicted_reaches = np.full_like(actual_reaches, mean_reach, dtype=float)
    
    type_mae = mean_absolute_error(actual_reaches, predicted_reaches)
    type_mape = np.mean(np.abs((actual_reaches - predicted_reaches) / actual_reaches)) * 100
    type_r2 = r2_score(actual_reaches, predicted_reaches)
    
    type_predictions.append({
        'Post Type': post_type,
        'MAE': type_mae,
        'MAPE': type_mape,
        'R²': type_r2,
        'Samples': len(type_df)
    })

type_pred_df = pd.DataFrame(type_predictions).sort_values('MAE')
for _, row in type_pred_df.iterrows():
    print(f"      {row['Post Type']}: MAE={row['MAE']:,.0f} | MAPE={row['MAPE']:.1f}% | R²={row['R²']*100:.1f}% | n={row['Samples']}")

# Day
print(f"\n   DAY-OF-WEEK PREDICTIONS:")
day_predictions = []
for day in day_order:
    day_df = df[df['day_name'] == day]
    if len(day_df) < 10:
        continue
    
    mean_reach = day_df['reach'].mean()
    actual_reaches = day_df['reach'].values
    predicted_reaches = np.full_like(actual_reaches, mean_reach, dtype=float)
    
    day_mae = mean_absolute_error(actual_reaches, predicted_reaches)
    day_mape = np.mean(np.abs((actual_reaches - predicted_reaches) / actual_reaches)) * 100
    day_r2 = r2_score(actual_reaches, predicted_reaches)
    
    day_predictions.append({
        'Day': day,
        'MAE': day_mae,
        'MAPE': day_mape,
        'R²': day_r2,
        'Samples': len(day_df)
    })

day_pred_df = pd.DataFrame(day_predictions).sort_values('MAE')
for _, row in day_pred_df.iterrows():
    print(f"      {row['Day']}: MAE={row['MAE']:,.0f} | MAPE={row['MAPE']:.1f}% | R²={row['R²']*100:.1f}% | n={row['Samples']}")

# -------------------------
# 6. CLASSIFICATION METRICS (High vs Low Reach)
# -------------------------
print("\n" + "=" * 70)
print("6. CLASSIFICATION METRICS - PREDICTING HIGH VS LOW REACH")
print("=" * 70)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Classify each post by segment
y_true_class = []
y_pred_class = []

# Post type classification
for post_type in df['post_type'].unique():
    type_df = df[df['post_type'] == post_type]
    if len(type_df) < 10:
        continue
    
    high_pct = (type_df['reach_level'] == 'High').sum() / len(type_df)
    threshold = 0.5
    
    for reach, reach_level in zip(type_df['reach'], type_df['reach_level']):
        y_true_class.append(1 if reach_level == 'High' else 0)
        y_pred_class.append(1 if high_pct >= threshold else 0)

# Day classification
for day in day_order:
    day_df = df[df['day_name'] == day]
    if len(day_df) < 10:
        continue
    
    high_pct = (day_df['reach_level'] == 'High').sum() / len(day_df)
    threshold = 0.5
    
    for reach, reach_level in zip(day_df['reach'], day_df['reach_level']):
        y_true_class.append(1 if reach_level == 'High' else 0)
        y_pred_class.append(1 if high_pct >= threshold else 0)

y_true_class = np.array(y_true_class)
y_pred_class = np.array(y_pred_class)

class_accuracy = accuracy_score(y_true_class, y_pred_class)
class_precision = precision_score(y_true_class, y_pred_class, zero_division=0)
class_recall = recall_score(y_true_class, y_pred_class, zero_division=0)
class_f1 = f1_score(y_true_class, y_pred_class, zero_division=0)

print(f"\n📊 HIGH VS LOW REACH CLASSIFICATION:")
print(f"   Accuracy:  {class_accuracy*100:.1f}%")
print(f"   Precision: {class_precision*100:.1f}%")
print(f"   Recall:    {class_recall*100:.1f}%")
print(f"   F1-Score:  {class_f1*100:.1f}%")

baseline_class = max((y_true_class == 1).sum(), (y_true_class == 0).sum()) / len(y_true_class)
print(f"\n   Baseline (always predict majority): {baseline_class*100:.1f}%")
if class_accuracy > baseline_class:
    print(f"   ✓ Improvement over baseline: {(class_accuracy - baseline_class)*100:.1f}%")
else:
    print(f"   ⚠ Below baseline - not useful for classification")

# Sample summary table
print(f"\n📈 OVERALL METRICS SUMMARY:")
print(f"\n   {'Metric':<25} {'Value':<20} {'Interpretation'}")
print(f"   {'-'*60}")
print(f"   {'R² (Regression)':<25} {r2*100:.2f}%            {f'Explains {r2*100:.1f}% of variance' if r2 > 0 else 'Worse than mean baseline'}")
print(f"   {'MAE':<25} {mae:,.0f}           {f'±{mae/y_actual.mean()*100:.1f}% average error' if mae else 'Perfect'}")
print(f"   {'MAPE':<25} {mape:.2f}%            {f'Average {mape:.0f}% percentage error' if mape else 'Perfect'}")
print(f"   {'RMSE':<25} {rmse:,.0f}          {f'Penalizes large errors' if rmse else 'Perfect'}")
print(f"   {'MASE':<25} {mase:.4f}           {f'Better than baseline' if mase < 1 else f'Worse than baseline'}")
print(f"   {'Classification Acc':<25} {class_accuracy*100:.1f}%            {f'Better than baseline' if class_accuracy > baseline_class else f'Baseline: {baseline_class*100:.1f}%'}")

# Get best performers
best_type = type_df_analysis.iloc[0]
best_day = day_stats_df.iloc[0]
best_hour = hour_stats_df.iloc[0]
best_combo = combo_stats_df.iloc[0]

recommendations = f"""
🎯 PRIMARY STRATEGY:
   1. Focus on {best_type['Post Type']} posts
      • {best_type['% High-Reach']:.0f}% achieve high reach
      • Post {int(best_type['Total Posts']*0.25)} times per quarter minimum

   2. Post on {best_day['Day']} 
      • {best_day['% High-Reach']:.0f}% of posts reach high audience
      • Best avg reach: {best_day['Avg Reach']:,.0f}

   3. Optimal time: {best_hour['Hour']}
      • {best_hour['% High-Reach']:.0f}% high-reach rate at this hour

   4. Golden combination: {best_combo['Day+Hour']}
      • {best_combo['% High-Reach']:.0f}% achieve high reach
      • Average reach: {best_combo['Avg Reach']:,.0f}

📋 CONTENT STRATEGY:
   • High-reach posts are primarily {best_type['Post Type']}
   • Consistency is key - maintain regular posting schedule
   • Reach varies significantly - don't expect every post to perform the same
   
⚠️  LIMITATIONS:
   • This analysis describes what HAS worked
   • Facebook's algorithm changes over time
   • External factors (trends, seasonality) impact reach
   • Post content/quality matters more than timing alone
"""

print(recommendations)

# -------------------------
# VISUALIZATIONS
# -------------------------
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# Plot 1: Post type performance
ax1 = fig.add_subplot(gs[0, :2])
colors = ['#E1306C', '#4267B2', '#FFA500', '#90EE90']
bars = ax1.bar(type_df_analysis['Post Type'], type_df_analysis['% High-Reach'],
               color=colors[:len(type_df_analysis)], alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('% of Posts with High Reach', fontsize=12, fontweight='bold')
ax1.set_title('Post Type: Percentage Achieving High Reach', fontsize=13, fontweight='bold')
ax1.set_ylim([0, 100])
ax1.grid(alpha=0.3, axis='y')
for i, (idx, row) in enumerate(type_df_analysis.iterrows()):
    ax1.text(i, row['% High-Reach'] + 2, f"{row['% High-Reach']:.0f}%", 
             ha='center', fontsize=11, fontweight='bold')

# Plot 2: Post type distribution
ax2 = fig.add_subplot(gs[0, 2])
type_counts = df['post_type'].value_counts()
ax2.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
        colors=colors[:len(type_counts)], startangle=90)
ax2.set_title('Posts by Type', fontsize=13, fontweight='bold')

# Plot 3: Day of week analysis
ax3 = fig.add_subplot(gs[1, :2])
day_order_short = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
day_stats_ordered = day_stats_df.set_index('Day').reindex(day_order).reset_index()
bars = ax3.bar(range(len(day_stats_ordered)), day_stats_ordered['% High-Reach'],
               color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_xticks(range(len(day_stats_ordered)))
ax3.set_xticklabels(day_order_short, fontsize=11)
ax3.set_ylabel('% of Posts with High Reach', fontsize=12, fontweight='bold')
ax3.set_title('Best Days to Post', fontsize=13, fontweight='bold')
ax3.set_ylim([0, 100])
ax3.grid(alpha=0.3, axis='y')
for i, val in enumerate(day_stats_ordered['% High-Reach']):
    ax3.text(i, val + 2, f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')

# Plot 4: Reach distribution
ax4 = fig.add_subplot(gs[1, 2])
high_low_counts = [len(high_reach_posts), len(low_reach_posts)]
ax4.pie(high_low_counts, labels=['High Reach (Top 25%)', 'Low Reach (Bottom 25%)'],
        autopct='%1.1f%%', colors=['#90EE90', '#FF6B6B'], startangle=90)
ax4.set_title('Reach Distribution', fontsize=13, fontweight='bold')

# Plot 5: Hour analysis
ax5 = fig.add_subplot(gs[2, :])
hour_range = hour_stats_df['Hour'].str[:2].astype(int).values
hour_perf = hour_stats_df['% High-Reach'].values
ax5.plot(hour_range, hour_perf, marker='o', linewidth=2.5, markersize=8,
         color='darkgreen', markerfacecolor='lightgreen', markeredgecolor='darkgreen', markeredgewidth=2)
ax5.fill_between(hour_range, hour_perf, alpha=0.3, color='lightgreen')
ax5.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
ax5.set_ylabel('% Posts with High Reach', fontsize=12, fontweight='bold')
ax5.set_title('Best Hours to Post', fontsize=13, fontweight='bold')
ax5.set_xticks(range(0, 24, 2))
ax5.grid(alpha=0.3)

# Plot 7: Accuracy by segment
ax7 = fig.add_subplot(gs[3, :2])
segments_list = []
accuracy_list = []
sample_size_list = []

for _, row in type_df_analysis.iterrows():
    segments_list.append(f"{row['Post Type']}\n(n={row['Total Posts']})")
    accuracy_list.append(row['% High-Reach'])
    sample_size_list.append(row['Total Posts'])

for _, row in day_stats_df.head(4).iterrows():
    segments_list.append(f"{row['Day']}\n(n={row['Posts']})")
    accuracy_list.append(row['% High-Reach'])
    sample_size_list.append(row['Posts'])

colors_accuracy = ['#90EE90' if acc >= 40 else '#FFA500' if acc >= 30 else '#FF6B6B' for acc in accuracy_list]
bars = ax7.barh(range(len(segments_list)), accuracy_list, color=colors_accuracy, alpha=0.7, edgecolor='black', linewidth=1.5)
ax7.set_yticks(range(len(segments_list)))
ax7.set_yticklabels(segments_list, fontsize=9)
ax7.set_xlabel('% of Posts Achieving High Reach', fontsize=11, fontweight='bold')
ax7.set_title('Prediction Accuracy by Segment', fontsize=12, fontweight='bold')
ax7.set_xlim([0, 100])
ax7.axvline(33.33, color='red', linestyle='--', linewidth=2, label='Random Guessing (33%)')
ax7.grid(alpha=0.3, axis='x')
ax7.legend(fontsize=9)
for i, val in enumerate(accuracy_list):
    ax7.text(val + 1, i, f'{val:.0f}%', va='center', fontsize=9, fontweight='bold')

# Plot 8: Sample size reliability
ax8 = fig.add_subplot(gs[3, 2])
reliability_categories = ['Small\n(<10)', 'Medium\n(10-20)', 'Good\n(20-50)', 'High\n(50+)']
post_counts = [
    len([x for x in sample_size_list if x < 10]),
    len([x for x in sample_size_list if 10 <= x < 20]),
    len([x for x in sample_size_list if 20 <= x < 50]),
    len([x for x in sample_size_list if x >= 50])
]
colors_reliability = ['#FF6B6B', '#FFA500', '#FFD700', '#90EE90']
bars = ax8.bar(reliability_categories, post_counts, color=colors_reliability, alpha=0.7, edgecolor='black', linewidth=1.5)
ax8.set_ylabel('Number of Segments', fontsize=11, fontweight='bold')
ax8.set_title('Reliability Distribution', fontsize=12, fontweight='bold')
ax8.grid(alpha=0.3, axis='y')
for i, v in enumerate(post_counts):
    ax8.text(i, v + 0.1, str(v), ha='center', fontsize=11, fontweight='bold')

plt.suptitle('Facebook Reach Analysis - High-Reach Characteristics & Optimal Timing', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('Facebook/high_reach_characterization.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: Facebook/high_reach_characterization.png")
plt.show()

print("\n" + "=" * 70)
print("✅ CHARACTERIZATION ANALYSIS COMPLETE")
print("=" * 70)