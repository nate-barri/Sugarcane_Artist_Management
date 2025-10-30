import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tkinter
from datetime import datetime

# Set the style for better visualization
sns.set_style("whitegrid")

# Load the data
df = pd.read_csv("Facebook/1_year_data.csv")

# Convert publish_time to datetime for proper time series analysis
df['publish_datetime'] = pd.to_datetime(df['publish_time'])
df = df.sort_values('publish_datetime')

# Extract hour of day from publish time
df['hour'] = df['publish_datetime'].dt.hour
df['day_of_week'] = df['publish_datetime'].dt.day_name()

# Function to determine the best times to post based on engagement metrics
def analyze_best_posting_times():
    print("\n===== BEST TIME TO POST ANALYSIS =====\n")
    
    # Create metrics for engagement (you can adjust weights based on what's most important)
    df['engagement_score'] = (
        (df['reactions'] / df['reactions'].max()) * 0.4 +  # 40% weight to reactions
        (df['seconds_viewed'] / df['seconds_viewed'].max()) * 0.4 +  # 40% weight to views
        (df['comments'] / df['comments'].max() if df['comments'].max() > 0 else 0) * 0.1 +  # 10% weight to comments
        (df['shares'] / df['shares'].max() if df['shares'].max() > 0 else 0) * 0.1  # 10% weight to shares
    )
    
    # === ANALYSIS BY HOUR OF DAY ===
    hourly_performance = df.groupby('hour').agg({
        'reactions': 'mean',
        'seconds_viewed': 'mean',
        'comments': 'mean',
        'shares': 'mean',
        'engagement_score': 'mean',
        'post_type': 'count'  # To see post count per hour
    }).rename(columns={'post_type': 'post_count'})
    
    # Find the best hours by engagement score
    best_hours = hourly_performance.sort_values('engagement_score', ascending=False)
    
    # Define time periods for easier interpretation
    def categorize_time(hour):
        if 5 <= hour <= 8:
            return "Early Morning (5-8 AM)"
        elif 9 <= hour <= 11:
            return "Morning (9-11 AM)"
        elif 12 <= hour <= 14:
            return "Midday (12-2 PM)"
        elif 15 <= hour <= 17:
            return "Afternoon (3-5 PM)"
        elif 18 <= hour <= 20:
            return "Evening (6-8 PM)"
        elif 21 <= hour <= 23:
            return "Night (9-11 PM)"
        else:
            return "Late Night/Early AM (12-4 AM)"
            
    df['time_period'] = df['hour'].apply(categorize_time)
    
    # Calculate performance by time period
    time_period_performance = df.groupby('time_period').agg({
        'reactions': 'mean',
        'seconds_viewed': 'mean',
        'comments': 'mean',
        'shares': 'mean',
        'engagement_score': 'mean',
        'post_type': 'count'
    }).rename(columns={'post_type': 'post_count'})
    
    # Top time periods
    best_periods = time_period_performance.sort_values('engagement_score', ascending=False)
    
    # === ANALYSIS BY DAY OF WEEK ===
    # Order days properly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=day_order, ordered=True)
    
    weekday_performance = df.groupby('day_of_week').agg({
        'reactions': 'mean',
        'seconds_viewed': 'mean',
        'comments': 'mean',
        'shares': 'mean',
        'engagement_score': 'mean',
        'post_type': 'count'
    }).rename(columns={'post_type': 'post_count'})
    
    # === HOUR + DAY COMBINED ANALYSIS ===
    # Create a heatmap-ready pivot table
    heatmap_data = df.pivot_table(
        index='day_of_week', 
        columns='hour', 
        values='engagement_score',
        aggfunc='mean'
    )
    
    # === VISUALIZATIONS AND REPORTING ===
    # Plot 1: Hour of day performance
    plt.figure(figsize=(14, 7))
    
    # Sub-plot 1: Engagement score by hour
    plt.subplot(1, 2, 1)
    ax = sns.barplot(x=hourly_performance.index, y=hourly_performance['engagement_score'], color="#5cb85c")
    plt.title('Average Engagement Score by Hour of Day', fontsize=16)
    plt.xlabel('Hour of Day (24hr)', fontsize=12)
    plt.ylabel('Engagement Score', fontsize=12)
    plt.xticks(rotation=0)
    
    # Add data labels to the bars
    for i, v in enumerate(hourly_performance['engagement_score']):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
    
    # Add post count as text below
    for i, v in enumerate(hourly_performance['post_count']):
        ax.text(i, -0.05, f"n={v}", ha='center', fontsize=8, color='gray')
    
    # Sub-plot 2: Time period performance
    plt.subplot(1, 2, 2)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=best_periods.index, y=best_periods['engagement_score'], color="#3498db")
    plt.title('Average Engagement by Time Period', fontsize=16)
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Engagement Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add data labels
    for i, v in enumerate(best_periods['engagement_score']):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
    
    # Add post count as text
    for i, v in enumerate(best_periods['post_count']):
        ax.text(i, -0.05, f"n={v}", ha='center', fontsize=8, color='gray')
        
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Day of week performance
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=weekday_performance.index, y=weekday_performance['engagement_score'], color="#ff7f0e")
    plt.title('Average Engagement by Day of Week', fontsize=16)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Engagement Score', fontsize=12)
    
    # Add data labels
    for i, v in enumerate(weekday_performance['engagement_score']):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
    
    # Add post count as text
    for i, v in enumerate(weekday_performance['post_count']):
        ax.text(i, -0.05, f"n={v}", ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Heatmap for day+hour combinations
    plt.figure(figsize=(16, 8))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.5)
    plt.title('Engagement Score by Day and Hour', fontsize=18)
    plt.xlabel('Hour of Day (24hr)', fontsize=14)
    plt.ylabel('Day of Week', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # REPORTING THE RESULTS
    print("\n----- TOP 3 BEST HOURS TO POST -----")
    print(best_hours[['engagement_score', 'reactions', 'seconds_viewed', 'post_count']].head(3))
    
    print("\n----- TOP TIME PERIODS TO POST -----")
    print(best_periods[['engagement_score', 'reactions', 'seconds_viewed', 'post_count']].head(3))
    
    print("\n----- BEST DAYS OF WEEK TO POST -----")
    print(weekday_performance[['engagement_score', 'reactions', 'seconds_viewed', 'post_count']].sort_values('engagement_score', ascending=False).head(3))
    
    # Find the absolute best day-hour combination
    day_hour_performance = df.groupby(['day_of_week', 'hour']).agg({
        'engagement_score': 'mean', 
        'post_type': 'count'
    }).rename(columns={'post_type': 'post_count'})
    
    best_combinations = day_hour_performance.sort_values('engagement_score', ascending=False)
    
    print("\n----- BEST DAY-HOUR COMBINATIONS -----")
    top_combinations = best_combinations.head(5)
    for idx, row in top_combinations.iterrows():
        day, hour = idx
        period = categorize_time(hour)
        print(f"{day} at {hour}:00 ({period}): Score {row['engagement_score']:.2f} (based on {row['post_count']} posts)")
    
    # FINAL RECOMMENDATION
    print("\n===== RECOMMENDATION =====")
    # Get the single best time to post
    if len(top_combinations) > 0:
        best_day, best_hour = top_combinations.index[0]
        best_period = categorize_time(best_hour)
        print(f" OPTIMAL POSTING TIME: {best_day} at {best_hour}:00 ({best_period})")
    
    # Get the best day
    best_day = weekday_performance['engagement_score'].idxmax()
    print(f" BEST DAY: {best_day}")
    
    # Get the best time period regardless of day
    best_period = best_periods.index[0]
    print(f" BEST TIME PERIOD: {best_period}")
    
    # Check if we have enough data
    low_data_warning = []
    if hourly_performance['post_count'].min() < 3:
        low_data_warning.append("Some hours have very few posts (<3), which may affect reliability")
    if weekday_performance['post_count'].min() < 5:
        low_data_warning.append("Some days have very few posts (<5), which may affect reliability")
        
    if low_data_warning:
        print("\n RELIABILITY WARNING:")
        for warning in low_data_warning:
            print(f"- {warning}")
        print("Consider testing these recommendations to confirm their effectiveness.")

# Run the best time to post analysis
analyze_best_posting_times()