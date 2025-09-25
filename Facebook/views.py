import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter

# Load the data
df = pd.read_csv("Facebook/1_year_data.csv")
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
df['month'] = df['publish_time'].dt.month_name()
df_sorted = df.sort_values('publish_time')
df_sorted['clipped_seconds_viewed'] = df_sorted['seconds_viewed'].clip(upper=df['seconds_viewed'].quantile(0.95))
df_sorted['rolling_avg'] = df_sorted['clipped_seconds_viewed'].rolling(window=5).mean()

sns.set(style="whitegrid")

# --- Plot 2: Box Plot (Zoomed) ---
plt.figure(figsize=(10, 5))
sns.boxplot(x='month', y='seconds_viewed', data=df_sorted, order=[
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'])
plt.title('Seconds Viewed Distribution by Month')
plt.ylim(0, df_sorted['seconds_viewed'].quantile(0.95))  # Limit to reduce outliers
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Plot 3: Bar Plot (Zoomed) using median instead of mean ---
plt.figure(figsize=(6, 4))
median_seconds = df_sorted.groupby('post_type')['seconds_viewed'].median().sort_values()
median_seconds.plot(kind='bar', color='skyblue')
plt.title('Median Seconds Viewed by Post Type')
plt.ylabel('Median Seconds Viewed')
plt.ylim(0, df_sorted['seconds_viewed'].quantile(0.95))  # Zoom in
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Plot 4: Time Series (Zoomed and Lengthened) ---
plt.figure(figsize=(40, 7))  # Wider and taller
plt.plot(df_sorted['publish_time'], df_sorted['clipped_seconds_viewed'], alpha=0.8, label='Clipped (95th percentile)', color='blue')
plt.plot(df_sorted['publish_time'], df_sorted['rolling_avg'], color='orange', linewidth=2, label='5-Point Rolling Avg')
plt.title("Seconds Viewed Over Time (Clipped & Smoothed)", fontsize=16)
plt.xlabel("Publish Date", fontsize=12)
plt.ylabel("Seconds Viewed", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

# Group by post_type and sum reactions
reactions_by_post_type = df.groupby('post_type')['reactions'].sum().sort_values()

# Plot total reactions by post_type
plt.figure(figsize=(8, 5))
reactions_by_post_type.plot(kind='bar', color='teal')
plt.title('Total Reactions by Post Type')
plt.ylabel('Total Reactions')
plt.xlabel('Post Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
