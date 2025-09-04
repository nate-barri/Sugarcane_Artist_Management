import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("1_year_data.csv")

# Convert publish_time to datetime
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

# Drop rows with invalid or missing dates
df = df.dropna(subset=['publish_time'])

# Create a new column for the day of the week
df['day_name'] = df['publish_time'].dt.day_name()

# Convert reach to numeric (if not already)
df['reach'] = pd.to_numeric(df['reach'], errors='coerce')

# Group by day of the week and calculate the average reach
avg_reach = df.groupby('day_name')['reach'].mean().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])

# Plot the average reach
plt.figure(figsize=(8, 5))
avg_reach.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Reach by Day of Week')
plt.ylabel('Average Reach')
plt.xlabel('Day of Week')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
