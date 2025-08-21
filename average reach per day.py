import pandas as pd
import matplotlib.pyplot as plt

# Load all CSV files into a list of DataFrames
files = [
    "Jan-01-2024_Apr-01-2024_1162450548895235.csv",
    "Apr-02-2024_Jul-02-2024_665360019788100.csv",
    "Jul-03-2024_Sep-03-2024_1203413187991881.csv",
    "Sep-04-2024_Nov-04-2024_1223451109303636.csv",
    "Nov-05-2024_Jan-05-2025_1016476477111055 .csv"
]

# Read and combine all files
dfs = [pd.read_csv(file) for file in files]
df = pd.concat(dfs, ignore_index=True)

# Parse the 'Publish time' column to datetime
df['Publish time'] = pd.to_datetime(df['Publish time'], errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=['Publish time'])

# Create a day name column
df['day_name'] = df['Publish time'].dt.day_name()

# Convert reach column to numeric
df['Reach'] = pd.to_numeric(df['Reach'], errors='coerce')

# Group by weekday and calculate average reach
avg_reach = df.groupby('day_name')['Reach'].mean().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])

# Plotting
plt.figure(figsize=(8, 5))
avg_reach.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Reach by Day of Week')
plt.ylabel('Average Reach')
plt.xlabel('Day of Week')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
