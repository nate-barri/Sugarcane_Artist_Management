import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Read CSV
df = pd.read_csv("YOUTUBE - SUGARCANE CONTENT DATA.csv")

# --- Data Preprocessing ---
# Ensure datetime format
df["Video publish time"] = pd.to_datetime(df["Video publish time"], errors="coerce")
df["DayOfWeek"] = df["Video publish time"].dt.day_name()

# --- 1. Top 10 videos by views ---
top10 = df.nlargest(10, "Views")[["Video title", "Views"]]
plt.figure(figsize=(10,5))
sns.barplot(x="Views", y="Video title", data=top10)
plt.title("Top 10 Videos by Views")
plt.tight_layout()
plt.show()

# --- 2. Views over time ---
views_over_time = df.groupby(df["Video publish time"].dt.date)["Views"].sum()
plt.figure(figsize=(10,5))
views_over_time.plot()
plt.title("Views Over Time")
plt.xlabel("Date")
plt.ylabel("Views")
plt.show()

# --- 3. Average views ---
avg_views = df["Views"].mean()
print(f"Average views per video: {avg_views:.2f}")

# --- 4. Average views by day of week ---
avg_views_day = df.groupby("DayOfWeek")["Views"].mean().sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=avg_views_day.index, y=avg_views_day.values)
plt.title("Average Views by Day of Week")
plt.show()

# --- 5. Best day to post ---
best_day = avg_views_day.idxmax()
print(f"Best day to post (based on avg views): {best_day}")

# --- 6. Video performance clusters with labels ---
features = ["Impressions", "Impressions click-through rate (%)", 
            "Average views per viewer", "New viewers", 
            "Subscribers gained", "Likes", "Shares", 
            "Comments added", "Views", "Watch time (hours)"]
X = df[features].fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Map cluster numbers to performance labels based on avg Views
cluster_avg_views = df.groupby("Cluster")["Views"].mean().sort_values()
cluster_label_map = {cluster_avg_views.index[0]: "Low",
                     cluster_avg_views.index[1]: "Mid",
                     cluster_avg_views.index[2]: "High"}

df["Performance Category"] = df["Cluster"].map(cluster_label_map)

# Scatter plot with labeled performance
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="Impressions", y="Views", hue="Performance Category", palette="viridis")
plt.title("Video Performance Clusters (Labeled)")
plt.show()

# Show videos in each category
for category in ["Low", "Mid", "High"]:
    print(f"\n{category} Performing Videos:")
    videos = df.loc[df["Performance Category"] == category, "Video title"].tolist()
    for v in videos:
        print(f" - {v}")

# --- 7. Feature importance for predicting views ---
X = df[features].drop(columns=["Views"])
y = df["Views"]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature Importance for Predicting Views")
plt.show()
