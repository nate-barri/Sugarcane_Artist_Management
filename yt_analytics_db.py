import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import pairwise_distances

# --- Database connection parameters ---
db_params = {
    'dbname': 'test1',
    'user': 'postgres',
    'password': 'admin',
    'host': 'localhost',
    'port': '5432'
}

# --- Simple K-Medoids Implementation ---
def kmedoids(X, k, max_iter=300, random_state=None):
    np.random.seed(random_state)
    m = X.shape[0]
    # Initial medoid indices
    medoid_indices = np.random.choice(m, k, replace=False)
    medoids = X[medoid_indices]

    for _ in range(max_iter):
        # Assign clusters
        distances = pairwise_distances(X, medoids)
        labels = np.argmin(distances, axis=1)

        # Update medoids
        new_medoids = np.copy(medoids)
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                continue
            intra_distances = pairwise_distances(cluster_points, cluster_points)
            total_distances = intra_distances.sum(axis=1)
            medoid_idx = np.argmin(total_distances)
            new_medoids[i] = cluster_points[medoid_idx]

        if np.all(new_medoids == medoids):
            break
        medoids = new_medoids

    return labels, medoid_indices


# --- Load data from PostgreSQL ---
conn = psycopg2.connect(**db_params)
query = """
SELECT video_title, views, publish_day, publish_month, publish_year,
       impressions, impressions_ctr, avg_views_per_viewer, new_viewers,
       subscribers_gained, likes, shares, comments_added, watch_time_hours
FROM yt_video_etl
"""
df = pd.read_sql(query, conn)
conn.close()

# --- Clean publish date ---
df = df.dropna(subset=["publish_day", "publish_month", "publish_year"])
df["publish_day"] = df["publish_day"].astype(int)
df["publish_month"] = df["publish_month"].astype(int)
df["publish_year"] = df["publish_year"].astype(int)

df["Publish_Date"] = pd.to_datetime(
    df.rename(columns={
        "publish_year": "year",
        "publish_month": "month",
        "publish_day": "day"
    })[["year", "month", "day"]],
    errors="coerce"
)

# Day of week
df["DayOfWeek"] = df["Publish_Date"].dt.day_name()

# --- 3. Top 10 videos by views ---
top10 = df.nlargest(10, "views")[["video_title", "views"]]
plt.figure(figsize=(10,5))
sns.barplot(x="views", y="video_title", data=top10)
plt.title("Top 10 Videos by Views")
plt.tight_layout()
plt.show()

# --- 4. Views over time ---
views_over_time = df.groupby(df["Publish_Date"].dt.date)["views"].sum()
plt.figure(figsize=(10,5))
views_over_time.plot()
plt.title("Views Over Time")
plt.xlabel("Date")
plt.ylabel("Views")
plt.show()

# --- 5. Average views ---
avg_views = df["views"].mean()
print(f"Average views per video: {avg_views:.2f}")

# --- 6. Average views by day of week (in calendar order) ---
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

avg_views_day = (
    df.groupby("DayOfWeek")["views"]
      .mean()
      .reindex(day_order)  # reorder to calendar order
)

plt.figure(figsize=(8,5))
sns.barplot(x=avg_views_day.index, y=avg_views_day.values, order=day_order)
plt.title("Average Views by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Views")
plt.show()

# --- NEW SECTION: Mean vs Median comparison ---
views_day_stats = (
    df.groupby("DayOfWeek")["views"]
      .agg(["mean", "median"])
      .reindex(day_order)
)

views_day_stats.plot(kind="bar", figsize=(10,6))
plt.title("Mean vs Median Views by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Views")
plt.legend(["Mean", "Median"])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print best days by both metrics
best_day_mean = views_day_stats["mean"].idxmax()
best_day_median = views_day_stats["median"].idxmax()
print(f"📊 Best day by average (mean) views: {best_day_mean}")
print(f"📊 Best day by typical (median) views: {best_day_median}")# --- 6a. Average (Mean) views by day of week ---
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

mean_views_day = (
    df.groupby("DayOfWeek")["views"]
      .mean()
      .reindex(day_order)
)

plt.figure(figsize=(8,5))
sns.barplot(x=mean_views_day.index, y=mean_views_day.values, order=day_order)
plt.title("Average (Mean) Views by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Mean Views")
plt.show()

# --- 6b. Median views by day of week ---
median_views_day = (
    df.groupby("DayOfWeek")["views"]
      .median()
      .reindex(day_order)
)

plt.figure(figsize=(8,5))
sns.barplot(x=median_views_day.index, y=median_views_day.values, order=day_order)
plt.title("Median Views by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Median Views")
plt.show()

# Print best days by both metrics
best_day_mean = mean_views_day.idxmax()
best_day_median = median_views_day.idxmax()
print(f"📊 Best day by average (mean) views: {best_day_mean}")
print(f"📊 Best day by typical (median) views: {best_day_median}")

# --- 7. Best day to post (keep this if you want old version too) ---
best_day = avg_views_day.idxmax()
print(f"✅ Best day to post (based on avg views): {best_day}")

# --- 8. Video performance clusters with custom K-Medoids ---
features = [
    "impressions", "impressions_ctr", "avg_views_per_viewer", "new_viewers",
    "subscribers_gained", "likes", "shares", "comments_added",
    "views", "watch_time_hours"
]
X = df[features].fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run K-Medoids
labels, _ = kmedoids(X_scaled, k=3, random_state=42)
df["Cluster"] = labels

# Map clusters to performance categories
cluster_avg_views = df.groupby("Cluster")["views"].mean().sort_values()
cluster_label_map = {
    cluster_avg_views.index[0]: "Low",
    cluster_avg_views.index[1]: "Mid",
    cluster_avg_views.index[2]: "High"
}
df["Performance Category"] = df["Cluster"].map(cluster_label_map)

# Scatter plot with labels
plt.figure(figsize=(12,7))
sns.scatterplot(
    data=df, x="impressions", y="views",
    hue="Performance Category", palette="viridis", s=80
)
for _, row in df.iterrows():
    plt.text(
        row["impressions"] + 0.01 * df["impressions"].max(),
        row["views"] + 0.01 * df["views"].max(),
        row["video_title"], fontsize=8
    )
plt.title("Video Performance Clusters (K-Medoids, Labeled)")
plt.xlabel("Impressions")
plt.ylabel("Views")
plt.tight_layout()
plt.show()

# Show videos in each category
for category in ["Low", "Mid", "High"]:
    print(f"\n{category} Performing Videos:")
    videos = df.loc[df["Performance Category"] == category, "video_title"].tolist()
    for v in videos:
        print(f" - {v}")

# --- 9. Feature importance for predicting views ---
X = df[features].drop(columns=["views"])
y = df["views"]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Variables for Predicting Views")
plt.show()

# --- 10. Descriptive Benchmarks / Indicators ---
benchmarks = {
    "watch_time_hours": df["watch_time_hours"].quantile(0.75),  # Top 25%
    "impressions": df["impressions"].quantile(0.75),
    "likes": df["likes"].quantile(0.75),
    "subscribers_gained": df["subscribers_gained"].quantile(0.75),
    "comments_added": df["comments_added"].quantile(0.75),
    "shares": df["shares"].quantile(0.75),
    "impressions_ctr": df["impressions_ctr"].quantile(0.75)
}

print("\n📊 Benchmark thresholds (top 25% of your dataset):")
for feature, threshold in benchmarks.items():
    print(f"{feature}: {threshold:.2f}")

plt.figure(figsize=(8,5))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Variables for Predicting Views (with Benchmarks)")
plt.xlabel("Importance")
plt.ylabel("Feature")

# Add red cutoff lines (descriptive indicators)
cutoffs = [0.05, 0.10, 0.20]
for cutoff in cutoffs:
    plt.axvline(cutoff, color="red", linestyle="--")

plt.tight_layout()
plt.show()
