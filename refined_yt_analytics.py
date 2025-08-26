# === YouTube Cross-Section Upgrade (no file saving) ===
# Keeps your structure; adds robust KPIs, clustering fixes, model importance (permutation),
# posting-time heatmap on rates, and a small recommender. — Step 11 intentionally skipped.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# --- Database connection parameters ---
db_params = {
    'dbname': 'test1',
    'user': 'postgres',
    'password': 'admin',
    'host': 'localhost',
    'port': '5432'
}

# =========================
# 0) Utilities / Helpers
# =========================

def winsorize(s: pd.Series, lower=0.01, upper=0.99):
    """Clip extreme tails to stabilize charts/stats."""
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)

# --- Simple, safer K-Medoids (fixed return indices + empty-cluster handling) ---
def kmedoids(X, k, max_iter=300, random_state=None):
    rng = np.random.default_rng(random_state)
    m = X.shape[0]
    if k > m:
        raise ValueError("k cannot be greater than number of rows")

    # Initial medoid indices (in original X index space)
    medoid_indices = rng.choice(m, k, replace=False)
    medoids = X[medoid_indices]

    for _ in range(max_iter):
        distances = pairwise_distances(X, medoids)
        labels = np.argmin(distances, axis=1)

        new_medoid_indices = medoid_indices.copy()

        for i in range(k):
            pts_idx = np.where(labels == i)[0]
            if len(pts_idx) == 0:
                # re-seed empty cluster with the farthest overall point from its nearest medoid
                far_idx = np.argmax(np.min(distances, axis=1))
                new_medoid_indices[i] = far_idx
                continue
            intra = pairwise_distances(X[pts_idx], X[pts_idx])
            medoid_local = pts_idx[np.argmin(intra.sum(axis=1))]
            new_medoid_indices[i] = medoid_local

        if np.array_equal(new_medoid_indices, medoid_indices):
            break

        medoid_indices = new_medoid_indices
        medoids = X[medoid_indices]

    # Final assignment
    distances = pairwise_distances(X, medoids)
    labels = np.argmin(distances, axis=1)
    return labels, medoid_indices

# =========================
# 1) Load & Clean
# =========================
conn = psycopg2.connect(**db_params)
query = """
SELECT video_title, views, publish_day, publish_month, publish_year,
       impressions, impressions_ctr, avg_views_per_viewer, new_viewers,
       subscribers_gained, likes, shares, comments_added, watch_time_hours
FROM yt_video_etl
"""
df = pd.read_sql(query, conn)
conn.close()

# Basic date cleaning
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
df = df.dropna(subset=["Publish_Date"]).copy()

# Day/Time features (hour will be 0 unless your data has time-of-day)
df["DayOfWeek"] = df["Publish_Date"].dt.day_name()
df["weekday_num"] = df["Publish_Date"].dt.weekday  # 0=Mon
df["hour"] = df["Publish_Date"].dt.hour.fillna(0).astype(int)
df["is_weekend"] = df["weekday_num"].isin([5, 6]).astype(int)

# Days since previous post (simple cadence control)
df = df.sort_values("Publish_Date")
df["days_since_prev"] = df["Publish_Date"].diff().dt.days
df["days_since_prev"] = df["days_since_prev"].fillna(df["days_since_prev"].median() or 7)

# =========================
# 2) Robust KPIs & Guards
# =========================
# Protect against divide-by-zero
df["impressions"] = df["impressions"].fillna(0).clip(lower=1)
df["views"] = df["views"].fillna(0).clip(lower=0)

for col in ["likes", "shares", "comments_added", "impressions_ctr",
            "avg_views_per_viewer", "new_viewers", "subscribers_gained",
            "watch_time_hours"]:
    if col not in df.columns:
        df[col] = 0
    df[col] = df[col].fillna(0)

# Rates
df["engagements"] = df[["likes", "comments_added", "shares"]].sum(axis=1)
df["engagement_rate"] = df["engagements"] / df["impressions"]
df["view_through_rate"] = df["views"] / df["impressions"]

# Winsorized versions for nicer summary plots
for col in ["views","impressions","likes","shares","comments_added",
            "engagement_rate","view_through_rate"]:
    df[f"{col}_w"] = winsorize(df[col])

# Log targets for modeling
df["log_views"] = np.log1p(df["views"])
df["log_eng_rate"] = np.log1p(df["engagement_rate"])

# Quick data-quality echo
print("Rows:", len(df), "| Date range:", df["Publish_Date"].min().date(), "→", df["Publish_Date"].max().date())

# =========================
# 3) Your existing visuals (kept)
# =========================
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Top 10 videos by views
top10 = df.nlargest(10, "views")[["video_title", "views"]]
plt.figure(figsize=(10,5))
sns.barplot(x="views", y="video_title", data=top10)
plt.title("Top 10 Videos by Views")
plt.tight_layout()
plt.show()

# Views over time (sum per day)
views_over_time = df.groupby(df["Publish_Date"].dt.date)["views"].sum()
plt.figure(figsize=(10,5))
views_over_time.plot()
plt.title("Views Over Time")
plt.xlabel("Date")
plt.ylabel("Views")
plt.tight_layout()
plt.show()

# Average views overall
avg_views = df["views"].mean()
print(f"Average views per video: {avg_views:.2f}")

# Average views by DayOfWeek (calendar order)
avg_views_day = (
    df.groupby("DayOfWeek")["views"]
      .mean()
      .reindex(day_order)
)
plt.figure(figsize=(8,5))
sns.barplot(x=avg_views_day.index, y=avg_views_day.values, order=day_order)
plt.title("Average Views by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Views")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Mean vs Median comparison (day of week)
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

best_day_mean = views_day_stats["mean"].idxmax()
best_day_median = views_day_stats["median"].idxmax()
print(f"📊 Best day by average (mean) views: {best_day_mean}")
print(f"📊 Best day by typical (median) views: {best_day_median}")

# =========================
# 4) Rate-first timing heatmap (weekday × hour, median VTR)
# =========================
pivot_vtr = (df
    .assign(hour=df["hour"].fillna(0).astype(int))
    .pivot_table(index="weekday_num", columns="hour",
                 values="view_through_rate", aggfunc="median")
)
plt.figure(figsize=(12,5))
sns.heatmap(pivot_vtr, cmap="viridis", linewidths=.5)
plt.title("Median View-Through Rate by Weekday × Hour")
plt.ylabel("Weekday (0=Mon)")
plt.xlabel("Hour of Day")
plt.tight_layout()
plt.show()

# =========================
# 5) K-Medoids on normalized rates + automatic k selection
# =========================
rate_cols = ["engagement_rate", "view_through_rate"]
Z = StandardScaler().fit_transform(df[rate_cols].fillna(0))

k_candidates = [2, 3, 4, 5, 6]
best = None
for k in k_candidates:
    try:
        labels_k, _ = kmedoids(Z, k=k, random_state=42)
        if len(np.unique(labels_k)) == 1:
            continue
        score = silhouette_score(Z, labels_k, metric='euclidean')
        best = max(best or (-1,None,None), (score, k, labels_k))
    except ValueError:
        pass

if best is not None:
    sil_score, best_k, best_labels = best
    df["Cluster"] = best_labels
    print(f"Best k by silhouette on rates: k={best_k} (score={sil_score:.3f})")
else:
    # Fallback to k=3 if selection failed
    best_k = 3
    best_labels, _ = kmedoids(Z, k=best_k, random_state=42)
    df["Cluster"] = best_labels
    print("Silhouette selection failed; using k=3 fallback.")

# Human-readable cluster names by average of rates (z space)
centers = pd.DataFrame(Z, columns=[f"z_{c}" for c in rate_cols]).groupby(df["Cluster"]).mean()
name_map = {}
for c in centers.index:
    name_map[c] = f"C{c} — ER{centers.loc[c, 'z_engagement_rate']:+.2f}, VTR{centers.loc[c, 'z_view_through_rate']:+.2f}"
df["Cluster_Name"] = df["Cluster"].map(name_map)

# Quick labeled scatter (impressions vs views) using performance clusters
plt.figure(figsize=(12,7))
sns.scatterplot(
    data=df, x="impressions", y="views",
    hue="Cluster_Name", palette="viridis", s=80
)
# Optional: label points (comment out if crowded)
# for _, row in df.iterrows():
#     plt.text(row["impressions"]*1.01, row["views"]*1.01, str(row["video_title"])[:20], fontsize=8)
plt.title("Video Performance Clusters (K-Medoids on Rates)")
plt.xlabel("Impressions")
plt.ylabel("Views")
plt.tight_layout()
plt.show()

# Show videos in each cluster (trim title for readability)
for cname in sorted(df["Cluster_Name"].dropna().unique()):
    print(f"\n{cname} — Videos:")
    vids = df.loc[df["Cluster_Name"] == cname, "video_title"].astype(str).tolist()
    for v in vids:
        print(f" - {v}")

# =========================
# 6) Permutation Importance (more reliable than impurity)
# =========================
features = [
    "impressions","impressions_ctr","avg_views_per_viewer","new_viewers",
    "subscribers_gained","likes","shares","comments_added","watch_time_hours",
    "weekday_num","hour","is_weekend","days_since_prev"
]
X = df[features].fillna(0)
y = df["log_views"]  # or df["log_eng_rate"] for engagement modeling

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=600, random_state=42, n_jobs=-1, min_samples_leaf=2, max_features="sqrt"
)
rf.fit(X_train, y_train)
r2 = rf.score(X_test, y_test)
print(f"RandomForest R^2 on test (log_views): {r2:.3f}")

perm = permutation_importance(
    rf, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1
)
pi = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
pi_std = pd.Series(perm.importances_std, index=X.columns).loc[pi.index]
print("\nPermutation importance (log_views):")
print(pi.head(15))

plt.figure(figsize=(8,5))
sns.barplot(x=pi.values, y=pi.index)
plt.title("Permutation Importance — Drivers of log(views)")
plt.xlabel("Mean Importance (ΔR² on permutation)")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# =========================
# 7) Benchmark thresholds (top quartile) — retained from your code
# =========================
benchmarks = {
    "watch_time_hours": df["watch_time_hours"].quantile(0.75),
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

# =========================
# 8) “Top Slots” recommender (median engagement_rate)
# =========================
slot = (df.groupby(["weekday_num","hour"])["engagement_rate"]
          .agg(["median","count"])
          .query("count >= 5")  # only keep slots with enough samples; tweak as you like
          .sort_values("median", ascending=False)
          .head(3))
print("\nTop 3 weekday×hour slots to test next (by median engagement_rate, n≥5):")
print(slot)

print("\nContent heuristics to test next week:")
print("- Optimize thumbnails/titles aiming to lift impressions_ctr into your top quartile.")
print("- Encourage comments (calls to action) where comments per 1k views historically skew high.")
print("- Experiment with a 3–5s live-performance hook; track changes in view_through_rate.")
print("- Add captions/subtitles if you can measure watch_time_hours uplift.")
