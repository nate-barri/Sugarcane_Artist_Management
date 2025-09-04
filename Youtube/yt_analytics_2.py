# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 20:36:58 2025

@author: Nathaniel
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import pairwise_distances
import matplotlib.dates as mdates

# --- Database connection parameters ---
db_params = {
    'dbname': 'test1',
    'user': 'postgres',
    'password': 'admin',
    'host': 'localhost',
    'port': '5432'
}

# --- Simple K-Medoids Implementation (as in your code) ---
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
df = df.dropna(subset=["publish_day", "publish_month", "publish_year"]).copy()
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
plt.tight_layout()
plt.show()

# --- 5. Average views ---
avg_views = df["views"].mean()
print(f"Average views per video: {avg_views:.2f}")

# --- 6. Average views by day of week (in calendar order) ---
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

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
plt.show()

# --- Mean vs Median comparison ---
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
print(f"📊 Best day by typical (median) views: {best_day_median}")

# --- 6a. Average (Mean) views by day of week ---
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

# --- 7. Best day to post (based on avg views) ---
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
plt.tight_layout()
plt.show()

# ============================================================================
# === 10. OPTIMIZED, TIME-AWARE BENCHMARKS (robust + rolling with no reindex error)
# ============================================================================

# Safety + core rates
for c in ["likes","shares","comments_added","impressions","views"]:
    if c not in df.columns: df[c] = 0
df["likes"] = df["likes"].fillna(0)
df["shares"] = df["shares"].fillna(0)
df["comments_added"] = df["comments_added"].fillna(0)
df["impressions"] = df["impressions"].fillna(0).clip(lower=1)
df["views"] = df["views"].fillna(0)

df["engagements"] = df["likes"] + df["comments_added"] + df["shares"]
df["engagement_rate"]   = df["engagements"] / df["impressions"]
df["view_through_rate"] = df["views"] / df["impressions"]

# Winsorize heavy tails
def winsorize(s, p=0.01):
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lo, hi)

for col in ["views","impressions","engagement_rate","view_through_rate"]:
    df[f"{col}_w"] = winsorize(df[col])

# Robust z-scores (median/MAD) → CPSr (composite)
def robust_z(s):
    s = s.astype(float)
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med))
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return 0.6745 * (s - med) / mad  # ~std-normal scaling

df["log_views"] = np.log1p(df["views_w"])
df["CPSr"] = (robust_z(df["log_views"])
            + robust_z(df["view_through_rate_w"])
            + robust_z(df["engagement_rate_w"])) / 3.0

# Overall cuts (OK = 50th, Great = 75th)
q50 = np.nanpercentile(df["CPSr"], 50)
q75 = np.nanpercentile(df["CPSr"], 75)

def tier_overall(x):
    if x >= q75: return "Great"
    if x >= q50: return "OK"
    return "Needs Work"

df["PerfTier_Overall"] = df["CPSr"].apply(tier_overall)
df["Will_Do_OK_Overall"] = (df["CPSr"] >= q50).astype(int)

# Monthly baseline (growth-aware)
df["YearMonth"] = df["Publish_Date"].dt.to_period("M")
monthly = (df.groupby("YearMonth")["CPSr"]
             .agg(q50="median", q75=lambda s: s.quantile(0.75), n="count"))
df = df.merge(monthly, left_on="YearMonth", right_index=True, how="left")

def tier_month(row):
    if row["n"] < 8 or pd.isna(row["q50"]) or pd.isna(row["q75"]):
        return np.nan
    if row["CPSr"] >= row["q75"]: return "Great"
    if row["CPSr"] >= row["q50"]: return "OK"
    return "Needs Work"

df["PerfTier_Month"] = df.apply(tier_month, axis=1)

# Rolling 180-day baseline (unique-by-day to avoid reindexing error)
df["pub_date"] = pd.to_datetime(df["Publish_Date"]).dt.normalize()
daily_cpsr = (df.groupby("pub_date")["CPSr"]
                .median()
                .sort_index())

roll_p50_daily = (daily_cpsr.rolling("180D", min_periods=10).quantile(0.50).shift(1))
roll_p75_daily = (daily_cpsr.rolling("180D", min_periods=10).quantile(0.75).shift(1))

df["CPSr_p50_180d"] = df["pub_date"].map(roll_p50_daily)
df["CPSr_p75_180d"] = df["pub_date"].map(roll_p75_daily)

def tier_roll(row):
    if pd.isna(row["CPSr_p50_180d"]) or pd.isna(row["CPSr_p75_180d"]):
        return np.nan
    if row["CPSr"] >= row["CPSr_p75_180d"]: return "Great"
    if row["CPSr"] >= row["CPSr_p50_180d"]: return "OK"
    return "Needs Work"

df["PerfTier_Roll180"] = df.apply(tier_roll, axis=1)

# Gating: don’t call it “Great” if ER or VTR are weak for that month
month_er_vtr_q40 = (df.groupby("YearMonth")[["engagement_rate_w","view_through_rate_w"]]
                      .quantile(0.40)
                      .rename(columns={"engagement_rate_w":"er_q40_m","view_through_rate_w":"vtr_q40_m"}))
df = df.merge(month_er_vtr_q40, left_on="YearMonth", right_index=True, how="left")

def tier_gated(row):
    base = row["PerfTier_Overall"]
    if pd.isna(base): return base
    weak = ((row["engagement_rate_w"] < row.get("er_q40_m", -np.inf)) or
            (row["view_through_rate_w"] < row.get("vtr_q40_m", -np.inf)))
    if weak:
        return {"Great":"OK", "OK":"Needs Work", "Needs Work":"Needs Work"}[base]
    return base

df["PerfTier_Overall_Gated"] = df.apply(tier_gated, axis=1)

print("\n=== Optimized Benchmarks ===")
print(f"Overall CPSr cuts — OK@{q50:.3f}, Great@{q75:.3f}")
print("Overall tiers:", df["PerfTier_Overall"].value_counts().to_dict())
print("Monthly tiers:", df["PerfTier_Month"].value_counts(dropna=True).to_dict())
print("Rolling-180d tiers:", df["PerfTier_Roll180"].value_counts(dropna=True).to_dict())
print("Overall (gated) tiers:", df["PerfTier_Overall_Gated"].value_counts(dropna=True).to_dict())

# ============================================================================
# === 10A. BENCHMARK VISUALIZATIONS
# ============================================================================

# A) Overall CPSr histogram + cut lines
plt.figure(figsize=(9,4))
sns.histplot(df["CPSr"].dropna(), bins=30, kde=True)
plt.axvline(q50, linestyle="--", label=f"OK (50th) = {q50:.2f}")
plt.axvline(q75, linestyle="--", label=f"Great (75th) = {q75:.2f}")
plt.title("Composite Performance Score (CPSr) — Overall Benchmarks")
plt.xlabel("CPSr"); plt.ylabel("Videos"); plt.legend()
plt.tight_layout(); plt.show()

# B) Monthly benchmark lines (median & 75th)
mplot = monthly.reset_index().rename(columns={"YearMonth":"month"})
mplot["month_str"] = mplot["month"].astype(str)
plt.figure(figsize=(10,4))
plt.plot(mplot["month_str"], mplot["q50"], marker="o", label="Monthly Median (OK cut)")
plt.plot(mplot["month_str"], mplot["q75"], marker="o", label="Monthly 75th (Great cut)")
plt.xticks(rotation=45, ha="right"); plt.ylabel("CPSr")
plt.title("Monthly CPSr Benchmarks"); plt.legend()
plt.tight_layout(); plt.show()

# C) Monthly CPSr distribution (boxplots)
plt.figure(figsize=(10,4))
sns.boxplot(
    data=df.dropna(subset=["YearMonth","CPSr"]).assign(month_str=lambda x: x["YearMonth"].astype(str)),
    x="month_str", y="CPSr", showfliers=False
)
plt.title("CPSr by Month"); plt.xlabel("Month"); plt.ylabel("CPSr")
plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()

# D) Rolling 180-day cuts vs dots
df_sorted_plot = df.sort_values("Publish_Date")
plt.figure(figsize=(11,4))
plt.scatter(df_sorted_plot["Publish_Date"], df_sorted_plot["CPSr"], s=12, alpha=0.35, label="Videos")
plt.plot(df_sorted_plot["Publish_Date"], df_sorted_plot["CPSr_p50_180d"], linewidth=2, label="Rolling 180d Median (OK)")
plt.plot(df_sorted_plot["Publish_Date"], df_sorted_plot["CPSr_p75_180d"], linewidth=2, label="Rolling 180d 75th (Great)")
ax = plt.gca()
locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
plt.ylabel("CPSr"); plt.title("Rolling 180-Day CPSr Benchmarks")
plt.legend(); plt.tight_layout(); plt.show()

# E) Overall (gated) tier distribution
plt.figure(figsize=(6,4))
tier_counts = df["PerfTier_Overall_Gated"].value_counts(dropna=False)
sns.barplot(x=tier_counts.index, y=tier_counts.values)
plt.title("Tier Distribution (Overall, Gated)")
plt.xlabel("Tier"); plt.ylabel("Videos")
plt.tight_layout(); plt.show()

# F) Monthly tier shares (stacked)
tiers = df.dropna(subset=["PerfTier_Month"]).copy()
tiers["month_str"] = tiers["YearMonth"].astype(str)
share = (tiers.groupby(["month_str","PerfTier_Month"]).size()
              .groupby(level=0).apply(lambda s: s / s.sum())
              .unstack(fill_value=0)[["Needs Work","OK","Great"]])
share.plot(kind="bar", stacked=True, figsize=(10,4))
plt.title("Monthly Tier Shares (Needs Work / OK / Great)")
plt.xlabel("Month"); plt.ylabel("Share of Posts")
plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()
