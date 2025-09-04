import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import numpy as np
import re  
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

# --- NEW: remove emojis from ALL text columns BEFORE any charts/aggregations ---
def remove_emojis(text):
    if not isinstance(text, str):
        return text
    emoji_pattern = re.compile(
        "["                     # common emoji blocks
        "\U0001F600-\U0001F64F" # emoticons
        "\U0001F300-\U0001F5FF" # symbols & pictographs
        "\U0001F680-\U0001F6FF" # transport & map
        "\U0001F700-\U0001F77F" # alchemical
        "\U0001F780-\U0001F7FF" # geometric shapes ext
        "\U0001F800-\U0001F8FF" # arrows-C
        "\U0001F900-\U0001F9FF" # supplemental pictographs
        "\U0001FA00-\U0001FA6F" # chess etc.
        "\U0001FA70-\U0001FAFF" # pictographs ext-A
        "\U00002702-\U000027B0" # dingbats
        "\U000024C2-\U0001F251" # enclosed
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].apply(remove_emojis)


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
# plt.show()

# --- 4. Views over time ---
views_over_time = df.groupby(df["Publish_Date"].dt.date)["views"].sum()
plt.figure(figsize=(10,5))
views_over_time.plot()
plt.title("Views Over Time")
plt.xlabel("Date")
plt.ylabel("Views")
# plt.show()

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
# plt.show()

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
# plt.show()

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
# plt.show()

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
# plt.show()

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
# plt.show()

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
# plt.show()

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
# plt.show()


# --- 10F. Frozen Future Benchmark (for NEW videos) --------------------------
AS_OF = pd.Timestamp.today().normalize()
HOLDOUT_DAYS = 14      # ignore the most recent N days
HIST_DAYS    = 365     # use the last 365 days before holdout

hist_end   = AS_OF - pd.Timedelta(days=HOLDOUT_DAYS)
hist_start = hist_end - pd.Timedelta(days=HIST_DAYS)

hist = df[(df["Publish_Date"] >= hist_start) & (df["Publish_Date"] < hist_end)].copy()
if hist.empty:
    hist = df[df["Publish_Date"] < hist_end].copy()

for c in ["likes", "shares", "comments_added", "impressions", "views"]:
    if c not in hist.columns:
        hist[c] = 0
hist["likes"] = hist["likes"].fillna(0)
hist["shares"] = hist["shares"].fillna(0)
hist["comments_added"] = hist["comments_added"].fillna(0)
hist["impressions"] = hist["impressions"].fillna(0).clip(lower=1)
hist["views"] = hist["views"].fillna(0)

hist["ER"]  = (hist["likes"] + hist["comments_added"] + hist["shares"]) / hist["impressions"]
hist["VTR"] = hist["views"] / hist["impressions"]

def _winsorize(s, p=0.01):
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lo, hi)

hist["views_w"] = _winsorize(hist["views"])
hist["ER_w"]    = _winsorize(hist["ER"])
hist["VTR_w"]   = _winsorize(hist["VTR"])
hist["log_views"] = np.log1p(hist["views_w"])

def _robust_params(s: pd.Series):
    med = float(np.nanmedian(s))
    mad = float(np.nanmedian(np.abs(s - med)))
    if not np.isfinite(mad) or mad == 0:
        mad = np.nan
    return {"med": med, "mad": mad}

rp_logv = _robust_params(hist["log_views"])
rp_vtr  = _robust_params(hist["VTR_w"])
rp_er   = _robust_params(hist["ER_w"])

def _r_z(x, rp):
    if (rp is None) or (rp.get("mad") is None) or (not np.isfinite(rp["mad"])) or rp["mad"] == 0:
        return 0.0
    return 0.6745 * (x - rp["med"]) / rp["mad"]

hist["CPSr_hist"] = (
    hist["log_views"].apply(lambda x: _r_z(x, rp_logv)) +
    hist["VTR_w"].apply(lambda x: _r_z(x, rp_vtr)) +
    hist["ER_w"].apply(lambda x: _r_z(x, rp_er))
) / 3.0

standard = {
    "as_of": AS_OF.date().isoformat(),
    "history_window": {"start": hist_start.date().isoformat(), "end": hist_end.date().isoformat()},
    "cuts": {
        "cpsr_ok":    float(np.nanpercentile(hist["CPSr_hist"], 50)),
        "cpsr_great": float(np.nanpercentile(hist["CPSr_hist"], 75)),
    },
    "gates": {
        "vtr_floor":   float(hist["VTR_w"].quantile(0.40)),
        "er_floor":    float(hist["ER_w"].quantile(0.40)),
        "views_floor": int(hist["views_w"].quantile(0.25))
    },
    "robust_params": {"log_views": rp_logv, "VTR": rp_vtr, "ER": rp_er}
}

print("\n=== Frozen Future Benchmark (do not use current videos to set the bar) ===")
print(f"As-of date: {standard['as_of']} | History: {standard['history_window']}")
print(f"CPSr OK cut  : {standard['cuts']['cpsr_ok']:.3f}")
print(f"CPSr GREAT cut: {standard['cuts']['cpsr_great']:.3f}")
print(f"VTR floor (40th pct): {standard['gates']['vtr_floor']:.4f}")
print(f"ER floor  (40th pct): {standard['gates']['er_floor']:.4f}")
print(f"Views floor (25th pct): {standard['gates']['views_floor']}")

def classify_future_video(metrics: dict, std: dict) -> dict:
    imp = max(1.0, float(metrics.get("impressions", 0)))
    views = max(0.0, float(metrics.get("views", 0)))
    likes = float(metrics.get("likes", 0))
    comments = float(metrics.get("comments", 0))
    shares = float(metrics.get("shares", 0))

    log_views = np.log1p(views)
    VTR = views / imp
    ER  = (likes + comments + shares) / imp

    rp = std["robust_params"]
    cpsr = (_r_z(log_views, rp["log_views"]) +
            _r_z(VTR, rp["VTR"]) +
            _r_z(ER,  rp["ER"])) / 3.0

    ok_cut    = std["cuts"]["cpsr_ok"]
    great_cut = std["cuts"]["cpsr_great"]
    vtr_floor = std["gates"]["vtr_floor"]
    er_floor  = std["gates"]["er_floor"]
    views_floor = std["gates"]["views_floor"]

    tier = "Needs Work"
    if (cpsr >= great_cut and VTR >= vtr_floor and ER >= er_floor and views >= views_floor):
        tier = "Great"
    elif (cpsr >= ok_cut and VTR >= vtr_floor and ER >= er_floor and views >= views_floor):
        tier = "OK"

    return {"CPSr": float(cpsr), "tier": tier, "VTR": float(VTR), "ER": float(ER), "views": int(views)}

recent = df[df["Publish_Date"] >= hist_end].copy()
if not recent.empty:
    recent["ER"]  = (recent["likes"].fillna(0) + recent["comments_added"].fillna(0) + recent["shares"].fillna(0)) / recent["impressions"].clip(lower=1)
    recent["VTR"] = recent["views"].fillna(0) / recent["impressions"].clip(lower=1)

    def _apply_row(row):
        metrics = dict(
            impressions=float(row["impressions"] or 0),
            views=float(row["views"] or 0),
            likes=float(row.get("likes", 0) or 0),
            comments=float(row.get("comments_added", 0) or 0),
            shares=float(row.get("shares", 0) or 0),
        )
        out = classify_future_video(metrics, standard)
        return pd.Series([out["tier"], out["CPSr"], out["VTR"], out["ER"]], index=["Tier_Frozen","CPSr_Frozen","VTR","ER"])

    recent[["Tier_Frozen","CPSr_Frozen","VTR","ER"]] = recent.apply(_apply_row, axis=1)
    print("\nHow the frozen standard scores the most recent period (sanity check):")
    print(recent["Tier_Frozen"].value_counts())

# --- 10G. Aspirational Top-Performer Benchmark (median-of-top cohort) ------
TOP_SHARE = 0.20
MIN_TOP_N = 20
RELAX = 0.80

if "hist" not in locals() or hist.empty:
    AS_OF = pd.Timestamp.today().normalize()
    HOLDOUT_DAYS, HIST_DAYS = 14, 365
    hist_end = AS_OF - pd.Timedelta(days=HOLDOUT_DAYS)
    hist_start = hist_end - pd.Timedelta(days=HIST_DAYS)
    hist = df[(df["Publish_Date"] >= hist_start) & (df["Publish_Date"] < hist_end)].copy()
    if hist.empty: hist = df[df["Publish_Date"] < hist_end].copy()

for c in ["likes","shares","comments_added","impressions","views"]:
    if c not in hist.columns: hist[c] = 0
hist["impressions"] = hist["impressions"].fillna(0).clip(lower=1)
hist["views"]       = hist["views"].fillna(0)
hist["likes"]       = hist["likes"].fillna(0)
hist["shares"]      = hist["shares"].fillna(0)
hist["comments_added"] = hist["comments_added"].fillna(0)
hist["ER"]  = (hist["likes"] + hist["comments_added"] + hist["shares"]) / hist["impressions"]
hist["VTR"] = hist["views"] / hist["impressions"]

def _winsorize(s, p=0.01):
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lo, hi)

hist["views_w"] = _winsorize(hist["views"])
hist["ER_w"]    = _winsorize(hist["ER"])
hist["VTR_w"]   = _winsorize(hist["VTR"])
hist["log_views"] = np.log1p(hist["views_w"])

def _robust_params(s):
    med = float(np.nanmedian(s))
    mad = float(np.nanmedian(np.abs(s - med)))
    return {"med": med, "mad": (mad if (np.isfinite(mad) and mad > 0) else np.nan)}

def _r_z(x, rp):
    if not rp or not np.isfinite(rp["mad"]) or rp["mad"] == 0: return 0.0
    return 0.6745 * (x - rp["med"]) / rp["mad"]

if "rp_logv" not in locals(): rp_logv = _robust_params(hist["log_views"])
if "rp_vtr"  not in locals(): rp_vtr  = _robust_params(hist["VTR_w"])
if "rp_er"   not in locals(): rp_er   = _robust_params(hist["ER_w"])

hist["CPSr_hist"] = (
    hist["log_views"].apply(lambda x: _r_z(x, rp_logv)) +
    hist["VTR_w"].apply(lambda x: _r_z(x, rp_vtr)) +
    hist["ER_w"].apply(lambda x: _r_z(x, rp_er))
) / 3.0

n_top = max(MIN_TOP_N, int(np.ceil(len(hist) * TOP_SHARE)))
top = hist.nlargest(n_top, "CPSr_hist")

aspirational = {
    "as_of": pd.Timestamp.today().date().isoformat(),
    "top_share": TOP_SHARE,
    "n_top": int(len(top)),
    "targets": {
        "cpsr_target": float(np.nanmedian(top["CPSr_hist"])),
        "vtr_target":  float(np.nanmedian(top["VTR_w"])),
        "er_target":   float(np.nanmedian(top["ER_w"])),
        "views_target": int(np.nanmedian(top["views_w"]))
    },
    "robust_params": {"log_views": rp_logv, "VTR": rp_vtr, "ER": rp_er},
    "relax": RELAX
}

print("\n=== Aspirational Top-Performer Standard (median of top cohort) ===")
print(f"Using top {aspirational['n_top']} posts (~{int(TOP_SHARE*100)}%) as 'best'.")
print("Targets  →  CPSr ≥ {cpsr_target:.2f},  VTR ≥ {vtr:.4f},  ER ≥ {er:.4f},  Views ≥ {views}"
      .format(cpsr_target=aspirational["targets"]["cpsr_target"],
              vtr=aspirational["targets"]["vtr_target"],
              er=aspirational["targets"]["er_target"],
              views=aspirational["targets"]["views_target"]))

def classify_future_video_aspirational(metrics: dict, std: dict) -> dict:
    imp = max(1.0, float(metrics.get("impressions", 0)))
    views = max(0.0, float(metrics.get("views", 0)))
    likes = float(metrics.get("likes", 0))
    comments = float(metrics.get("comments", 0))
    shares = float(metrics.get("shares", 0))

    log_views = np.log1p(views)
    VTR = views / imp
    ER  = (likes + comments + shares) / imp

    rp = std["robust_params"]
    cpsr = (_r_z(log_views, rp["log_views"]) +
            _r_z(VTR, rp["VTR"]) +
            _r_z(ER,  rp["ER"])) / 3.0

    t = std["targets"]; relax = std["relax"]
    great = (cpsr >= t["cpsr_target"] and VTR >= t["vtr_target"] and
             ER >= t["er_target"] and views >= t["views_target"])
    ok    = (cpsr >= relax*t["cpsr_target"] and VTR >= relax*t["vtr_target"] and
             ER >= relax*t["er_target"] and views >= relax*t["views_target"])

    tier = "Great (Aspirational)" if great else ("OK (Aspirational)" if ok else "Needs Work")
    return {"tier": tier, "CPSr": float(cpsr), "VTR": float(VTR), "ER": float(ER), "views": int(views)}

# --- 10H. Graphs for Aspirational (Top-Performer) Benchmarks ----------------
# Guard: ensure required objects exist
assert 'aspirational' in globals(), "Run 10G first to build 'aspirational'."
assert 'hist' in globals() and 'top' in globals(), "Run 10G to create 'hist' and 'top'."

t = aspirational["targets"]
relax = aspirational.get("relax", 0.80)

cpsr_t  = t["cpsr_target"];  cpsr_t_rel  = relax * cpsr_t
vtr_t   = t["vtr_target"];   vtr_t_rel   = relax * vtr_t
er_t    = t["er_target"];    er_t_rel    = relax * er_t
views_t = t["views_target"]; views_t_rel = int(np.round(relax * views_t))

# 1) CPSr histogram with target lines
plt.figure(figsize=(9,4))
sns.histplot(hist["CPSr_hist"].dropna(), bins=30, kde=True)
plt.axvline(cpsr_t, color="black", linestyle="--", label=f"CPSr target (median of top) = {cpsr_t:.2f}")
plt.axvline(cpsr_t_rel, color="gray", linestyle="--", label=f"CPSr relaxed ({int(relax*100)}%) = {cpsr_t_rel:.2f}")
plt.title("Aspirational CPSr Benchmark (Top-Performer Median)")
plt.xlabel("CPSr (robust composite)"); plt.ylabel("Posts")
plt.legend(); plt.tight_layout()
# plt.show()

# 2) VTR vs ER scatter with targets (highlight top cohort)
plot_df = hist.copy()
plot_df["is_top"] = plot_df.index.isin(top.index)

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=plot_df, x="VTR_w", y="ER_w",
    hue="is_top", palette={False:"tab:blue", True:"tab:orange"},
    size=np.clip(plot_df["views_w"], 1, np.nanpercentile(plot_df["views_w"], 95)),
    sizes=(20, 120), alpha=0.6
)
plt.axvline(vtr_t, color="black", linestyle="--", label=f"VTR target = {vtr_t:.4f}")
plt.axvline(vtr_t_rel, color="gray", linestyle="--", label=f"VTR relaxed = {vtr_t_rel:.4f}")
plt.axhline(er_t, color="black", linestyle="--", label=f"ER target = {er_t:.4f}")
plt.axhline(er_t_rel, color="gray", linestyle="--", label=f"ER relaxed = {er_t_rel:.4f}")
plt.title("Aspirational Quality Targets — VTR vs ER (point size ~ views)")
plt.xlabel("VTR (views / impressions)")
plt.ylabel("ER  (likes+comments+shares / impressions)")
plt.legend(title="Top cohort?"); plt.tight_layout()
# plt.show()

# 3) Views histogram (winsorized) with target lines
plt.figure(figsize=(9,4))
sns.histplot(plot_df["views_w"].dropna(), bins=30, kde=False)
plt.axvline(views_t, color="black", linestyle="--", label=f"Views target = {views_t:,}")
plt.axvline(views_t_rel, color="gray", linestyle="--", label=f"Views relaxed = {views_t_rel:,}")
plt.title("Aspirational Reach Benchmark (Views, winsorized)")
plt.xlabel("Views (winsorized)"); plt.ylabel("Posts")
plt.legend(); plt.tight_layout()
# plt.show()

#  render ALL figures at once
plt.show()


