# ===============================================
# FACEBOOK DESCRIPTIVE ANALYTICS
# - Impute missing reach with RandomForest
# - Evaluate (MAE, R², MAE%, MAPE)
# - Charts (all use reach_filled)
# - Text Findings printed at the end
# ===============================================
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --- DB connection ---
db_params = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host': 'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port': '5432',
    'sslmode': 'require'
}

conn = psycopg2.connect(**db_params)
base = pd.read_sql("""
    SELECT publish_time, reactions, comments, shares, reach, post_type
    FROM public.facebook_data_set
""", conn)
conn.close()

# --- cleaning ---
base['publish_time'] = pd.to_datetime(base['publish_time'], errors='coerce', utc=True).dt.tz_convert(None)
base = base.dropna(subset=['publish_time']).copy()
for col in ['reactions','comments','shares','reach']:
    base[col] = pd.to_numeric(base[col], errors='coerce')

# --- feature engineering ---
base['month'] = base['publish_time'].values.astype('datetime64[M]')
base['hour'] = base['publish_time'].dt.hour
base['dow'] = base['publish_time'].dt.dayofweek  # 0=Mon..6=Sun
dow_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
base['engagement'] = base[['reactions','comments','shares']].sum(axis=1)

# --- Impute missing reach with RandomForest ---
X = pd.get_dummies(base[['engagement','post_type']], dummy_na=True)
y = base['reach']
train_mask = y.notna()
X_train, y_train = X[train_mask], y[train_mask]
X_missing = X[~train_mask]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred_train = model.predict(X_train)
mae = mean_absolute_error(y_train, y_pred_train)
r2 = r2_score(y_train, y_pred_train)
mean_reach_actual = y_train.mean()
mae_pct = (mae / mean_reach_actual) * 100

# MAPE (drop div-by-zero and infs)
mape = (np.abs((y_train - y_pred_train) / y_train)
        .replace([np.inf, -np.inf], np.nan)
        .dropna().mean() * 100)

print(f"Training MAE : {mae:.2f}")
print(f"Training R²  : {r2:.3f}")
print(f"Training MAE%: {mae_pct:.2f}% of mean reach")
print(f"Training MAPE: {mape:.2f}%")

# --- Impute missing reach ---
base.loc[train_mask, 'reach_filled'] = y[train_mask]
if not X_missing.empty:
    base.loc[~train_mask, 'reach_filled'] = model.predict(X_missing)

# ========== 1) Cumulative Reach ==========
monthly_orig = base.groupby('month')['reach'].sum(min_count=1)
cumulative_orig = monthly_orig.cumsum()

monthly_filled = base.groupby('month')['reach_filled'].sum()
full_idx = pd.date_range(start=monthly_filled.index.min(),
                         end=monthly_filled.index.max(), freq='MS')
monthly_filled = monthly_filled.reindex(full_idx, fill_value=0)
cumulative_filled = monthly_filled.cumsum()

plt.figure(figsize=(11,5))
plt.plot(cumulative_filled.index, cumulative_filled.values,
         label="Imputed Reach (Regression)", marker="o")
plt.plot(cumulative_orig.index, cumulative_orig.values,
         label="Original Reach (NaN gaps)", linestyle="--", marker="x")
plt.title("Cumulative Reach Growth Over Time")
plt.xlabel("Month"); plt.ylabel("Cumulative Reach")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# ========== 2) Monthly Engagement ==========
monthly_eng = base.groupby('month')['engagement'].sum()
plt.figure(figsize=(10,4.5))
plt.plot(monthly_eng.index, monthly_eng.values, marker='o')
plt.title("Monthly Engagement (Reactions + Comments + Shares)")
plt.xlabel("Month"); plt.ylabel("Engagement"); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# ========== 3) Post Type Performance ==========
post_totals = base.groupby('post_type')[['reactions','comments','shares']].sum().reset_index()
post_totals = post_totals.fillna(0)
x = np.arange(len(post_totals))
plt.figure(figsize=(9,5))
plt.bar(x, post_totals['reactions'], label="Reactions")
plt.bar(x, post_totals['comments'], bottom=post_totals['reactions'], label="Comments")
plt.bar(x, post_totals['shares'],
        bottom=post_totals['reactions']+post_totals['comments'], label="Shares")
plt.xticks(x, post_totals['post_type'], rotation=20, ha="right")
plt.title("Post Type Performance (Stacked)")
plt.ylabel("Count"); plt.legend()
plt.tight_layout(); plt.show()

# ========== 4) Engagement Rate by Hour ==========
valid = base[base['reach_filled'] > 0].copy()
valid['eng_rate'] = valid['engagement'] / valid['reach_filled']
by_hour = valid.groupby('hour')['eng_rate'].mean().reindex(range(24))
plt.figure(figsize=(10,4))
plt.bar(by_hour.index, by_hour.values)
plt.xticks(range(24))
plt.xlabel("Hour of Day"); plt.ylabel("Avg Engagement Rate")
plt.title("Average Engagement Rate by Hour")
plt.tight_layout(); plt.show()

# ========== 5) Median Engagement Rate by Weekday ==========
by_dow = valid.groupby('dow')['eng_rate'].median().reindex(range(7))  # Median instead of mean
plt.figure(figsize=(9,4))
plt.bar([dow_labels[d] for d in by_dow.index], by_dow.values)
plt.title("Median Engagement Rate by Day of Week")
plt.xlabel("Day of Week"); plt.ylabel("Median Engagement Rate")
plt.tight_layout(); plt.show()

# ========== 6) Median Reach by Weekday ==========
by_dow_reach = base.groupby('dow')['reach_filled'].median().reindex(range(7))
plt.figure(figsize=(9,4))
plt.bar(dow_labels, by_dow_reach.values)
plt.title("Median Reach by Day of Week (with Imputation)")
plt.xlabel("Day of Week"); plt.ylabel("Median Reach")
plt.tight_layout(); plt.show()

# ========== 7) Heatmap: Day × Hour ==========
heatmap = valid.groupby(['dow','hour'])['eng_rate'].mean().unstack(fill_value=0)
plt.figure(figsize=(10,5))
im = plt.imshow(heatmap.values, aspect='auto', origin='upper')
plt.colorbar(im, label="Avg Engagement Rate")
plt.xticks(ticks=np.arange(24), labels=np.arange(24))
plt.yticks(ticks=np.arange(7), labels=dow_labels)
plt.xlabel("Hour of Day"); plt.ylabel("Day of Week")
plt.title("Engagement Heatmap — Day × Hour")
plt.tight_layout(); plt.show()

# ======================================================
# ==================== FINDINGS ========================
# (Text summary printed from the computed aggregates)
# ======================================================
print("\n==================== FACEBOOK: Key Findings ====================\n")

# 1) Growth trend based on last 12 months slope (imputed reach)
if len(monthly_filled) >= 3:
    last_n = min(12, len(monthly_filled))
    y_vals = monthly_filled.tail(last_n).values
    x_idx = np.arange(len(y_vals))
    slope = np.polyfit(x_idx, y_vals, 1)[0]
    trend = "upward" if slope > 0 else ("downward" if slope < 0 else "flat")
    print(f"Growth Trend (last {last_n} months, imputed reach): {trend} (slope={slope:.2f} reach/month)")
else:
    print("Growth Trend: Not enough data to assess.")

# 2) Biggest engagement spike month
if not monthly_eng.empty:
    peak_month = monthly_eng.idxmax()
    peak_val = monthly_eng.max()
    median_val = monthly_eng.median()
    ratio = (peak_val / median_val) if median_val and median_val != 0 else np.nan
    print(f"Largest Engagement Spike: {peak_month.date()} (value={int(peak_val)}; vs median x{ratio:.2f})")

# 3) Top post type by total engagement, and share-heavy type
post_totals_eng = post_totals.copy()
post_totals_eng['total_eng'] = post_totals_eng[['reactions','comments','shares']].sum(axis=1)
top_type_row = post_totals_eng.loc[post_totals_eng['total_eng'].idxmax()] if not post_totals_eng.empty else None
if top_type_row is not None:
    print(f"Top Content Type by Engagement: {top_type_row['post_type']} "
          f"(total_eng={int(top_type_row['total_eng'])})")
    # Most share-weighted type (shares / total engagements)
    post_totals_eng['share_ratio'] = np.where(
        post_totals_eng['total_eng']>0,
        post_totals_eng['shares'] / post_totals_eng['total_eng'],
        np.nan
    )
    share_heavy = post_totals_eng.loc[post_totals_eng['share_ratio'].idxmax()]
    print(f"Most Viral Type (highest Shares/Engagement): {share_heavy['post_type']} "
          f"(share_ratio={share_heavy['share_ratio']:.2%})")

# 4) Best hour by average engagement rate
if by_hour.notna().any():
    best_hour = int(by_hour.idxmax())
    print(f"Best Posting Hour (Avg Engagement Rate): {best_hour:02d}:00 "
          f"(rate={by_hour.max():.4f})")

# 5) Best weekday by average engagement rate
if not by_dow.empty:
    best_dow_idx = int(by_dow.idxmax())
    print(f"Best Day (Median Engagement Rate): {dow_labels[best_dow_idx]} "
          f"(rate={by_dow.loc[best_dow_idx]:.4f})")

# 6) Highest median reach weekday
if not by_dow_reach.empty:
    best_med_dow = int(by_dow_reach.idxmax())
    print(f"Highest Median Reach Day: {dow_labels[best_med_dow]} "
          f"(median_reach={int(by_dow_reach.max())})")

# 7) Heatmap hotspots (top 3 Day×Hour cells)
heatmap_long = heatmap.stack().rename('avg_eng_rate').reset_index()
if not heatmap_long.empty:
    top3 = heatmap_long.sort_values('avg_eng_rate', ascending=False).head(3)
    print("Top 3 Day×Hour Hotspots (by Avg Engagement Rate):")
    for _, row in top3.iterrows():
        print(f"  - {dow_labels[int(row['dow'])]} @ {int(row['hour']):02d}:00  →  {row['avg_eng_rate']:.4f}")

print("\n===============================================================\n")

# ---- 1) Build a clean feature table for clustering ----
# We'll use rows with valid imputed reach_filled and computed eng_rate
clu = base.copy()
clu = clu[(clu['reach_filled'].notna()) & (clu['reach_filled'] > 0)].copy()

# Recompute engagement metrics just to be safe
clu['engagement'] = clu[['reactions','comments','shares']].sum(axis=1)
clu['eng_rate']   = clu['engagement'] / clu['reach_filled']

# Basic numeric features (log transforms to reduce skew)
def safe_log(x):
    # log1p avoids log(0); also handles negatives if ever present by clipping
    return np.log1p(np.clip(x, a_min=0, a_max=None))

clu['log_reach']   = safe_log(clu['reach_filled'])
clu['log_eng']     = safe_log(clu['engagement'])
clu['log_reacts']  = safe_log(clu['reactions'].fillna(0))
clu['log_comments']= safe_log(clu['comments'].fillna(0))
clu['log_shares']  = safe_log(clu['shares'].fillna(0))

# Time and content features
clu['hour'] = clu['publish_time'].dt.hour
clu['dow']  = clu['publish_time'].dt.dayofweek  # 0=Mon..6=Sun

# One-hot for post_type (keeps NA as separate column if any)
post_dummies = pd.get_dummies(clu['post_type'].fillna('Unknown'), prefix='type')

# Assemble feature matrix
feature_cols_cont = [
    'log_reach','log_eng','log_reacts','log_comments','log_shares',
    'eng_rate','hour','dow'
]
X = pd.concat([clu[feature_cols_cont], post_dummies], axis=1).fillna(0)

# Scale continuous features (but not the one-hots)
scaler = StandardScaler()
X_cont_scaled = scaler.fit_transform(X[feature_cols_cont])
X_scaled = np.hstack([X_cont_scaled, X[post_dummies.columns].values])

# ---- 2) Elbow (WCSS) and Silhouette over k=2..10 ----
ks = list(range(2, 11))
wcss = []
silh = []

for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)  # Within-cluster sum of squares
    # Silhouette requires >1 cluster and <n_samples clusters, which we have here
    silh.append(silhouette_score(X_scaled, labels))

# Plot: Elbow curve
plt.figure(figsize=(8,4))
plt.plot(ks, wcss, marker='o')
plt.title("Elbow Method (WCSS vs k)")
plt.xlabel("k (number of clusters)")
plt.ylabel("WCSS (inertia)")
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# Plot: Silhouette vs k
plt.figure(figsize=(8,4))
plt.plot(ks, silh, marker='o')
plt.title("Silhouette Score vs k")
plt.xlabel("k (number of clusters)")
plt.ylabel("Average Silhouette")
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# Auto-pick k with highest silhouette (you can override manually if elbow suggests otherwise)
k_best = ks[int(np.argmax(silh))]
print(f"Auto-selected k by silhouette: k={k_best} (score={max(silh):.3f})")

# ---- 3) Fit final KMeans with k_best ----
kmeans = KMeans(n_clusters=k_best, n_init=10, random_state=42)
clu['cluster'] = kmeans.fit_predict(X_scaled)

# ---- 4) Cluster profiling ----
print("\n==================== CLUSTER PROFILES ====================\n")

# Cluster sizes
sizes = clu['cluster'].value_counts().sort_index()
print("Cluster sizes:")
for cid, n in sizes.items():
    print(f"  Cluster {cid}: n={n}")

# Aggregate numeric behavior per cluster (original scale where meaningful)
prof_num = (clu
    .groupby('cluster')
    .agg(
        posts=('cluster','size'),
        avg_reach=('reach_filled','mean'),
        med_reach=('reach_filled','median'),
        avg_eng=('engagement','mean'),
        med_eng=('engagement','median'),
        avg_eng_rate=('eng_rate','mean'),
        med_eng_rate=('eng_rate','median'),
        top_hour=('hour', lambda s: s.value_counts().index[0] if len(s)>0 else np.nan),
        top_dow =('dow',  lambda s: s.value_counts().index[0] if len(s)>0 else np.nan)
    )
    .sort_index()
)

# Map DOW to labels for readability
dow_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
prof_num['top_dow_label'] = prof_num['top_dow'].map({i:l for i,l in enumerate(dow_labels)})

print("\nNumeric profile (means/medians and dominant posting time):")
print(prof_num[['posts','avg_reach','med_reach','avg_eng','med_eng','avg_eng_rate','med_eng_rate','top_hour','top_dow_label']])

# Content mix per cluster
mix = (pd.crosstab(clu['cluster'], clu['post_type'].fillna('Unknown'), normalize='index')*100).round(1)
print("\nPost type mix by cluster (%):")
print(mix)

# ---- 5) Quick visuals for clusters ----
# (a) Cluster sizes
plt.figure(figsize=(8,4))
plt.bar(sizes.index.astype(str), sizes.values)
plt.title("Cluster Sizes")
plt.xlabel("Cluster"); plt.ylabel("Count")
plt.tight_layout(); plt.show()

# (b) Avg engagement rate per cluster
plt.figure(figsize=(8,4))
plt.bar(prof_num.index.astype(str), prof_num['avg_eng_rate'].values)
plt.title("Average Engagement Rate by Cluster")
plt.xlabel("Cluster"); plt.ylabel("Avg Engagement Rate")
plt.tight_layout(); plt.show()

# (c) Median reach per cluster
plt.figure(figsize=(8,4))
plt.bar(prof_num.index.astype(str), prof_num['med_reach'].values)
plt.title("Median Reach by Cluster")
plt.xlabel("Cluster"); plt.ylabel("Median Reach")
plt.tight_layout(); plt.show()

# ---- 6) Optional: attach cluster back to your main base dataframe for downstream use ----
# (match by publish_time index)
base = base.join(clu[['publish_time','cluster']].set_index('publish_time'), on='publish_time')

print("\n✅ Clustering complete. You now have:")
print(" - Elbow & Silhouette plots")
print(" - Cluster labels in `clu['cluster']` (and merged into `base['cluster']`)") 
print(" - Printed profiles: size, engagement, reach, best hour/day, and post-type mix\n")

# --- Scatter 1: log_engagement vs log_reach ---
plt.figure(figsize=(8,6))
for cid in sorted(clu['cluster'].unique()):
    subset = clu[clu['cluster']==cid]
    plt.scatter(subset['log_reach'], subset['log_eng'], label=f"Cluster {cid}", alpha=0.6)
plt.xlabel("Log Reach")
plt.ylabel("Log Engagement")
plt.title("Clusters: Log Engagement vs Log Reach")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Scatter 2: PCA 2D projection of all features ---
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
clu['pca1'] = X_pca[:,0]
clu['pca2'] = X_pca[:,1]

plt.figure(figsize=(8,6))
for cid in sorted(clu['cluster'].unique()):
    subset = clu[clu['cluster']==cid]
    plt.scatter(subset['pca1'], subset['pca2'], label=f"Cluster {cid}", alpha=0.6)
plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
plt.title("Clusters (PCA-reduced space)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
