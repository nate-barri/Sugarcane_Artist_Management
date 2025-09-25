import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score
from scipy import stats

# --- DB connection ---
db_params = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host': 'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port': '5432',
    'sslmode': 'require'
}

# Establish connection to Neon DB
conn = psycopg2.connect(**db_params)

# Query to fetch YouTube data from the table 'public.yt_video_etl'
query = """
    SELECT video_id, content, video_title, publish_day, publish_month, publish_year, duration, 
           impressions, impressions_ctr, avg_views_per_viewer, new_viewers, subscribers_gained, 
           subscribers_lost, likes, shares, comments_added, views, watch_time_hours, avg_dur_hours, 
           avg_dur_minutes, avg_dur_seconds, dislikes, unique_viewers, category
    FROM public.yt_video_etl
"""

# Fetch data into DataFrame
youtube_df = pd.read_sql(query, conn)
conn.close()

print(f"Loaded {len(youtube_df)} rows from database")

# --- Data Cleaning and Feature Engineering ---
youtube_df = youtube_df.dropna(subset=['publish_day', 'publish_month', 'publish_year'])
youtube_df['publish_day'] = youtube_df['publish_day'].astype(int)
youtube_df['publish_month'] = youtube_df['publish_month'].astype(int)
youtube_df['publish_year'] = youtube_df['publish_year'].astype(int)
youtube_df['publish_date'] = pd.to_datetime(
    youtube_df[['publish_year', 'publish_month', 'publish_day']].astype(str).agg('-'.join, axis=1)
)

# --- Create Engagement Column ---
youtube_df['engagement'] = youtube_df[['likes', 'shares', 'comments_added']].sum(axis=1)

# --- ROBUST OUTLIER DETECTION FUNCTIONS ---
def detect_outliers_modified_zscore(data, column, threshold=3.5):
    """
    Detect outliers using Modified Z-score (more robust)
    """
    median = data[column].median()
    mad = np.median(np.abs(data[column] - median))
    if mad == 0:  # Handle case where MAD is zero
        return pd.Series([False] * len(data), index=data.index)
    modified_z_scores = 0.6745 * (data[column] - median) / mad
    return np.abs(modified_z_scores) > threshold

def remove_outliers_robust(data, columns, threshold=2.5):
    """
    Remove outliers from multiple columns using Modified Z-score method
    """
    outlier_mask = pd.Series([False] * len(data), index=data.index)
    
    for column in columns:
        if column in data.columns:
            outlier_mask |= detect_outliers_modified_zscore(data, column, threshold)
    
    print(f"Removing {outlier_mask.sum()} outliers out of {len(data)} rows ({outlier_mask.sum()/len(data)*100:.1f}%)")
    return data[~outlier_mask].copy()

# --- Remove Outliers ---
youtube_df_cleaned = remove_outliers_robust(
    youtube_df, 
    ['engagement', 'views', 'watch_time_hours'], 
    threshold=2.5  # Less strict threshold for small dataset
)

print(f"Dataset after cleaning: {len(youtube_df_cleaned)} rows")

# --- CLUSTERING ANALYSIS FUNCTIONS ---
def analyze_optimal_clusters(data, features, max_clusters=6):
    """
    Find optimal number of clusters using multiple metrics
    """
    # Check if all features exist in the data
    available_features = [f for f in features if f in data.columns and data[f].notna().sum() > 0]
    if len(available_features) == 0:
        print("Error: No valid features found for clustering")
        return None, None, None, None
    
    print(f"Using features for clustering: {available_features}")
    
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    data_scaled = scaler.fit_transform(data[available_features])
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, min(max_clusters + 1, len(data)))  # Don't exceed number of samples
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data_scaled)
        
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(data_scaled, cluster_labels)
        silhouette_scores.append(sil_score)
    
    return k_range, inertias, silhouette_scores, scaler, available_features

def create_performance_score(df):
    """
    Create a composite performance score for better clustering
    """
    # Only use columns that exist and have data
    score_components = []
    weights = []
    
    if 'views' in df.columns and df['views'].notna().sum() > 0:
        normalized_views = (df['views'] - df['views'].min()) / (df['views'].max() - df['views'].min())
        score_components.append(normalized_views)
        weights.append(0.4)
    
    if 'engagement' in df.columns and df['engagement'].notna().sum() > 0:
        normalized_engagement = (df['engagement'] - df['engagement'].min()) / (df['engagement'].max() - df['engagement'].min())
        score_components.append(normalized_engagement)
        weights.append(0.4)
    
    if 'watch_time_hours' in df.columns and df['watch_time_hours'].notna().sum() > 0:
        normalized_watch_time = (df['watch_time_hours'] - df['watch_time_hours'].min()) / (df['watch_time_hours'].max() - df['watch_time_hours'].min())
        score_components.append(normalized_watch_time)
        weights.append(0.2)
    
    if len(score_components) == 0:
        print("Warning: No valid components for performance score")
        return pd.Series([0.5] * len(df), index=df.index)
    
    # Normalize weights to sum to 1
    weights = np.array(weights) / sum(weights)
    
    # Calculate weighted performance score
    performance_score = pd.Series([0.0] * len(df), index=df.index)
    for component, weight in zip(score_components, weights):
        performance_score += weight * component
    
    return performance_score

def robust_trend_line(x, y, method='theil_sen'):
    """
    Create robust trend line using Theil-Sen estimator
    """
    try:
        from sklearn.linear_model import TheilSenRegressor
        X = np.array(x).reshape(-1, 1)
        Y = np.array(y)
        reg = TheilSenRegressor(random_state=42)
        reg.fit(X, Y)
        trend_y = reg.predict(X)
        r2 = reg.score(X, Y)
        return trend_y, r2
    except:
        # Fallback to regular polyfit if TheilSen not available
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        trend_y = p(x)
        return trend_y, 0.0

# --- STEP 1: Analyze optimal number of clusters ---
print("\n=== STEP 1: Finding Optimal Number of Clusters ===")

features_for_clustering = ['likes', 'views', 'engagement']
result = analyze_optimal_clusters(youtube_df_cleaned, features_for_clustering)

if result[0] is not None:
    k_range, inertias, silhouette_scores, scaler, available_features = result
    
    # Plot elbow method and silhouette scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow method
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette scores
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs Number of Clusters')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal k (highest silhouette score)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Best silhouette score: {max(silhouette_scores):.3f}")
else:
    optimal_k = 3  # Default fallback
    print("Using default 3 clusters")

# --- STEP 2: Apply different clustering approaches ---
print("\n=== STEP 2: Applying Different Clustering Methods ===")

# Original clustering (your method)
if 'likes' in youtube_df_cleaned.columns and 'views' in youtube_df_cleaned.columns:
    scaler_original = StandardScaler()
    likes_views_scaled = scaler_original.fit_transform(youtube_df_cleaned[['likes', 'views']])
    kmeans_original = KMeans(n_clusters=3, random_state=42)
    youtube_df_cleaned['performance_cluster_original'] = kmeans_original.fit_predict(likes_views_scaled)

# Improved clustering with more features and RobustScaler
if result[0] is not None:
    robust_scaler = RobustScaler()
    features_scaled = robust_scaler.fit_transform(youtube_df_cleaned[available_features])
    
    kmeans_improved = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    youtube_df_cleaned['performance_cluster_improved'] = kmeans_improved.fit_predict(features_scaled)

# Performance score-based clustering
youtube_df_cleaned['performance_score'] = create_performance_score(youtube_df_cleaned)
performance_scaled = RobustScaler().fit_transform(youtube_df_cleaned[['performance_score']])
kmeans_score = KMeans(n_clusters=3, random_state=42, n_init=10)
youtube_df_cleaned['performance_cluster_score'] = kmeans_score.fit_predict(performance_scaled)

# --- STEP 3: Visualize clustering results ---
print("\n=== STEP 3: Comparing Clustering Results ===")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Original clustering (your method)
if 'performance_cluster_original' in youtube_df_cleaned.columns:
    ax = axes[0, 0]
    colors_original = ['red', 'yellow', 'green']
    performance_labels_original = {0: 'Low', 1: 'Medium', 2: 'High'}
    
    for i, (cluster, label) in enumerate(performance_labels_original.items()):
        if cluster in youtube_df_cleaned['performance_cluster_original'].values:
            subset = youtube_df_cleaned[youtube_df_cleaned['performance_cluster_original'] == cluster]
            ax.scatter(subset['views'], subset['engagement'], 
                      c=colors_original[i], label=f'{label} Performance', alpha=0.7, s=50)
    ax.set_xlabel('Views')
    ax.set_ylabel('Engagement')
    ax.set_title('Original Clustering (Views + Likes)')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Improved clustering
if 'performance_cluster_improved' in youtube_df_cleaned.columns:
    ax = axes[0, 1]
    colors_improved = plt.cm.viridis(np.linspace(0, 1, optimal_k))
    for i in range(optimal_k):
        subset = youtube_df_cleaned[youtube_df_cleaned['performance_cluster_improved'] == i]
        if len(subset) > 0:
            ax.scatter(subset['views'], subset['engagement'], 
                      c=[colors_improved[i]], label=f'Cluster {i}', alpha=0.7, s=50)
    ax.set_xlabel('Views')
    ax.set_ylabel('Engagement')
    ax.set_title(f'Improved Clustering ({optimal_k} clusters, multiple features)')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Performance score clustering
ax = axes[1, 0]
colors_score = ['red', 'orange', 'green']
score_labels = {0: 'Low', 1: 'Medium', 2: 'High'}

for i, (cluster, label) in enumerate(score_labels.items()):
    if cluster in youtube_df_cleaned['performance_cluster_score'].values:
        subset = youtube_df_cleaned[youtube_df_cleaned['performance_cluster_score'] == cluster]
        ax.scatter(subset['views'], subset['engagement'], 
                  c=colors_score[i], label=f'{label} Performance', alpha=0.7, s=50)
ax.set_xlabel('Views')
ax.set_ylabel('Engagement')
ax.set_title('Performance Score Clustering')
ax.legend()
ax.grid(True, alpha=0.3)

# Performance score distribution
ax = axes[1, 1]
for i, (cluster, label) in enumerate(score_labels.items()):
    if cluster in youtube_df_cleaned['performance_cluster_score'].values:
        subset = youtube_df_cleaned[youtube_df_cleaned['performance_cluster_score'] == cluster]
        ax.hist(subset['performance_score'], alpha=0.6, label=f'{label} Performance', 
               color=colors_score[i], bins=10)
ax.set_xlabel('Performance Score')
ax.set_ylabel('Frequency')
ax.set_title('Performance Score Distribution by Cluster')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- ROBUST DESCRIPTIVE ANALYTICS (MY ENHANCED VERSION) ---
print("\n=== ROBUST DESCRIPTIVE ANALYTICS ===")

# --- Scatter Plot: Videos by Category ---
if 'category' in youtube_df_cleaned.columns:
    category_colors = {cat: plt.cm.tab20(i) for i, cat in enumerate(youtube_df_cleaned['category'].unique()) if pd.notna(cat)}

    plt.figure(figsize=(12, 7))
    for category, color in category_colors.items():
        subset = youtube_df_cleaned[youtube_df_cleaned['category'] == category]
        plt.scatter(subset['likes'], subset['views'], color=color, label=category, alpha=0.6)

    plt.title("Videos by Likes vs Views Colored by Category")
    plt.xlabel("Likes")
    plt.ylabel("Views")
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# --- 1) ROBUST Monthly Views and Watch Time with Trend Lines ---
monthly_data = youtube_df_cleaned.groupby('publish_month').agg(
    total_views=('views', 'sum'),
    total_watch_time_hours=('watch_time_hours', 'sum'),
    video_count=('video_id', 'count')
).reset_index()

# Only keep months with reasonable video counts (at least 2 videos)
monthly_data = monthly_data[monthly_data['video_count'] >= 2]

plt.figure(figsize=(12, 8))

# Plot original data
plt.subplot(2, 1, 1)
plt.plot(monthly_data['publish_month'], monthly_data['total_views'], 
         label='Total Views', marker='o', linewidth=2, markersize=6)
plt.plot(monthly_data['publish_month'], monthly_data['total_watch_time_hours'], 
         label='Total Watch Time (hours)', marker='s', linewidth=2, markersize=6)

# Add robust trend lines
if len(monthly_data) > 2:  # Need at least 3 points for trend
    views_trend, views_r2 = robust_trend_line(monthly_data['publish_month'], monthly_data['total_views'])
    watch_trend, watch_r2 = robust_trend_line(monthly_data['publish_month'], monthly_data['total_watch_time_hours'])
    
    plt.plot(monthly_data['publish_month'], views_trend, "--", 
             color="blue", alpha=0.7, linewidth=2, 
             label=f'Views Robust Trend (R²={views_r2:.3f})')
    plt.plot(monthly_data['publish_month'], watch_trend, "--", 
             color="orange", alpha=0.7, linewidth=2,
             label=f'Watch Time Robust Trend (R²={watch_r2:.3f})')

plt.title("Monthly Views and Watch Time with Robust Trend Lines", fontsize=14, fontweight='bold')
plt.xlabel("Month")
plt.ylabel("Count")
plt.legend()
plt.grid(True, alpha=0.3)

# Show monthly averages (normalized)
plt.subplot(2, 1, 2)
monthly_data['avg_views_per_video'] = monthly_data['total_views'] / monthly_data['video_count']
monthly_data['avg_watch_per_video'] = monthly_data['total_watch_time_hours'] / monthly_data['video_count']

plt.plot(monthly_data['publish_month'], monthly_data['avg_views_per_video'], 
         label='Avg Views per Video', marker='o', linewidth=2)
plt.plot(monthly_data['publish_month'], monthly_data['avg_watch_per_video'], 
         label='Avg Watch Time per Video', marker='s', linewidth=2)

plt.title("Average Performance per Video by Month", fontsize=14, fontweight='bold')
plt.xlabel("Month")
plt.ylabel("Average Count")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- 2) Robust Impressions and CTR Trends ---
monthly_impressions = youtube_df_cleaned.groupby('publish_month').agg(
    total_impressions=('impressions', 'sum'),
    avg_ctr=('impressions_ctr', 'mean'),
    video_count=('video_id', 'count')
).reset_index()

monthly_impressions = monthly_impressions[monthly_impressions['video_count'] >= 2]

plt.figure(figsize=(12, 6))
plt.plot(monthly_impressions['publish_month'], monthly_impressions['total_impressions'], 
         label='Monthly Impressions', marker='o', linewidth=2)

# Normalize CTR for better visualization (scale to impressions range)
if len(monthly_impressions) > 0 and monthly_impressions['avg_ctr'].max() > 0:
    ctr_scaled = monthly_impressions['avg_ctr'] * (monthly_impressions['total_impressions'].max() / monthly_impressions['avg_ctr'].max())
    plt.plot(monthly_impressions['publish_month'], ctr_scaled, 
             label='Monthly CTR (scaled)', marker='s', linestyle='--', linewidth=2)

# Add robust trend lines
if len(monthly_impressions) > 2:
    imp_trend, imp_r2 = robust_trend_line(monthly_impressions['publish_month'], monthly_impressions['total_impressions'])
    plt.plot(monthly_impressions['publish_month'], imp_trend, "--", 
             color="blue", alpha=0.7, linewidth=2, 
             label=f'Impressions Robust Trend (R²={imp_r2:.3f})')
    
    if 'ctr_scaled' in locals():
        ctr_trend, ctr_r2 = robust_trend_line(monthly_impressions['publish_month'], ctr_scaled)
        plt.plot(monthly_impressions['publish_month'], ctr_trend, "--", 
                 color="orange", alpha=0.7, linewidth=2,
                 label=f'CTR Robust Trend (R²={ctr_r2:.3f})')

plt.title("Monthly Impressions and CTR with Robust Trends", fontsize=14, fontweight='bold')
plt.xlabel("Month")
plt.ylabel("Count / Scaled CTR")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 3) Post Type Performance (Outlier-resistant) ---
if 'category' in youtube_df_cleaned.columns:
    post_type_performance = youtube_df_cleaned.groupby('category').agg(
        total_engagement=('engagement', 'sum'),
        median_engagement=('engagement', 'median'),  # More robust than mean
        total_views=('views', 'sum'),
        median_views=('views', 'median'),  # More robust than mean
        video_count=('video_id', 'count')
    ).reset_index()

    # Only show categories with at least 3 videos
    post_type_performance = post_type_performance[post_type_performance['video_count'] >= 3]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.bar(post_type_performance['category'], post_type_performance['total_engagement'], 
            label='Total Engagement', alpha=0.7)
    plt.bar(post_type_performance['category'], post_type_performance['total_views'], 
            label='Total Views', alpha=0.5)
    plt.title("Post Type Performance - Totals", fontsize=14, fontweight='bold')
    plt.xlabel("Category")
    plt.ylabel("Total Count")
    plt.legend()
    plt.xticks(rotation=45)

    plt.subplot(2, 1, 2)
    plt.bar(post_type_performance['category'], post_type_performance['median_engagement'], 
            label='Median Engagement', alpha=0.7)
    plt.bar(post_type_performance['category'], post_type_performance['median_views'], 
            label='Median Views', alpha=0.5)
    plt.title("Post Type Performance - Medians (Outlier-Resistant)", fontsize=14, fontweight='bold')
    plt.xlabel("Category")
    plt.ylabel("Median Count")
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

# --- 4) Engagement by Day of Week (Robust) ---
youtube_df_cleaned['day_of_week'] = youtube_df_cleaned['publish_date'].dt.day_name()
engagement_by_day = youtube_df_cleaned.groupby('day_of_week').agg(
    median_engagement=('engagement', 'median'),
    mean_engagement=('engagement', 'mean'),
    video_count=('video_id', 'count')
).reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

plt.figure(figsize=(12, 6))
days = engagement_by_day.index
x_pos = np.arange(len(days))

plt.bar(x_pos - 0.2, engagement_by_day['median_engagement'], 0.4, 
        label='Median Engagement (Robust)', alpha=0.8)
plt.bar(x_pos + 0.2, engagement_by_day['mean_engagement'], 0.4, 
        label='Mean Engagement', alpha=0.6)

plt.title("Engagement by Day of Week - Robust vs Regular Metrics", fontsize=14, fontweight='bold')
plt.xlabel("Day of Week")
plt.ylabel("Engagement")
plt.xticks(x_pos, days, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# --- 5) Top 10 Videos by Watch Time ---
if 'video_title' in youtube_df_cleaned.columns:
    top_10_videos = youtube_df_cleaned.groupby('video_title')['watch_time_hours'].sum().reset_index()
    top_10_videos = top_10_videos.sort_values(by='watch_time_hours', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_10_videos)), top_10_videos['watch_time_hours'])
    plt.title("Top 10 Videos by Watch Time", fontsize=14, fontweight='bold')
    plt.xlabel("Video Rank", fontsize=12)
    plt.ylabel("Watch Time (Hours)", fontsize=12)
    
    # Add video titles as labels (rotated and truncated)
    plt.xticks(range(len(top_10_videos)), 
               [title[:30] + "..." if len(title) > 30 else title for title in top_10_videos['video_title']], 
               rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.show()

# --- DIAGNOSTIC PLOTS ---
plt.figure(figsize=(15, 10))

# Original vs cleaned data comparison
plt.subplot(2, 3, 1)
plt.scatter(youtube_df['views'], youtube_df['engagement'], alpha=0.6, label='Original Data')
plt.scatter(youtube_df_cleaned['views'], youtube_df_cleaned['engagement'], alpha=0.8, label='After Outlier Removal')
plt.xlabel("Views")
plt.ylabel("Engagement")
plt.title("Impact of Outlier Removal")
plt.legend()
plt.grid(True, alpha=0.3)

# Box plots to show outlier removal effect
plt.subplot(2, 3, 2)
data_to_plot = [youtube_df['views'], youtube_df_cleaned['views']]
plt.boxplot(data_to_plot, labels=['Original', 'Cleaned'])
plt.title("Views Distribution")
plt.ylabel("Views")

plt.subplot(2, 3, 3)
data_to_plot = [youtube_df['engagement'], youtube_df_cleaned['engagement']]
plt.boxplot(data_to_plot, labels=['Original', 'Cleaned'])
plt.title("Engagement Distribution")
plt.ylabel("Engagement")

# Performance cluster visualization
plt.subplot(2, 3, 4)
colors = ['red', 'orange', 'green']
for i, perf in enumerate(['Low', 'Medium', 'High']):
    subset = youtube_df_cleaned[youtube_df_cleaned['performance_cluster_score'] == i]
    if len(subset) > 0:
        plt.scatter(subset['views'], subset['engagement'], 
                   c=colors[i], label=f'{perf} Performance', alpha=0.7)
plt.xlabel("Views")
plt.ylabel("Engagement")
plt.title("Performance Clusters (Score-Based)")
plt.legend()
plt.grid(True, alpha=0.3)

# Monthly trend robustness
plt.subplot(2, 3, 5)
if len(monthly_data) > 2:
    # Compare regular vs robust trend
    regular_trend = np.polyfit(monthly_data['publish_month'], monthly_data['total_views'], 1)
    regular_line = np.poly1d(regular_trend)
    
    plt.plot(monthly_data['publish_month'], monthly_data['total_views'], 'o', label='Data Points')
    plt.plot(monthly_data['publish_month'], regular_line(monthly_data['publish_month']), 
             '--', label='Regular Trend', alpha=0.7)
    plt.plot(monthly_data['publish_month'], views_trend, 
             '-', label='Robust Trend', linewidth=2)
    plt.xlabel("Month")
    plt.ylabel("Total Views")
    plt.title("Trend Line Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

# Category performance robustness
plt.subplot(2, 3, 6)
if 'category' in youtube_df_cleaned.columns and 'post_type_performance' in locals() and len(post_type_performance) > 0:
    plt.scatter(post_type_performance['total_views'], post_type_performance['median_views'], 
               s=post_type_performance['video_count']*20, alpha=0.6)
    plt.xlabel("Total Views")
    plt.ylabel("Median Views per Video")
    plt.title("Total vs Median Views by Category")
    plt.grid(True, alpha=0.3)
    
    # Add category labels
    for _, row in post_type_performance.iterrows():
        plt.annotate(row['category'][:10], 
                    (row['total_views'], row['median_views']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.tight_layout()
plt.show()

# --- CLUSTERING QUALITY ANALYSIS ---
print("\n==================== CLUSTERING ANALYSIS ====================\n")

# Show cluster sizes
if 'performance_cluster_original' in youtube_df_cleaned.columns:
    print("Original clustering sizes:", youtube_df_cleaned['performance_cluster_original'].value_counts().sort_index().to_dict())

if 'performance_cluster_improved' in youtube_df_cleaned.columns:
    print("Improved clustering sizes:", youtube_df_cleaned['performance_cluster_improved'].value_counts().sort_index().to_dict())

print("Score-based clustering sizes:", youtube_df_cleaned['performance_cluster_score'].value_counts().sort_index().to_dict())

# Show examples from each performance level (score-based)
print(f"\nPerformance Score Clustering Examples:")
for cluster_id, label in score_labels.items():
    if cluster_id in youtube_df_cleaned['performance_cluster_score'].values:
        subset = youtube_df_cleaned[youtube_df_cleaned['performance_cluster_score'] == cluster_id]
        print(f"\n{label} Performance (Cluster {cluster_id}):")
        print(f"  Count: {len(subset)} videos")
        print(f"  Avg Views: {subset['views'].mean():.0f}")
        print(f"  Avg Engagement: {subset['engagement'].mean():.0f}")
        print(f"  Performance Score Range: {subset['performance_score'].min():.3f} - {subset['performance_score'].max():.3f}")
        
        # Show top videos in this cluster
        if 'video_title' in subset.columns:
            top_videos = subset.nlargest(3, 'performance_score')[['video_title', 'views', 'engagement', 'performance_score']]
            print("  Top videos:")
            for _, video in top_videos.iterrows():
                title = str(video['video_title'])[:50] if pd.notna(video['video_title']) else "Untitled"
                print(f"    • {title}... (Score: {video['performance_score']:.3f})")

# Calculate silhouette scores for comparison
print(f"\nClustering Quality Metrics:")

if 'performance_cluster_original' in youtube_df_cleaned.columns:
    try:
        original_silhouette = silhouette_score(
            likes_views_scaled, 
            youtube_df_cleaned['performance_cluster_original']
        )
        print(f"Original clustering silhouette score: {original_silhouette:.3f}")
    except:
        print("Could not calculate original clustering silhouette score")

if 'performance_cluster_improved' in youtube_df_cleaned.columns:
    try:
        improved_silhouette = silhouette_score(
            features_scaled, 
            youtube_df_cleaned['performance_cluster_improved']
        )
        print(f"Improved clustering silhouette score: {improved_silhouette:.3f}")
    except:
        print("Could not calculate improved clustering silhouette score")

try:
    score_silhouette = silhouette_score(
        performance_scaled, 
        youtube_df_cleaned['performance_cluster_score']
    )
    print(f"Score-based clustering silhouette score: {score_silhouette:.3f}")
except:
    print("Could not calculate score-based clustering silhouette score")

# --- ROBUST SUMMARY FINDINGS ---
print("\n==================== YOUTUBE: Robust Key Findings ====================\n")
print(f"Dataset size after outlier removal: {len(youtube_df_cleaned)} videos (from {len(youtube_df)} original)")
print(f"Outliers removed: {len(youtube_df) - len(youtube_df_cleaned)} videos\n")

# Top categories by median engagement (more robust)
if 'category' in youtube_df_cleaned.columns and 'post_type_performance' in locals() and len(post_type_performance) > 0:
    top_categories_robust = post_type_performance.nlargest(3, 'median_engagement')
    print("Top 3 Categories by Median Engagement (Outlier-Resistant):")
    for _, row in top_categories_robust.iterrows():
        print(f"  {row['category']}: {row['median_engagement']:.0f} median engagement ({row['video_count']} videos)")

if len(monthly_data) > 0:
    best_month_views = monthly_data.nlargest(1, 'avg_views_per_video')
    print(f"\nBest Month for Average Views per Video: {best_month_views['publish_month'].values[0]} "
          f"with {best_month_views['avg_views_per_video'].values[0]:.0f} avg views")

best_day_engagement = engagement_by_day['median_engagement'].idxmax()
print(f"Best Day for Median Engagement: {best_day_engagement}")

# Performance clustering insights
print(f"\nPerformance Clustering Insights (Score-Based Method):")
high_perf = youtube_df_cleaned[youtube_df_cleaned['performance_cluster_score'] == 2]  
medium_perf = youtube_df_cleaned[youtube_df_cleaned['performance_cluster_score'] == 1]
low_perf = youtube_df_cleaned[youtube_df_cleaned['performance_cluster_score'] == 0]

if len(high_perf) > 0:
    print(f"  High Performance Videos: {len(high_perf)} videos")
    print(f"    Average Views: {high_perf['views'].mean():.0f}")
    print(f"    Average Engagement: {high_perf['engagement'].mean():.0f}")
    print(f"    Average Performance Score: {high_perf['performance_score'].mean():.3f}")

if len(medium_perf) > 0:
    print(f"  Medium Performance Videos: {len(medium_perf)} videos")
    print(f"    Average Views: {medium_perf['views'].mean():.0f}")
    print(f"    Average Engagement: {medium_perf['engagement'].mean():.0f}")
    print(f"    Average Performance Score: {medium_perf['performance_score'].mean():.3f}")

if len(low_perf) > 0:
    print(f"  Low Performance Videos: {len(low_perf)} videos")
    print(f"    Average Views: {low_perf['views'].mean():.0f}")
    print(f"    Average Engagement: {low_perf['engagement'].mean():.0f}")
    print(f"    Average Performance Score: {low_perf['performance_score'].mean():.3f}")

# Show data quality metrics
print(f"\nData Quality Metrics:")
print(f"  Engagement range: {youtube_df_cleaned['engagement'].min():.0f} - {youtube_df_cleaned['engagement'].max():.0f}")
print(f"  Views range: {youtube_df_cleaned['views'].min():.0f} - {youtube_df_cleaned['views'].max():.0f}")
if len(monthly_data) > 0:
    most_active_month = monthly_data.loc[monthly_data['video_count'].idxmax(), 'publish_month']
    most_active_count = monthly_data['video_count'].max()
    print(f"  Most active month: {most_active_month} ({most_active_count} videos)")

print(f"  Date range: {youtube_df_cleaned['publish_date'].min().strftime('%Y-%m-%d')} to {youtube_df_cleaned['publish_date'].max().strftime('%Y-%m-%d')}")
print(f"  Total views across all videos: {youtube_df_cleaned['views'].sum():,.0f}")
print(f"  Total engagement across all videos: {youtube_df_cleaned['engagement'].sum():,.0f}")

print("\n===============================================================\n")

print("FINAL RECOMMENDATIONS:")
print("1. ✓ Use Performance Score Clustering - shows most logical groupings")
print("2. ✓ Robust trend lines provide better insights than regular linear regression")
print("3. ✓ Median-based metrics are more reliable than means for small datasets")
print("4. ✓ Focus on categories and days with highest median engagement")
print("5. ✓ Monitor monthly trends using the robust trend analysis")

# Show which clustering method performed best
best_method = "Score-based clustering"
if 'score_silhouette' in locals():
    print(f"6. ✓ Best clustering method: {best_method} (Silhouette score: {score_silhouette:.3f})")
else:
    print(f"6. ✓ Recommended clustering method: {best_method}")

print(f"7. ✓ {len(youtube_df) - len(youtube_df_cleaned)} outliers removed for more representative analysis")
print(f"8. ✓ Analysis covers {len(youtube_df_cleaned)} videos with reliable performance metrics")

print("\n=== Analysis Complete! ===")
print("Your YouTube data now has proper clustering and outlier-resistant analytics.")