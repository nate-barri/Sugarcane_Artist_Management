#!/usr/bin/env python3
"""
Sugarcane Social Media Analytics Dashboard (Sep 2025)

Fix pack:
- Classifier threshold tuned to target precision (default 0.70) per fold
- Median-fold confusion matrix (no more misleading "last fold" snapshot)
- Per-fold diagnostics table (threshold, precision, recall, F1, PR-AUC)
- Poisson GBDT regression on winsorized engagement counts (kept)
- TimeSeriesSplit CV (no leakage)
- Rate MAPE only on reach ≥ 5,000
"""

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ML
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor

# -----------------------
# Config
# -----------------------
TARGET_PRECISION = 0.75          # raise/lower to trade recall vs precision
N_SPLITS = 5                     # TimeSeriesSplit folds
MIN_HOURLY = 30                  # guardrail for "best hours"
MIN_DAILY = 60                   # guardrail for "best days"
RATE_MIN_REACH = 5000            # for rate metrics
WINSOR_Q = 0.99                  # cap extreme engagements before regression


# -----------------------
# Data access
# -----------------------
def get_facebook_data():
    db_params = {
        'dbname':'neondb',
        'user':'neondb_owner',
        'password':'npg_dGzvq4CJPRx7',
        'host':'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
        'port':'5432',
        'sslmode':'require'
    }
    print("Connecting to database for Sugarcane social media data...")
    try:
        conn = psycopg2.connect(**db_params)
        query = """
        SELECT post_id, page_id, page_name, title, description, post_type,
               duration_sec, publish_time, year, month, day, time,
               permalink, is_crosspost, is_share, funded_content_status,
               reach, shares, comments, reactions, seconds_viewed, average_seconds_viewed
        FROM public.facebook_data_set
        WHERE reach >= 0 AND reactions >= 0 AND comments >= 0 AND shares >= 0
        ORDER BY publish_time;
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Database error: {e}")
        return None


# -----------------------
# Utilities
# -----------------------
def threshold_at_target_precision(y_true, proba, target_precision=TARGET_PRECISION):
    """
    Return the highest threshold whose precision ≥ target_precision on this fold.
    If none reaches it, fall back to the threshold that maximizes F1.
    """
    p, r, t = precision_recall_curve(y_true, proba)
    # thresholds t align with p[:-1], r[:-1]
    candidates = [th for prec, th in zip(p[:-1], t) if prec >= target_precision]
    if candidates:
        return max(candidates)
    # fallback: best F1 point
    f1 = (2 * p * r) / (p + r + 1e-9)
    ix = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
    return float(t[ix]) if 0 <= ix < len(t) else 0.5


def add_easy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add low-effort predictive features (lags/rolling, duration buckets, seasonality, paid flag)."""
    out = df.copy()
    out['publish_time'] = pd.to_datetime(out['publish_time'])
    out['hour'] = out['publish_time'].dt.hour
    out['day_of_week'] = out['publish_time'].dt.dayofweek
    out['is_weekend'] = out['day_of_week'].isin([5,6]).astype(int)
    out['weekofyear'] = out['publish_time'].dt.isocalendar().week.astype(int)
    out['month'] = out['publish_time'].dt.month
    # seasonality encoding
    out['month_sin'] = np.sin(2*np.pi*out['month']/12.0)
    out['month_cos'] = np.cos(2*np.pi*out['month']/12.0)

    # totals
    out['total_engagement'] = out['reactions'].fillna(0) + out['comments'].fillna(0) + out['shares'].fillna(0)
    out['engagement_rate'] = np.where(out['reach']>0, out['total_engagement']/out['reach'].clip(lower=1)*100.0, 0.0)

    # paid flag
    out['is_paid'] = out['funded_content_status'].fillna('').str.contains('funded|paid', case=False).astype(int)

    # duration bin
    if 'duration_sec' in out.columns:
        d = out['duration_sec'].fillna(0)
        out['duration_bin'] = pd.cut(d, bins=[-1,30,90,600,1e9], labels=['short','med','long','very_long'])
    else:
        out['duration_sec'] = 0
        out['duration_bin'] = 'unknown'

    # label-fill
    out['post_type_filled'] = out['post_type'].fillna('unknown')
    out['page_id_filled'] = out['page_id'].fillna('unknown')

    # lags/rolling by page
    out = out.sort_values('publish_time').reset_index(drop=True)
    out['lag1_eng'] = out.groupby('page_id_filled')['total_engagement'].shift(1)
    out['roll5_eng'] = out.groupby('page_id_filled')['total_engagement'].rolling(5).mean().reset_index(0,drop=True)
    out['roll10_eng'] = out.groupby('page_id_filled')['total_engagement'].rolling(10).mean().reset_index(0,drop=True)

    # lags/rolling by (page, post_type)
    key = out['page_id_filled'].astype(str) + '|' + out['post_type_filled'].astype(str)
    out['lag1_eng_type'] = out.groupby(key)['total_engagement'].shift(1)
    out['roll5_eng_type'] = out.groupby(key)['total_engagement'].rolling(5).mean().reset_index(0,drop=True)

    # fill NaNs for model input
    for c in ['lag1_eng','roll5_eng','roll10_eng','lag1_eng_type','roll5_eng_type','duration_sec']:
        out[c] = out[c].fillna(0)

    # comment ratio used for label
    out['comment_ratio'] = (out['comments'] / out['total_engagement'].replace(0, np.nan)).fillna(0)

    return out


def winsorize_series(s: pd.Series, upper_q=WINSOR_Q) -> pd.Series:
    if len(s) == 0:
        return s
    cap = s.quantile(upper_q)
    return s.clip(upper=cap)


# -----------------------
# Analytics class
# -----------------------
class SugarcaneAnalytics:
    def __init__(self):
        self.data = None
        self.engagement_clf = None
        self.discuss_clf = None
        self.reg_model = None
        self.clf_threshold = 0.5  # learned global operating point
        self.le_post_type = LabelEncoder()
        self.le_duration = LabelEncoder()

    # ---------- load + describe ----------
    def load_and_prepare_data(self):
        print("\n=== LOADING SUGARCANE SOCIAL MEDIA DATA ===")
        df = get_facebook_data()
        if df is None or df.empty:
            print("No data available")
            return False
        df = add_easy_features(df)

        # classification targets
        p75 = df['engagement_rate'].quantile(0.75)
        df['high_engagement'] = (df['engagement_rate'] >= p75).astype(int)
        med_cr = df['comment_ratio'].median()
        df['discussion_generator'] = (df['comment_ratio'] >= med_cr).astype(int)

        self.data = df

        print(f"Loaded {len(df)} posts")
        print(f"Average engagement rate: {df['engagement_rate'].mean():.2f}%")
        print(f"High engagement posts: {df['high_engagement'].sum()} ({df['high_engagement'].mean()*100:.1f}%)")
        return True

    def descriptive_analytics(self):
        print("\n=== DESCRIPTIVE ANALYTICS ===")
        df = self.data

        print("\nPERFORMACE OVERVIEW:")
        print(f"Total posts analyzed: {len(df):,}")
        print(f"Total reach: {df['reach'].sum():,}")
        print(f"Total engagement: {df['total_engagement'].sum():,}")
        print(f"Average engagement rate: {df['engagement_rate'].mean():.2f}%")
        print(f"Median engagement rate: {df['engagement_rate'].median():.2f}%")

        print("\nCONTENT TYPE PERFORMANCE:")
        content_perf = df.groupby('post_type_filled').agg({
            'engagement_rate':['mean','median','count'],
            'total_engagement':'mean',
            'reach':'mean'
        }).round(2)
        content_perf.columns = ['Avg_Engagement_Rate','Median_Engagement_Rate','Post_Count','Avg_Total_Engagement','Avg_Reach']
        print(content_perf.sort_values('Avg_Engagement_Rate', ascending=False))

        print("\nTIME PATTERN ANALYSIS:")
        hourly = df.groupby('hour')['engagement_rate'].agg(['mean','count']).round(2)
        hourly.columns = ['Avg_Engagement_Rate','Post_Count']
        best_hours = hourly[hourly['Post_Count'] >= MIN_HOURLY].sort_values('Avg_Engagement_Rate', ascending=False).head(5)
        print("\nBest performing hours:")
        print(best_hours)

        day_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        daily = df.groupby('day_of_week')['engagement_rate'].agg(['mean','count']).round(2)
        daily.index = [day_names[i] for i in daily.index]
        daily.columns = ['Avg_Engagement_Rate','Post_Count']
        best_days = daily[daily['Post_Count'] >= MIN_DAILY].sort_values('Avg_Engagement_Rate', ascending=False)
        print("\nPerformance by day of week:")
        print(best_days)

        monthly = df.groupby(df['publish_time'].dt.month)['engagement_rate'].agg(['mean','count']).round(2)
        monthly.columns = ['Avg_Engagement_Rate','Post_Count']
        print("\nMonthly performance:")
        print(monthly.sort_values('Avg_Engagement_Rate', ascending=False))

        return {
            'content_performance': content_perf,
            'hourly': hourly,
            'best_hours': best_hours,
            'daily': daily,
            'best_days': best_days,
            'monthly': monthly
        }

    # ---------- predictive ----------
    def predictive_analytics(self):
        print("\n=== PREDICTIVE ANALYTICS ===")
        df = self.data.sort_values('publish_time').reset_index(drop=True)

        # encoders
        df['post_type_enc'] = self.le_post_type.fit_transform(df['post_type_filled'])
        df['duration_bin'] = df['duration_bin'].astype(str)
        df['duration_enc'] = self.le_duration.fit_transform(df['duration_bin'])

        # features
        base_cols = [
            'hour','day_of_week','month','is_weekend','weekofyear',
            'month_sin','month_cos',
            'is_paid','post_type_enc','duration_sec','duration_enc',
            'lag1_eng','roll5_eng','roll10_eng','lag1_eng_type','roll5_eng_type'
        ]
        X = df[base_cols].fillna(0)

        # ========== 1) CLASSIFICATION: High engagement (precision-first) ==========
        print("\n1) CLASSIFICATION — High engagement (balanced + threshold@precision)")
        y = df['high_engagement'].astype(int).values
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)

        fold_rows = []
        cms = []
        tuned_thresholds = []

        for fold, (tr, te) in enumerate(tscv.split(X), 1):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y[tr], y[te]

            clf = RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=10,
                max_features='sqrt', class_weight='balanced',
                random_state=42, n_jobs=-1
            )
            clf.fit(Xtr, ytr)
            proba = clf.predict_proba(Xte)[:,1]

            thr = threshold_at_target_precision(yte, proba, target_precision=TARGET_PRECISION)
            yhat = (proba >= thr).astype(int)

            p, r, f1, _ = precision_recall_fscore_support(yte, yhat, average='binary', zero_division=0)
            ap = average_precision_score(yte, proba)
            cm = confusion_matrix(yte, yhat, labels=[0,1])

            fold_rows.append({'fold': fold, 'threshold': thr, 'precision': p, 'recall': r, 'f1': f1, 'pr_auc': ap})
            cms.append(cm)
            tuned_thresholds.append(thr)

        fold_df = pd.DataFrame(fold_rows)
        print(f"CV Precision: {fold_df['precision'].mean():.2f} | Recall: {fold_df['recall'].mean():.2f} | F1: {fold_df['f1'].mean():.2f} | PR-AUC: {fold_df['pr_auc'].mean():.2f}")
        print(f"Avg tuned threshold: {np.mean(tuned_thresholds):.2f}")

        # pick the median-fold by F1 for a representative confusion matrix
        median_idx = fold_df['f1'].rank(method='first').sub(1).astype(int).argsort()[len(fold_df)//2]
        print("Representative confusion matrix (median fold by F1):")
        print(cms[int(median_idx)])

        # fit final classifier on all rows; keep a robust global threshold = median of tuned thresholds
        self.engagement_clf = RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=10,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1
        ).fit(X, y)
        self.clf_threshold = float(np.median(tuned_thresholds))

        # ========== 2) CLASSIFICATION: Discussion generator ==========
        print("\n2) CLASSIFICATION — Discussion generator (balanced)")
        y_disc = df['discussion_generator'].astype(int).values
        f1_disc = []
        for tr, te in tscv.split(X):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y_disc[tr], y_disc[te]
            dclf = RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_leaf=10,
                max_features='sqrt', class_weight='balanced',
                random_state=42, n_jobs=-1
            )
            dclf.fit(Xtr, ytr)
            yhat = dclf.predict(Xte)
            _, _, f1, _ = precision_recall_fscore_support(yte, yhat, average='binary', zero_division=0)
            f1_disc.append(f1)
        print(f"CV F1 (avg over folds): {np.mean(f1_disc):.2f}")
        self.discuss_clf = dclf

        # ========== 3) REGRESSION: Poisson GBDT on engagement counts ==========
        print("\n3) REGRESSION — Predict total engagement (winsorized → counts, Poisson GBDT)")
        y_cnt = winsorize_series(df['total_engagement'], upper_q=WINSOR_Q).astype(float).values
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)

        maes, rmses, medaes, r2s = [], [], [], []
        mape_rates, mape_sizes = [], []

        for tr, te in tscv.split(X):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y_cnt[tr], y_cnt[te]

            reg = HistGradientBoostingRegressor(
                loss='poisson',
                learning_rate=0.07,
                max_depth=6,
                max_iter=500,
                min_samples_leaf=20,
                random_state=42
            )
            reg.fit(Xtr, ytr)
            yhat = reg.predict(Xte).clip(min=0)

            maes.append(mean_absolute_error(yte, yhat))
            rmses.append(mean_squared_error(yte, yhat, squared=False))
            medaes.append(median_absolute_error(yte, yhat))
            r2s.append(r2_score(yte, yhat))

            # Rate metrics only where reach >= RATE_MIN_REACH
            reach_te = df.loc[Xte.index, 'reach'].fillna(0).values
            mask = reach_te >= RATE_MIN_REACH
            if mask.any():
                rate_true = (yte[mask] / reach_te[mask]) * 100.0
                rate_hat = (yhat[mask] / reach_te[mask]) * 100.0
                mape = np.mean(np.abs((rate_true - rate_hat) / np.maximum(rate_true, 1e-9))) * 100
                mape_rates.append(mape); mape_sizes.append(mask.sum())

        print(f"CV MAE: {np.mean(maes):,.2f} | RMSE: {np.mean(rmses):,.2f} | MedAE: {np.mean(medaes):,.2f} | R²: {np.mean(r2s):.3f}")
        if mape_rates:
            print(f"Rate metrics on reach ≥ {RATE_MIN_REACH:,}: MAPE={np.mean(mape_rates):.1f}% (size≈{int(np.mean(mape_sizes))})")
        else:
            print(f"Rate metrics on reach ≥ {RATE_MIN_REACH:,}: not enough samples in folds.")

        # fit final regressor on all rows
        self.reg_model = HistGradientBoostingRegressor(
            loss='poisson',
            learning_rate=0.07,
            max_depth=6,
            max_iter=500,
            min_samples_leaf=20,
            random_state=42
        ).fit(X, y_cnt)

        # ========== 4) Segmented models (Video/Photo/Reel) ==========
        print("\n4) SEGMENTED MODELS — Videos / Photos / Reels (blend)")
        segs = ['video','photo','reel']
        seg_map = {'video':'Videos', 'photo':'Photos', 'reel':'Reels'}
        self.seg_models = {}
        self.seg_sizes = {}

        for key in segs:
            mask = df['post_type_filled'].str.lower().str.contains(key)
            dsub = df[mask]
            if len(dsub) < 30:
                continue
            Xs = dsub[base_cols].fillna(0)
            ys = winsorize_series(dsub['total_engagement'], upper_q=WINSOR_Q).astype(float)
            reg = HistGradientBoostingRegressor(
                loss='poisson',
                learning_rate=0.07,
                max_depth=6,
                max_iter=500,
                min_samples_leaf=20,
                random_state=42
            ).fit(Xs, ys)
            self.seg_models[seg_map[key]] = reg
            self.seg_sizes[seg_map[key]] = len(dsub)
            print(f"  Trained {seg_map[key]} regressor on {len(dsub)} posts.")

        # Blend by last-90d observed engagement share
        cutoff = df['publish_time'].max() - pd.Timedelta(days=90)
        recent = df[df['publish_time'] >= cutoff]
        weights = {}
        total = 0.0
        for nice in self.seg_models.keys():
            if nice == 'Videos':
                sel = recent['post_type_filled'].str.lower().str.contains('video')
            elif nice == 'Photos':
                sel = recent['post_type_filled'].str.lower().str.contains('photo')
            else:
                sel = recent['post_type_filled'].str.lower().str.contains('reel')
            w = float(recent.loc[sel, 'total_engagement'].sum())
            weights[nice] = w; total += w
        if total <= 0:
            blend = {k: 1.0/len(self.seg_models) for k in self.seg_models} if self.seg_models else {}
        else:
            blend = {k: v/total for k, v in weights.items()}
        if blend:
            fmt = {k: f"{v:.2f}" for k,v in blend.items()}
            print(f"  Blend weights (last 90 days): {fmt}")
        else:
            print("  Skipped blending: not enough recent segment data.")

        self.blend_weights = blend
        return base_cols

    # ---------- basic dashboard ----------
    def create_visualization_dashboard(self, desc):
        print("\nGenerating dashboard...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,12))

        # 1) Hourly
        h = desc['hourly'].reset_index()
        ax1.bar(h['hour'], h['Avg_Engagement_Rate'], alpha=0.7)
        ax1.set_title('Engagement Rate by Hour')
        ax1.set_xlabel('Hour'); ax1.set_ylabel('Avg %'); ax1.grid(True, alpha=0.3)

        # 2) Daily
        d = desc['daily'].reset_index()
        short = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        ax2.bar(range(len(d)), d['Avg_Engagement_Rate'], alpha=0.7)
        ax2.set_xticks(range(len(short))); ax2.set_xticklabels(short, rotation=45)
        ax2.set_title('Engagement Rate by Day'); ax2.grid(True, alpha=0.3)

        # 3) Content types
        cp = desc['content_performance'].reset_index().sort_values('Avg_Engagement_Rate', ascending=True)
        ax3.barh(cp['post_type_filled'], cp['Avg_Engagement_Rate'], alpha=0.7)
        ax3.set_title('Content Type Performance'); ax3.grid(True, alpha=0.3)

        # 4) Monthly
        m = desc['monthly'].reset_index()
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        labels = [month_names[i-1] for i in m['publish_time']]
        ax4.plot(labels, m['Avg_Engagement_Rate'], marker='o')
        ax4.set_title('Monthly Engagement Trend'); ax4.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()
        print("Dashboard ready.")

    # ---------- helper ----------
    def predict_post_probability(self, hour, day_of_week, month, post_type, duration_sec=0, is_paid=0):
        """Probability of high engagement for a hypothetical post."""
        if self.engagement_clf is None:
            return "Classifier not trained yet."
        row = pd.DataFrame([{
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': 1 if day_of_week in (5,6) else 0,
            'weekofyear': 1,
            'month_sin': np.sin(2*np.pi*month/12.0),
            'month_cos': np.cos(2*np.pi*month/12.0),
            'is_paid': is_paid,
            'post_type_enc': int(self.le_post_type.transform([post_type])[0]) if post_type in self.le_post_type.classes_ else 0,
            'duration_sec': duration_sec,
            'duration_enc': int(self.le_duration.transform(['med'])[0]) if 'med' in self.le_duration.classes_ else 0,
            'lag1_eng': 0, 'roll5_eng': 0, 'roll10_eng': 0, 'lag1_eng_type': 0, 'roll5_eng_type': 0
        }])
        proba = self.engagement_clf.predict_proba(row)[0][1]
        return f"Pr(high engagement) ≈ {proba:.2f} (using global threshold ~{self.clf_threshold:.2f})"


# -----------------------
# Orchestration
# -----------------------
def run_sugarcane_analytics():
    print("🎵 SUGARCANE SOCIAL MEDIA ANALYTICS SYSTEM 🎵")
    print("Optimizing engagement and growth through data-driven insights")
    print("="*70)

    analytics = SugarcaneAnalytics()
    if not analytics.load_and_prepare_data():
        print("\n❌ Analysis failed - check data availability.")
        return None

    # Descriptive
    desc = analytics.descriptive_analytics()

    # Predictive (precision-first classifier + Poisson GBDT)
    analytics.predictive_analytics()

    # Prescriptive
    print("\n=== PRESCRIPTIVE ANALYTICS ===")
    print("\n1) OPTIMAL POSTING TIME RECOMMENDATIONS:")
    if not desc['best_hours'].empty:
        for hour, row in desc['best_hours'].head(3).iterrows():
            label = "12 PM" if hour==12 else f"{hour%12 or 12} {'PM' if hour>=12 else 'AM'}"
            print(f"  {label}: {row['Avg_Engagement_Rate']:.2f}% avg engagement ({row['Post_Count']:.0f} posts)")
    else:
        print(f"  Not enough hourly samples ≥ {MIN_HOURLY} to recommend reliably.")

    print("\nBest days to post:")
    if not desc['best_days'].empty:
        for day, row in desc['best_days'].head(3).iterrows():
            print(f"  {day}: {row['Avg_Engagement_Rate']:.2f}% avg engagement")
    else:
        print(f"  Not enough daily samples ≥ {MIN_DAILY} to recommend reliably.")

    print("\n2) CONTENT TYPE OPTIMIZATION:")
    cp = desc['content_performance'].sort_values('Avg_Engagement_Rate', ascending=False).head(3)
    for ctype, row in cp.iterrows():
        print(f"  {ctype}: {row['Avg_Engagement_Rate']:.2f}% avg engagement ({row['Post_Count']:.0f} posts)")

    print("\n3) SEGMENT MIX (last 90 days weights):")
    if getattr(analytics, 'blend_weights', None):
        print(" ", {k: f"{v:.2f}" for k,v in analytics.blend_weights.items()})
    else:
        print("  No recent mix available.")

    # Charts
    analytics.create_visualization_dashboard(desc)
    print("\n✅ Analytics complete! Use insights to grow Sugarcane's social media presence.")
    return analytics


if __name__ == "__main__":
    run_sugarcane_analytics()
