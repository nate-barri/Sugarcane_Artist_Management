"""
TikTok Predictive Analytics - Returns JSON only (no matplotlib)
"""
import sys
import json
import os
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Database connection using environment variables
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('PGDATABASE', 'neondb'),
            user=os.getenv('PGUSER', 'neondb_owner'),
            password=os.getenv('PGPASSWORD', 'npg_dGzvq4CJPRx7'),
            host=os.getenv('PGHOST', 'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech'),
            port=os.getenv('PGPORT', '5432'),
            sslmode='require'
        )
        return conn
    except Exception as e:
        raise Exception(f"Database connection failed: {str(e)}")

def fetch_tiktok_data(conn):
    """Fetch TikTok data for predictions"""
    query = """
    SELECT 
        video_id, title, views, likes, shares, comments_added, saves,
        duration_sec, post_type, sound_used, publish_year, publish_month, publish_day,
        publish_time,
        EXTRACT(DOW FROM publish_time) as day_of_week,
        EXTRACT(HOUR FROM publish_time) as hour_of_day
    FROM public.tt_video_etl
    WHERE views IS NOT NULL AND views > 0
        AND duration_sec IS NOT NULL
        AND publish_time IS NOT NULL
    ORDER BY publish_time DESC
    LIMIT 500
    """
    return pd.read_sql_query(query, conn)

def engineer_features(df):
    """Create features for prediction"""
    df = df.copy()
    df['engagement_rate'] = ((df.get('likes', 0) + df.get('shares', 0) + df.get('comments_added', 0) + df.get('saves', 0)) / df['views'] * 100).fillna(0)
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    df['is_weekend'] = df['day_of_week'].isin([0, 6]).astype(int)
    df['is_prime_time'] = df['hour_of_day'].between(17, 22).astype(int)
    df['has_sound'] = (df['sound_used'].notna() & (df['sound_used'] != '')).astype(int)
    df['post_type'] = df['post_type'].fillna('unknown')
    df['log_views'] = np.log1p(df['views'])
    return df

def predict_virality(df):
    """Predict virality (views >= 100K)"""
    le_post = LabelEncoder()
    df_model = df.copy()
    df_model['post_type_encoded'] = le_post.fit_transform(df_model['post_type'])
    
    feature_cols = ['duration_sec', 'publish_month', 'day_of_week', 'hour_of_day',
                   'is_weekend', 'is_prime_time', 'has_sound', 'post_type_encoded']
    
    X = df_model[feature_cols].fillna(0)
    y = (df_model['views'] >= 100000).astype(int)
    
    if len(X) < 20:
        return {"error": "Insufficient data", "predictions": []}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    probabilities = model.predict_proba(X_train_scaled)[:, 1]
    
    return {
        "metric": "Virality Probability",
        "model_type": "GradientBoosting",
        "data_points": len(df),
        "avg_probability": float(np.mean(probabilities)),
        "probability_distribution": {
            "min": float(np.min(probabilities)),
            "max": float(np.max(probabilities)),
            "mean": float(np.mean(probabilities)),
            "std": float(np.std(probabilities))
        }
    }

def predict_engagement(df):
    """Predict engagement rate"""
    le_post = LabelEncoder()
    df_model = df.copy()
    df_model['post_type_encoded'] = le_post.fit_transform(df_model['post_type'])
    
    feature_cols = ['duration_sec', 'publish_month', 'day_of_week', 'is_weekend', 
                   'is_prime_time', 'has_sound', 'post_type_encoded']
    
    X = df_model[feature_cols].fillna(0)
    y = df_model['engagement_rate']
    
    if len(X) < 20:
        return {"error": "Insufficient data", "predictions": []}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return {
        "metric": "Engagement Rate (%)",
        "model_type": "RandomForest",
        "data_points": len(df),
        "model_performance": {
            "r2_score": float(r2),
            "mae": float(mae),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred)))
        },
        "predictions": {
            "avg_engagement": float(np.mean(y_test)),
            "predicted_avg": float(np.mean(y_pred)),
            "distribution": {
                "min": float(np.min(y_pred)),
                "max": float(np.max(y_pred)),
                "mean": float(np.mean(y_pred)),
                "std": float(np.std(y_pred))
            }
        }
    }

def main():
    import os
    
    try:
        conn = get_db_connection()
        df = fetch_tiktok_data(conn)
        conn.close()
        
        if len(df) == 0:
            print(json.dumps({"error": "No data found"}))
            return
        
        df = engineer_features(df)
        
        virality_result = predict_virality(df)
        engagement_result = predict_engagement(df)
        
        result = {
            "platform": "TikTok",
            "timestamp": datetime.now().isoformat(),
            "data_points": len(df),
            "predictions": {
                "virality": virality_result,
                "engagement": engagement_result
            },
            "summary": {
                "avg_views": float(df['views'].mean()),
                "avg_engagement_rate": float(df['engagement_rate'].mean()),
                "total_videos": len(df)
            }
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
