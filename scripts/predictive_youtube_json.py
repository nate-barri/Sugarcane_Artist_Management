"""
YouTube Predictive Analytics - Returns JSON only
"""
import sys
import json
import os
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

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

def fetch_youtube_data(conn):
    """Fetch YouTube data for predictions"""
    query = """
    SELECT 
        video_id, title, views, likes, comments, shares,
        watch_time_minutes, average_view_duration_percent, 
        publish_date, category
    FROM public.yt_video_etl
    WHERE views IS NOT NULL AND views > 0
        AND watch_time_minutes IS NOT NULL
        AND publish_date IS NOT NULL
    ORDER BY publish_date DESC
    LIMIT 500
    """
    return pd.read_sql_query(query, conn)

def engineer_features(df):
    """Create features for prediction"""
    df = df.copy()
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df['day_of_week'] = df['publish_date'].dt.dayofweek
    df['month'] = df['publish_date'].dt.month
    df['engagement_rate'] = ((df.get('likes', 0) + df.get('comments', 0)) / df['views'] * 100).fillna(0)
    df['watch_rate'] = (df.get('watch_time_minutes', 0) / df['views']).fillna(0)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['category'] = df['category'].fillna('unknown')
    return df

def predict_views(df):
    """Predict views performance"""
    feature_cols = ['day_of_week', 'month', 'is_weekend', 'average_view_duration_percent']
    
    X = df[feature_cols].fillna(0)
    y = df['views']
    
    if len(X) < 20:
        return {"error": "Insufficient data"}
    
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
        "metric": "Views Prediction",
        "model_type": "RandomForest",
        "performance": {
            "r2": float(r2),
            "mae": float(mae),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred)))
        },
        "predictions": {
            "avg_views": float(np.mean(y_test)),
            "predicted_avg": float(np.mean(y_pred))
        }
    }

def predict_engagement(df):
    """Predict engagement potential"""
    feature_cols = ['day_of_week', 'month', 'average_view_duration_percent']
    
    X = df[feature_cols].fillna(0)
    y = (df['engagement_rate'] > df['engagement_rate'].median()).astype(int)
    
    if len(X) < 20 or y.sum() < 5:
        return {"error": "Insufficient data"}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    proba = model.predict_proba(X_train_scaled)[:, 1]
    
    return {
        "metric": "High Engagement Probability",
        "model_type": "GradientBoosting",
        "predictions": {
            "probability": float(np.mean(proba)),
            "avg_engagement_rate": float(df['engagement_rate'].mean())
        }
    }

def main():
    try:
        conn = get_db_connection()
        df = fetch_youtube_data(conn)
        conn.close()
        
        if len(df) == 0:
            print(json.dumps({"error": "No data found"}))
            return
        
        df = engineer_features(df)
        
        views_result = predict_views(df)
        engagement_result = predict_engagement(df)
        
        result = {
            "platform": "YouTube",
            "timestamp": datetime.now().isoformat(),
            "data_points": len(df),
            "predictions": {
                "views": views_result,
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
