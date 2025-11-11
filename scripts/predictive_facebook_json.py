"""
Facebook Predictive Analytics - Returns JSON only
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

def fetch_facebook_data(conn):
    """Fetch Facebook data for predictions"""
    query = """
    SELECT 
        post_id, reach, reactions, comments, shares,
        publish_time, type
    FROM public.facebook_data_set
    WHERE reach IS NOT NULL AND reach > 0
        AND publish_time IS NOT NULL
    ORDER BY publish_time DESC
    LIMIT 500
    """
    return pd.read_sql_query(query, conn)

def engineer_features(df):
    """Create features for prediction"""
    df = df.copy()
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    df['day_of_week'] = df['publish_time'].dt.dayofweek
    df['hour_of_day'] = df['publish_time'].dt.hour
    df['engagement'] = (df.get('reactions', 0) + df.get('comments', 0) + df.get('shares', 0)).fillna(0)
    df['engagement_rate'] = (df['engagement'] / df['reach'] * 100).fillna(0)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_prime_time'] = df['hour_of_day'].between(17, 22).astype(int)
    df['type'] = df['type'].fillna('unknown')
    return df

def predict_reach_potential(df):
    """Predict reach potential"""
    feature_cols = ['day_of_week', 'hour_of_day', 'is_weekend', 'is_prime_time']
    
    X = df[feature_cols].fillna(0)
    y = df['reach']
    
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
        "metric": "Reach Potential",
        "model_type": "RandomForest",
        "performance": {
            "r2": float(r2),
            "mae": float(mae),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred)))
        },
        "predictions": {
            "avg_reach": float(np.mean(y_test)),
            "predicted_avg": float(np.mean(y_pred))
        }
    }

def predict_engagement(df):
    """Predict engagement rate"""
    feature_cols = ['day_of_week', 'hour_of_day', 'is_weekend', 'is_prime_time']
    
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
            "probability_high_engagement": float(np.mean(proba)),
            "distribution": {
                "min": float(np.min(proba)),
                "max": float(np.max(proba)),
                "mean": float(np.mean(proba))
            }
        }
    }

def main():
    try:
        conn = get_db_connection()
        df = fetch_facebook_data(conn)
        conn.close()
        
        if len(df) == 0:
            print(json.dumps({"error": "No data found"}))
            return
        
        df = engineer_features(df)
        
        reach_result = predict_reach_potential(df)
        engagement_result = predict_engagement(df)
        
        result = {
            "platform": "Facebook",
            "timestamp": datetime.now().isoformat(),
            "data_points": len(df),
            "predictions": {
                "reach": reach_result,
                "engagement": engagement_result
            },
            "summary": {
                "avg_reach": float(df['reach'].mean()),
                "avg_engagement_rate": float(df['engagement_rate'].mean()),
                "total_posts": len(df)
            }
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
