"""
Spotify Predictive Analytics - Returns JSON only
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

def fetch_spotify_data(conn):
    """Fetch Spotify data for predictions"""
    query = """
    SELECT 
        track_id, track_name, streams, listeners, followers,
        release_date, explicit
    FROM public.spotify_data
    WHERE streams IS NOT NULL AND streams > 0
        AND listeners IS NOT NULL
        AND release_date IS NOT NULL
    ORDER BY release_date DESC
    LIMIT 500
    """
    return pd.read_sql_query(query, conn)

def engineer_features(df):
    """Create features for prediction"""
    df = df.copy()
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['month'] = df['release_date'].dt.month
    df['quarter'] = df['release_date'].dt.quarter
    df['year'] = df['release_date'].dt.year
    df['streams_per_listener'] = (df['streams'] / df['listeners']).fillna(0)
    df['listener_engagement'] = (df['listeners'] / df.get('followers', 1)).fillna(0)
    df['explicit'] = df.get('explicit', False).astype(int)
    return df

def predict_streams(df):
    """Predict streams potential"""
    feature_cols = ['month', 'quarter', 'explicit', 'listeners']
    
    X = df[feature_cols].fillna(0)
    y = df['streams']
    
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
        "metric": "Streams Prediction",
        "model_type": "RandomForest",
        "performance": {
            "r2": float(r2),
            "mae": float(mae),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred)))
        },
        "predictions": {
            "avg_streams": float(np.mean(y_test)),
            "predicted_avg": float(np.mean(y_pred))
        }
    }

def predict_popularity(df):
    """Predict popularity potential"""
    feature_cols = ['month', 'quarter', 'explicit']
    
    X = df[feature_cols].fillna(0)
    y = (df['listeners'] > df['listeners'].median()).astype(int)
    
    if len(X) < 20 or y.sum() < 5:
        return {"error": "Insufficient data"}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    proba = model.predict_proba(X_train_scaled)[:, 1]
    
    return {
        "metric": "High Popularity Probability",
        "model_type": "GradientBoosting",
        "predictions": {
            "probability_high_listeners": float(np.mean(proba)),
            "avg_listeners": float(df['listeners'].mean())
        }
    }

def main():
    try:
        conn = get_db_connection()
        df = fetch_spotify_data(conn)
        conn.close()
        
        if len(df) == 0:
            print(json.dumps({"error": "No data found"}))
            return
        
        df = engineer_features(df)
        
        streams_result = predict_streams(df)
        popularity_result = predict_popularity(df)
        
        result = {
            "platform": "Spotify",
            "timestamp": datetime.now().isoformat(),
            "data_points": len(df),
            "predictions": {
                "streams": streams_result,
                "popularity": popularity_result
            },
            "summary": {
                "avg_streams": float(df['streams'].mean()),
                "avg_listeners": float(df['listeners'].mean()),
                "total_tracks": len(df)
            }
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
