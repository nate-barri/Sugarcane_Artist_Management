# fb_reach_prediction_fixed.py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import psycopg2
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------
# NEON DB CONNECTION
# -------------------------
db_params = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host': 'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port': '5432',
    'sslmode': 'require'
}

# -------------------------
# LOAD DATA
# -------------------------
print("Loading data from Neon DB...")
conn = psycopg2.connect(**db_params)

query = """
SELECT
    publish_time,
    reach,
    reactions,
    comments,
    shares,
    post_type,
    seconds_viewed,
    average_seconds_viewed,
    duration_sec
FROM public.facebook_data_set
WHERE reach IS NOT NULL
ORDER BY publish_time;
"""

df = pd.read_sql(query, conn)
conn.close()
print(f"Raw rows loaded: {len(df)}")

# -------------------------
# FEATURE ENGINEERING
# -------------------------
print("Engineering features...")

df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
df['hour'] = df['publish_time'].dt.hour.fillna(0).astype(int)
df['dayofweek'] = df['publish_time'].dt.dayofweek.fillna(0).astype(int)
df['month'] = df['publish_time'].dt.month.fillna(0).astype(int)
df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)

numeric_features = [
    "reactions", "comments", "shares", "seconds_viewed",
    "average_seconds_viewed", "duration_sec", "hour",
    "dayofweek", "month", "is_weekend"
]
for c in numeric_features:
    if c not in df.columns:
        df[c] = 0.0

cat_features = ['post_type']
if 'post_type' not in df.columns:
    df['post_type'] = 'unknown'

df = df.dropna(subset=['reach']).copy()
print(f"After dropping missing reach: {len(df)} rows")

for c in numeric_features:
    df[c] = pd.to_numeric(df[c], errors='coerce')

X = df[cat_features + numeric_features].copy()
y = df['reach'].astype(float).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True
)
print(f"Data points: total={len(df)}, train={len(X_train)}, test={len(X_test)}")

# -------------------------
# PREPROCESSING
# -------------------------
if sklearn.__version__ >= "1.2":
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
else:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', ohe)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

def get_feature_names_from_preprocessor(preprocessor, numeric_features, cat_features):
    try:
        cat_pipeline = preprocessor.named_transformers_['cat']
        if hasattr(cat_pipeline, 'named_steps') and 'ohe' in cat_pipeline.named_steps:
            ohe_step = cat_pipeline.named_steps['ohe']
            cat_feature_names = list(ohe_step.get_feature_names_out(cat_features))
        else:
            cat_feature_names = cat_features
        return numeric_features + cat_feature_names
    except Exception:
        return numeric_features + cat_features

# -------------------------
# MODEL TRAIN/EVAL
# -------------------------
def train_and_eval_model(model, model_name, show_importance=True):
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    medae = median_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))   # ✅ fixed
    r2 = r2_score(y_test, preds)

    baseline = y_test.mean()
    print(f"\n{model_name} Performance:")
    print(f"  MAE:   {mae:,.2f}  ({mae/baseline:.2%} of avg reach)")
    print(f"  MedAE: {medae:,.2f}  ({medae/baseline:.2%} of avg reach)")
    print(f"  RMSE:  {rmse:,.2f}  ({rmse/baseline:.2%} of avg reach)")
    print(f"  R²:    {r2:.4f}")

    if show_importance:
        feature_names = get_feature_names_from_preprocessor(
            pipe.named_steps['preprocessor'], numeric_features, cat_features
        )

        if hasattr(pipe.named_steps['model'], 'coef_'):
            coefs = pipe.named_steps['model'].coef_
            abs_coefs = np.abs(coefs)
            pct = 100.0 * abs_coefs / abs_coefs.sum()
            df_coef = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefs,
                '% Contribution': pct
            }).sort_values('% Contribution', ascending=False)
            print(f"\n🔎 {model_name} Coefficients (% contribution):")
            print(df_coef.to_string(index=False, float_format='%.2f'))

        elif hasattr(pipe.named_steps['model'], 'feature_importances_'):
            imps = pipe.named_steps['model'].feature_importances_
            pct = 100.0 * imps / imps.sum()
            df_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': imps,
                '% Contribution': pct
            }).sort_values('% Contribution', ascending=False)
            print(f"\n🌲 {model_name} Feature Importances (% contribution):")
            print(df_imp.to_string(index=False, float_format='%.2f'))

    # 📊 Plot predictions vs actual
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, preds, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Reach")
    plt.ylabel("Predicted Reach")
    plt.title(f"{model_name}: Predictions vs Actual")
    plt.show()

    return pipe, preds

# -------------------------
# RUN MODELS
# -------------------------
print("\nTraining Linear Regression...")
pipe_lr, preds_lr = train_and_eval_model(LinearRegression(), "Linear Regression")

print("\nTraining Random Forest...")
pipe_rf, preds_rf = train_and_eval_model(RandomForestRegressor(n_estimators=200, random_state=42), "Random Forest")

print("\nTraining Gradient Boosting...")
pipe_gb, preds_gb = train_and_eval_model(GradientBoostingRegressor(n_estimators=200, random_state=42), "Gradient Boosting")

print("\n✅ Done. Models trained, contributions printed, and graphs plotted.")
