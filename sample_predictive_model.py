import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Database connection parameters
db_params = {
    'dbname': 'test1',
    'user': 'postgres',
    'password': 'admin',
    'host': 'localhost',
    'port': '5432'
}

# Fetch data from PostgreSQL
def fetch_data():
    conn = psycopg2.connect(**db_params)
    query = """
        SELECT publish_time, reach 
        FROM facebook_data_set 
        WHERE reach IS NOT NULL 
        ORDER BY publish_time;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    df['date_ordinal'] = df['publish_time'].apply(lambda x: x.toordinal())
    return df

# Load and preprocess data
df = fetch_data()
df['log_reach'] = np.log1p(df['reach'])  # Apply log transformation to stabilize variance

# Prepare training data
X = df[['date_ordinal']]
y = df['log_reach']

# Use Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=10)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# Generate future dates for prediction
future_dates = pd.date_range(start=df['publish_time'].max(), periods=180, freq='D')
future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)

# Make future predictions
future_X_poly = poly.transform(future_ordinals)
future_log_predictions = model.predict(future_X_poly)

# Convert back from log scale
future_predictions = np.expm1(future_log_predictions)

# Evaluate model fit
r2 = r2_score(y, model.predict(X_poly))
print(f"R² Score: {r2}")  # Close to 0 means time isn't a strong predictor

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df['publish_time'], df['reach'], label='Actual Reach', color='blue')
plt.plot(df['publish_time'], np.expm1(model.predict(X_poly)), linestyle='dotted', color='green', label='Predicted Reach (Historical)')
plt.plot(future_dates, future_predictions, linestyle='dashed', color='red', label='Future Predictions (Next 6 Months)')
plt.xlabel('Date')
plt.ylabel('Reach')
plt.title('Reach Prediction for the Next 6 Months (Using Log & Polynomial Regression)')
plt.legend()
plt.show()