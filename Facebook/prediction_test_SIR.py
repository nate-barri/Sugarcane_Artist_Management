

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Load data from CSV file
df = pd.read_csv('Jan-01-2025_Mar-20-2025_1340916307234368.csv'),
df = pd.read_csv('Oct-01-2024_Dec-31-2024_659394146642923.csv')
# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
df[['Views', 'Reactions', 'Comments', 'Shares', 'Total clicks', 'Other Clicks']] = imputer.fit_transform(df[['Views', 'Reactions', 'Comments', 'Shares', 'Total clicks', 'Other Clicks']])

# Convert Publish time to datetime and extract date ordinal
df['Publish time'] = pd.to_datetime(df['Publish time'])
df['date_ordinal'] = df['Publish time'].apply(lambda x: x.toordinal())

# Normalize date ordinals
scaler = StandardScaler()
df['date_ordinal_scaled'] = scaler.fit_transform(df[['date_ordinal']])

# Prepare training data with additional features
features = ['date_ordinal_scaled', 'Views', 'Reactions', 'Comments', 'Shares', 'Total clicks', 'Other Clicks']
X = df[features]
y = df['Reach']

# Polynomial Regression (degree 3)
poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X, y)

# Generate future dates for prediction
future_dates = pd.date_range(start=df['Publish time'].max(), periods=180, freq='D')
future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
future_ordinals_scaled = scaler.transform(future_ordinals)

# Create future data with additional features (assuming constant values for simplicity)
future_data = pd.DataFrame({
    'date_ordinal_scaled': future_ordinals_scaled.flatten(),
    'Views': [df['Views'].mean()] * len(future_dates),
    'Reactions': [df['Reactions'].mean()] * len(future_dates),
    'Comments': [df['Comments'].mean()] * len(future_dates),
    'Shares': [df['Shares'].mean()] * len(future_dates),
    'Total clicks': [df['Total clicks'].mean()] * len(future_dates),
    'Other Clicks': [df['Other Clicks'].mean()] * len(future_dates)
})

# Ensure future_data has the same feature names
future_data.columns = features

# Make future predictions using different models
future_X_poly = poly.transform(future_data)
poly_predictions = poly_model.predict(future_X_poly)
rf_predictions = rf_model.predict(future_data)
gb_predictions = gb_model.predict(future_data)

# Ensure predictions start from 0.0
poly_predictions[poly_predictions < 0] = 0
rf_predictions[rf_predictions < 0] = 0
gb_predictions[gb_predictions < 0] = 0

# Evaluate model fit
poly_r2 = r2_score(y, poly_model.predict(X_poly))
rf_r2 = r2_score(y, rf_model.predict(X))
gb_r2 = r2_score(y, gb_model.predict(X))

print(f"Polynomial Regression R² Score: {poly_r2}")
print(f"Random Forest R² Score: {rf_r2}")
print(f"Gradient Boosting R² Score: {gb_r2}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df['Publish time'], df['Reach'], label='Actual Reach', color='blue')
plt.plot(df['Publish time'], poly_model.predict(X_poly), linestyle='dotted', color='green', label='Predicted Reach (Historical) - Poly')
plt.plot(future_dates, poly_predictions, linestyle='dashed', color='red', label='Future Predictions (Next 6 Months) - Poly')
plt.plot(future_dates, rf_predictions, linestyle='dashed', color='orange', label='Future Predictions (Next 6 Months) - RF')
plt.plot(future_dates, gb_predictions, linestyle='dashed', color='purple', label='Future Predictions (Next 6 Months) - GB')
plt.xlabel('Date')
plt.ylabel('Reach')
plt.title('Reach Prediction for the Next 6 Months (Using Different Models)')
plt.legend()
plt.show()