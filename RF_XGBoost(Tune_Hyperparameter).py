import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load each dataset
df_temp = pd.read_csv('temperature.csv')
df_humidity = pd.read_csv('humidity.csv')
df_pressure = pd.read_csv('pressure.csv')
df_wind_speed = pd.read_csv('wind_speed.csv')
df_rainfall = pd.read_csv('rainfall.csv')  # This file is similar to the one attached

# Merge the datasets on year, month, and day
df = pd.merge(df_temp, df_humidity, on=['Year', 'Month', 'Day'])
df = pd.merge(df, df_pressure, on=['Year', 'Month', 'Day'])
df = pd.merge(df, df_wind_speed, on=['Year', 'Month', 'Day'])
df = pd.merge(df, df_rainfall, on=['Year', 'Month', 'Day'])

# Convert columns to numeric data types
df['Temperature'] = pd.to_numeric(df['Temperature'])
df['Pressure'] = pd.to_numeric(df['Pressure'])
df['Wind_Speed'] = pd.to_numeric(df['Wind_Speed'])
df['Rainfall'] = pd.to_numeric(df['Rainfall'])

# Use IterativeImputer to fill missing values
imputer = IterativeImputer(random_state=42)
df[['Temperature', 'Humidity', 'Pressure', 'Wind_Speed', 'Rainfall']] = \
    (imputer.fit_transform(df[['Temperature', 'Humidity', 'Pressure', 'Wind_Speed', 'Rainfall']]))

# Define features and target variable
X = df[['Temperature', 'Humidity', 'Pressure', 'Wind_Speed']]
y = df['Rainfall']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [500, 750, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [1, 3, 5],
    'gamma': [0.1, 0.5, 1]
}

# Use GridSearchCV to tune the hyperparameters
xgb = XGBRegressor(random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Train the final model with the best hyperparameters
final_model = XGBRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# Predict on the test set
y_pred = final_model.predict(X_test)

# Calculate RMSE
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Example: Predicting with new data
temp = input('Enter the temperature in Celsius: ')
humidity = input('Enter the humidity in % :')
pressure = input('Enter the atmospheric pressure in hPa: ')
wind_speed = input('Enter the wind speed in km/hr: ')

new_data = pd.DataFrame({
    'Temperature': [temp],
    'Humidity': [humidity],
    'Pressure': [pressure],
    'Wind_Speed': [wind_speed]
})
new_rainfall_prediction = final_model.predict(new_data)
print(f'Predicted Rainfall: {new_rainfall_prediction[0]} mm')
