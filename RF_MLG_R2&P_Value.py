import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

# Initialize and train the Multivariate Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the coefficients of the linear regression model
coef = model.coef_
intercept = model.intercept_

# Print the coefficients
print("Regression Equation:")
print(f"Rainfall = {intercept:.2f} + {coef[0]:.2f} * Temperature + {coef[1]:.2f} * Humidity + {coef[2]:.2f} * Pressure + {coef[3]:.2f} * Wind_Speed")
# Get the R-squared value
r_squared = model.score(X_test, y_test)
print(f"R-squared: {r_squared:.2f}")

# Get the coefficient p-values
import statsmodels.api as sm
X_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_sm).fit()
print(model_sm.summary())
