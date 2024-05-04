import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=5, min_samples_leaf=1, min_samples_split=10)
model.fit(X_train, y_train)

# Use Recursive Feature Elimination (RFE) to rank the feature importance
rfe = RFE(model, n_features_to_select=4)
rfe = rfe.fit(X, y)

# Print the ranking of the features
print("Feature ranking:")
for i, rank in enumerate(rfe.ranking_):
    print(f"{X.columns[i]}: Rank {rank}")

# The output should show that temperature, humidity, and air_pressure have the lowest ranks (most important),
# while wind_speed has a higher rank (still important, but less so than the other 3 variables).
