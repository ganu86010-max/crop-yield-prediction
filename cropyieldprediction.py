# Crop Yield Prediction using Polynomial Regression

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 2. Load Dataset
df = pd.read_csv("crop_yield_dataset.csv")

print("First 5 rows:")
print(df.head())


# 3. Check Missing Values
print("\nMissing values:")
print(df.isnull().sum())


# 4. Feature Selection
X = df[['Rainfall', 'Temperature', 'Fertilizer', 'Soil_Quality']]
y = df['Crop_Yield']


# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 6. Apply Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# 7. Train Model
model = LinearRegression()
model.fit(X_train_poly, y_train)


# 8. Prediction
y_pred = model.predict(X_test_poly)

print("\nPredictions:")
print(y_pred[:5])


# 9. Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)


# 10. Visualization (Actual vs Predicted)
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yield")
plt.show()


# 11. Example Prediction
sample = [[900, 25, 120, 7]]  # Rainfall, Temp, Fertilizer, Soil
sample_poly = poly.transform(sample)

prediction = model.predict(sample_poly)

print("\nSample Prediction:")
print("Predicted Crop Yield:", prediction[0])