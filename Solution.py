import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r'C:\Users\sathi\OneDrive\Desktop\WORKSPACE\Prodigy\TASK-1\house-prices-advanced-regression-techniques\train.csv')

# Select relevant features (square footage, number of bedrooms, and number of bathrooms)
df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

# Drop rows with missing values
df.dropna(inplace=True)

# Features (independent variables)
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# Target variable (dependent variable)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared (R²) score
r2 = r2_score(y_test, y_pred)

# Display results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Plot Actual vs Predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs Predicted SalePrice')
plt.show()

# Display coefficients
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")