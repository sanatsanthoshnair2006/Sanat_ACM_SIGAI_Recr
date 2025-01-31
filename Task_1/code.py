import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data and handle missing values
data = pd.read_csv("C:\\Users\\Sanat Nair\\Downloads\\Housing.csv")
df2 = data.bfill()

# Extract features and target variable
X = df2['area'].values.reshape(-1, 1)  # Feature: Area
y = df2['price'].values.reshape(-1, 1)  # Target: Price

# Normalize the data for better gradient descent performance
X_mean, X_std = np.mean(X), np.std(X)
y_mean, y_std = np.mean(y), np.std(y)
X = (X - X_mean) / X_std
y = (y - y_mean) / y_std

# Add a bias term to feature matrix
X_bias = np.hstack((np.ones((X.shape[0], 1)), X))

# Initialize weights
weights = np.zeros((2, 1))  # [bias, slope]

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Gradient Descent for Linear Regression
for epoch in range(epochs):
    y_pred = np.dot(X_bias, weights)
    error = y_pred - y
    gradient = (1 / X.shape[0]) * np.dot(X_bias.T, error)
    weights -= learning_rate * gradient

# Final weights
bias, slope = weights.flatten()
print(f"Trained Linear Regression Weights -> Bias: {bias:.4f}, Slope: {slope:.4f}")

# Plotting Linear Regression Fit
plt.scatter(X * X_std + X_mean, y * y_std + y_mean, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X * X_std + X_mean, (y_pred * y_std + y_mean), color='red', label='Linear Regression Fit')
plt.title('Linear Regression: Area vs Price')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation Metrics
mse = np.mean((y_pred - y) ** 2)
rmse = np.sqrt(mse)
print(f"Linear Regression -> MSE: {mse:.4f}, RMSE: {rmse:.4f}")

# Polynomial Regression (Degree 2)
X_poly = np.hstack((X_bias, (X ** 2)))

# Initialize weights for Polynomial Regression
weights_poly = np.zeros((3, 1))

# Gradient Descent for Polynomial Regression
for epoch in range(epochs):
    y_pred_poly = np.dot(X_poly, weights_poly)
    error_poly = y_pred_poly - y
    gradient_poly = (1 / X.shape[0]) * np.dot(X_poly.T, error_poly)
    weights_poly -= learning_rate * gradient_poly

# Plotting Polynomial Regression Fit
plt.scatter(X * X_std + X_mean, y * y_std + y_mean, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X * X_std + X_mean, (y_pred_poly * y_std + y_mean), color='green', label='Polynomial Regression Fit')
plt.title('Comparison of Linear and Polynomial Regression')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation Metrics for Polynomial Regression
mse_poly = np.mean((y_pred_poly - y) ** 2)
rmse_poly = np.sqrt(mse_poly)
print(f"Polynomial Regression -> MSE: {mse_poly:.4f}, RMSE: {rmse_poly:.4f}")
