# How to Run the Code

Install dependencies if not already installed:

pip install numpy pandas matplotlib
Ensure the dataset Housing.csv is located at C:\Users\Sanat Nair\Downloads\Housing.csv.
Run the Python script in a terminal or Jupyter Notebook.

# What the Code Does

Loads Data: Reads the CSV file and fills missing values using backward fill (bfill).
Prepares Features: Extracts the area column as input (X) and price column as output (y).
Normalizes Data: Standardizes X and y to improve gradient descent performance.
Linear Regression:
Adds a bias term and initializes weights.
Uses gradient descent for 1000 iterations to minimize error.
Plots the best-fit line and prints the trained weights.
Evaluates Linear Regression: Computes MSE (Mean Squared Error) and RMSE (Root Mean Squared Error).
Polynomial Regression (Degree 2):
Extends features by adding a squared term (X²).
Trains a polynomial regression model with gradient descent.
Plots the polynomial curve alongside linear regression.
Evaluates Polynomial Regression: Computes MSE and RMSE for comparison.

# Expected Results

The first plot shows Linear Regression, with actual data points (blue) and the predicted line (red).
The second plot compares Linear vs Polynomial Regression, where the polynomial curve (green) may fit better.
The printed weights (Bias and Slope) represent the trained model parameters.
Lower MSE/RMSE indicates better model accuracy, and polynomial regression should have a lower error if the data has a nonlinear pattern.
