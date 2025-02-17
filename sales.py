import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Define the correct file path
file_path = r"C:\Users\ukar1\OneDrive\Desktop\vscodeforweb\vscodeforweb\datascience\advertising.csv"
import os
print(os.path.exists(file_path))  # Should print True

# Check if the file exists before proceeding
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found. Check the path.")
    exit()

# Read the dataset
data = pd.read_csv(r"C:\Users\ukar1\OneDrive\Desktop\vscodeforweb\vscodeforweb\datascience\advertising.csv")


# Basic data checks
print("Dataset Preview:\n", data.head())
print("Dataset Shape:", data.shape)
print("Dataset Info:")
print(data.info())
print("Dataset Description:\n", data.describe())

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Outlier Analysis
fig, axs = plt.subplots(3, figsize=(6, 5))
sns.boxplot(x=data['TV'], ax=axs[0])
sns.boxplot(x=data['Newspaper'], ax=axs[1])
sns.boxplot(x=data['Radio'], ax=axs[2])
plt.tight_layout()
plt.show()

# Scatter plot to visualize relationships
sns.pairplot(data, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()

# Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
plt.title("Feature Correlation")
plt.show()

# Define independent and dependent variables
X = data[['TV']]
y = data['Sales']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

# Add constant to get intercept
X_train_sm = sm.add_constant(X_train)

# Fit regression model
lr = sm.OLS(y_train, X_train_sm).fit()
print(lr.summary())

# Plot regression line
plt.scatter(X_train, y_train, label="Actual Data")
plt.plot(X_train, lr.predict(X_train_sm), 'r', label="Regression Line")
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.legend()
plt.title("Regression Line")
plt.show()

# Residual analysis
residuals = y_train - lr.predict(X_train_sm)
sns.histplot(residuals, bins=15, kde=True)
plt.title('Error Terms Distribution')
plt.xlabel('Residuals')
plt.show()

plt.scatter(X_train, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("TV Advertising Budget")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Predict on test set
X_test_sm = sm.add_constant(X_test)
y_pred = lr.predict(X_test_sm)

# Model evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r_squared = r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R-squared:", r_squared)

# Plot test predictions
plt.scatter(X_test, y_test, label="Actual Data")
plt.plot(X_test, y_pred, 'r', label="Regression Line")
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.legend()
plt.title("Test Predictions")
plt.show()
