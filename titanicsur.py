# 🔥 Import necessary libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ✅ Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ✅ Load dataset with the correct file path
file_path = r"C:\Users\ukar1\OneDrive\Desktop\vscodeforweb\vscodeforweb\datascience\Titanic-Dataset.csv"

# ✅ Check if file exists before proceeding
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found. Check the path.")
    exit()

# ✅ Read the dataset
data = pd.read_csv(file_path)

# ✅ Display first 5 rows
print(data.head())

# ✅ Basic dataset information
print(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.\n")
print("Column Names:", data.columns.tolist())

# ✅ Summary statistics
print(data.describe())
print(data.describe(include='object'))
print(data.info())

# ✅ Check for duplicate values
print(f"Total Duplicates: {data.duplicated().sum()}")

# ✅ Convert categorical 'Sex' column to numeric
df = pd.get_dummies(data['Sex'], drop_first=True).astype('int64')
data = pd.concat([df, data], axis=1)
data.drop('Sex', axis=1, inplace=True)
data.rename(columns={'male': 'gender'}, inplace=True)

# ✅ Drop unnecessary columns
data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# ✅ Save cleaned data
data.to_csv('cleaned_data.csv', index=False)

# ✅ Check for missing values
sns.heatmap(data.isnull(), cmap="viridis")
plt.title("Missing Data Heatmap")
plt.show()

# ✅ Fill missing 'Age' values with median
data['Age'].fillna(data['Age'].median(), inplace=True)

# ✅ Drop 'Cabin' column due to too many missing values
data.drop(columns='Cabin', inplace=True)

# ✅ Fill missing 'Embarked' values with mode
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# ✅ Drop 'Embarked' after filling (optional)
data.drop(columns=['Embarked'], inplace=True)

# ✅ Save cleaned dataset
data.to_csv('cleaned_data.csv', index=False)

# ✅ Data Imputation using Different Methods

# 👉 Simple Imputer (Mean)
imputer = SimpleImputer(strategy='mean')
data_SimpleImputer = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 👉 Iterative Imputer
iterative_imputer = IterativeImputer()
data_IterativeImputer = pd.DataFrame(iterative_imputer.fit_transform(data), columns=data.columns)

# 👉 KNN Imputer
knn_imputer = KNNImputer(n_neighbors=2)
data_KNNImputer = pd.DataFrame(knn_imputer.fit_transform(data), columns=data.columns)

# 👉 Linear Interpolation
data_interpolate = data.interpolate(method='linear')

# ✅ Splitting Data for Training
def split_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    print(X.shape, y.shape, end='\n')
    return X, y  

X_data, y_data = split_data(data, 'Survived')
X_data_simple, y_data_simple = split_data(data_SimpleImputer, 'Survived')
X_data_iterative, y_data_iterative = split_data(data_IterativeImputer, 'Survived')
X_data_knn, y_data_knn = split_data(data_KNNImputer, 'Survived')
X_data_interpolate, y_data_interpolate = split_data(data_interpolate, 'Survived')

# ✅ Function to Split into Train-Test
def splitting_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, shuffle=True)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test  

# Splitting different datasets
X_train, X_test, y_train, y_test = splitting_train_test(X_data, y_data)  
X_train_simple, X_test_simple, y_train_simple, y_test_simple = splitting_train_test(X_data_simple, y_data_simple) 
X_train_iterative, X_test_iterative, y_train_iterative, y_test_iterative = splitting_train_test(X_data_iterative, y_data_iterative) 
X_train_knn, X_test_knn, y_train_knn, y_test_knn = splitting_train_test(X_data_knn, y_data_knn)
X_train_interpolate, X_test_interpolate, y_train_interpolate, y_test_interpolate = splitting_train_test(X_data_interpolate, y_data_interpolate)  

# ✅ Train Logistic Regression Model
log_reg = LogisticRegression()

# ✅ Training & Evaluating Model
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred) * 100
    accuracy_train = model.score(X_train, y_train) * 100
    print(f"Training Accuracy: {accuracy_train:.2f}%")
    print(f"Testing Accuracy: {accuracy_test:.2f}%\n")

# Running models on different imputed datasets
print("\n➡️ Logistic Regression Results:")
train_and_evaluate(log_reg, X_train, X_test, y_train, y_test)
train_and_evaluate(log_reg, X_train_simple, X_test_simple, y_train_simple, y_test_simple)
train_and_evaluate(log_reg, X_train_iterative, X_test_iterative, y_train_iterative, y_test_iterative)
train_and_evaluate(log_reg, X_train_knn, X_test_knn, y_train_knn, y_test_knn)
train_and_evaluate(log_reg, X_train_interpolate, X_test_interpolate, y_train_interpolate, y_test_interpolate)
