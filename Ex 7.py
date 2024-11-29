import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = pd.read_csv(r"C:\Users\roani\OneDrive\Desktop\New folder\student_lifestyle_dataset.csv")


for col in ['Extracurricular_Hours_Per_Day', 'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day']:
    data[col] = pd.to_numeric(data[col], errors='coerce')


data = data.dropna()


X = data[['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 
          'Extracurricular_Hours_Per_Day', 'Social_Hours_Per_Day', 
          'Physical_Activity_Hours_Per_Day']]
y = data['GPA']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, y_pred_lr)


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.7, color='blue', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Ideal Fit')
plt.title("Linear Regression: Predicted GPA vs Actual GPA")
plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")
plt.legend()
plt.grid(True)
plt.show()

# 2. k-Nearest Neighbors with Optimal k Search
errors = []
k_values = range(1, 21)

for k in k_values:
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    errors.append(mean_squared_error(y_test, y_pred_knn))

optimal_k = k_values[np.argmin(errors)]
knn_model_optimal = KNeighborsRegressor(n_neighbors=optimal_k)
knn_model_optimal.fit(X_train, y_train)
y_pred_knn_optimal = knn_model_optimal.predict(X_test)
knn_mse = mean_squared_error(y_test, y_pred_knn_optimal)

# 3. Scatter Plot of Study Hours vs GPA
plt.figure(figsize=(10, 6))
plt.scatter(data['Study_Hours_Per_Day'], data['GPA'], alpha=0.7, color='blue')
plt.title("Scatter Plot: Study Hours vs GPA")
plt.xlabel("Study Hours Per Day")
plt.ylabel("GPA")
plt.grid(True)
plt.show()

# 4. Plotting Optimal k Search Results
plt.figure(figsize=(10, 6))
plt.plot(k_values, errors, marker='o', color='red')
plt.title("Optimal k Search")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.axvline(x=optimal_k, color='green', linestyle='--', label=f'Optimal k = {optimal_k}')
plt.legend()
plt.show()

# Results Summary
print(f"Linear Regression MSE: {lr_mse}")
print(f"k-Nearest Neighbors (Optimal k = {optimal_k}) MSE: {knn_mse}")

