# Task 4 – Sales Prediction using Linear Regression | CODSOFT Internship

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: Load Dataset
df = pd.read_csv("advertising.csv")
print("✅ Data Loaded")
print(df.head())

# STEP 2: Explore and Visualize (Optional)
sns.pairplot(df)
plt.suptitle("Pairplot of Features vs Sales", y=1.02)
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# STEP 3: Define Features and Target
X = df.drop("Sales", axis=1)
y = df["Sales"]

# STEP 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 5: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 6: Predict
y_pred = model.predict(X_test)

# STEP 7: Evaluation
print("\n🎯 Model Evaluation:")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# STEP 8: Actual vs Predicted Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid()
plt.tight_layout()
plt.show()
