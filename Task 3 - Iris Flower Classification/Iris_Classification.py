# Task 3 – Iris Flower Classification | CODSOFT Internship

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# STEP 1: Load Dataset
df = pd.read_csv("IRIS.csv")  # Make sure it's in the same folder
print("✅ Data Loaded")
print(df.head())

# STEP 2: Rename Columns (optional but cleaner)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# STEP 3: Encode Target Label (Species)
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])  # 0: setosa, 1: versicolor, 2: virginica

# STEP 4: Define Features and Target
X = df.drop('species', axis=1)
y = df['species']

# STEP 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 6: Train Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# STEP 7: Make Predictions
y_pred = model.predict(X_test)

# STEP 8: Evaluate Model
print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# STEP 9: Feature Importance
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(8, 5))
plt.title("Feature Importance")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()
