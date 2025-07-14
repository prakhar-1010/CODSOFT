# Titanic Survival Prediction using Titanic-Dataset.csv

# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# STEP 2: Load Dataset
df = pd.read_csv('Titanic-Dataset.csv')
print("✅ Data Loaded Successfully!\n")
print(df.head())

# STEP 3: Check for missing values
print("\nMissing values before cleaning:\n", df.isnull().sum())

# STEP 4: Fill or drop missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)

# STEP 5: Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])        # male = 1, female = 0
df['Embarked'] = le.fit_transform(df['Embarked'])

# STEP 6: Drop irrelevant columns
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# STEP 7: Features and Target
X = df.drop('Survived', axis=1)
y = df['Survived']

# STEP 8: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 9: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# STEP 10: Predictions and Evaluation
y_pred = model.predict(X_test)

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# STEP 11: Plot Feature Importance
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10,6))
plt.title("Feature Importance")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()
