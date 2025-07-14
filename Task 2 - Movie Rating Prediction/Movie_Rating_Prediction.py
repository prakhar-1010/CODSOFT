# Task 2 – Movie Rating Prediction | CODSOFT Internship

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: Load the dataset with correct encoding
df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")
print("✅ Data Loaded")

# STEP 2: Drop unnecessary columns
df.drop(columns=['Name', 'Actor 2', 'Actor 3'], inplace=True)

# STEP 3: Handle missing values
df.dropna(subset=['Rating'], inplace=True)
df.ffill(inplace=True)  # forward fill to handle remaining missing values

# STEP 4: Clean 'Votes' column (remove commas and convert to int)
df['Votes'] = df['Votes'].astype(str).str.replace(",", "").replace("-", "0")
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce').fillna(0).astype(int)

# STEP 5: Convert 'Year' to numeric (handle messy rows)
df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)

# STEP 6: Clean 'Duration' (convert minutes to int)
df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce').fillna(0).astype(int)

# STEP 7: Encode categorical variables
label_cols = ['Genre', 'Director', 'Actor 1']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# STEP 8: Define features and target
X = df.drop('Rating', axis=1)
y = df['Rating']

# STEP 9: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 10: Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# STEP 11: Make predictions
y_pred = model.predict(X_test)

# STEP 12: Evaluate the model
print("\n🎯 Model Evaluation")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# STEP 13: Plot feature importance
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()
