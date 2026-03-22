# Preprocessing script will go here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("data/raw/raw_dataset.csv")

print("Initial Shape:", df.shape)

# -------------------------------
# Drop unnecessary column
# -------------------------------
df = df.drop(columns=['employee_id'])

# -------------------------------
# Encode categorical columns
# -------------------------------
le = LabelEncoder()

categorical_cols = ['department', 'access_level']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# Define target and features
# -------------------------------
target_column = 'breach_risk_label'

X = df.drop(columns=[target_column])
y = df[target_column]

# -------------------------------
# Feature scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# Save processed data
# -------------------------------
os.makedirs("data/processed", exist_ok=True)

pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

df.to_csv("data/processed/cleaned_dataset.csv", index=False)

print("✅ Preprocessing completed successfully!")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

import matplotlib.pyplot as plt

y.value_counts().plot(kind='bar')
plt.title("Risk Distribution")

os.makedirs("data/visualizations", exist_ok=True)
plt.savefig("data/visualizations/risk_distribution.png")