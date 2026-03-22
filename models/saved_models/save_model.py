import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load data
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

# Train best model (update if needed)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("models/saved_models/saved_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")