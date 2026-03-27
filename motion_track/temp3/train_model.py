# train_model.py
import joblib
from sklearn.ensemble import RandomForestClassifier

# Dummy data (replace later)
X = [
    [90, 90, 100, 2, 30],
    [85, 88, 95, 3, 25],
    [60, 65, 70, 8, 10],
    [55, 60, 65, 9, 8],
]

y = [0, 0, 1, 1]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("✅ model.pkl created")