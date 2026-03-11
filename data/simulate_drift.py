import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris_data = load_iris(as_frame=True)
df = iris_data.frame

X = df.drop(columns=["target"])
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Simulate feature drift
X_drifted = X_test.copy()

X_drifted["sepal length (cm)"] += np.random.normal(
    loc=2.0,
    scale=0.3,
    size=len(X_drifted)
)

# Save datasets
X_train.to_csv("reference_data.csv", index=False)
X_drifted.to_csv("current_data.csv", index=False)

print("Drifted dataset created.")