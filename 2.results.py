from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

# read CSV
df = pd.read_csv("data/results.csv")

# Features
X = df[[
    "Bangla",
    "English",
    "Science",
    "Maths",
    "History",
    "Geograpgy"
]]

# Target
y = df["Results"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression()

# Train
model.fit(X_train, y_train)

print("Training complete!")

# Predict Grades
predictions = model.predict(X_test)

print(predictions[41])