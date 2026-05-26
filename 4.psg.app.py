# Predict student grades based on study habits and lifestyle factors
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# =========================
# LOAD CSV
# =========================

df = pd.read_csv("data/student.csv")

def Image(df):

    # 1. study/grads graph
    plt.scatter(df["Study Hours"], df["Grades"])
    plt.xlabel("Study Hours")
    plt.ylabel("Grades")
    # Save
    plt.savefig(
        "data/graph/graph.study-grads.1.png",
        dpi=300
    )

    # 2. Sleep Hours/grads graph
    plt.scatter(df["Sleep Hours"], df["Grades"])
    plt.xlabel("Sleep Hours")
    plt.ylabel("Grades")
    # Save
    plt.savefig(
        "data/graph/graph.Sleep-grads.1.png",
        dpi=300
    )

    # 3. moony/grads graph
    plt.scatter(df["Socioeconomic Score"], df["Grades"])
    plt.xlabel("Socioeconomic Score")
    plt.ylabel("Grades")
    # Save
    plt.savefig(
        "data/graph/graph.rich-grads.1.png",
        dpi=300
    )

    # 4. Attendance/grads graph
    plt.scatter(df["Attendance (%)"], df["Grades"])
    plt.xlabel("Attendance (%)")
    plt.ylabel("Grades")
    # Save
    plt.savefig(
        "data/graph/graph.Attendance-grads.3.png",
        dpi=300
    )
# Image(df)

# Simple Linear Regression Model

# =========================
# TRAIN MODEL
# =========================

# Features
X = df[[
    "Study Hours",
    "Sleep Hours",
    "Socioeconomic Score",
    "Attendance (%)"
]]

# Target
y = df["Grades"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

print("Training complete!")

# Predict
predictions = model.predict(X_test)

print(predictions[1])