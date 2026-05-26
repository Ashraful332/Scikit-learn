from sklearn.linear_model import LogisticRegression

# Example data
X = [[40], [50], [60], [70]]
y = [0, 0, 1, 1]

# Create model
model = LogisticRegression()

# Train model
model.fit(X, y)

# Predict
print(model.predict([[5]]))