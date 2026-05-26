# Predict student grades based on study habits and lifestyle factors
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/student.csv")


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
    "data/graph/graph.Attendance-grads.1.png",
    dpi=300
)

