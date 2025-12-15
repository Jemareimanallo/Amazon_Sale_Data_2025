import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

DATA_PATH = "outputs/cleaned_dataset.csv"
OUTPUT_DIR = "outputs/predictive_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
numeric_cols = df.select_dtypes(include="number").columns

if len(numeric_cols) < 2:
    raise ValueError("Not enough numeric columns for prediction.")

target = numeric_cols[-1]
features = numeric_cols.drop(target)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Actual vs Predicted ({target})")
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"))
plt.close()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
with open(os.path.join(OUTPUT_DIR, "model_metrics.txt"), "w") as f:
    f.write(f"MSE: {mse}\nR²: {r2}\n")

print("Predictive Analytics completed.")
