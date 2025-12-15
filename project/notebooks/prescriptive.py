import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD
import os

PRED_PATH = "outputs/predictive_results/predictions.csv"
OUTPUT_DIR = "outputs/prescriptive_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pred_df = pd.read_csv(PRED_PATH)
if pred_df.empty:
    raise ValueError("Predictions file is empty!")

# Ensure non-negative predictions
pred_df["Predicted"] = pred_df["Predicted"].apply(lambda x: max(x, 0))
pred_df["Cost"] = pred_df["Predicted"] * 10
budget = 1000

prob = LpProblem("Maximize_Sales", LpMaximize)
x = {i: LpVariable(f"x_{i}", lowBound=0) for i in pred_df.index}  # continuous

prob += lpSum([pred_df.loc[i, "Predicted"] * x[i] for i in pred_df.index])
prob += lpSum([pred_df.loc[i, "Cost"] * x[i] for i in pred_df.index]) <= budget

solver = PULP_CBC_CMD(msg=0, timeLimit=10)
prob.solve(solver)

results = pd.DataFrame({
    "Index": pred_df.index,
    "Units_Allocated": [x[i].varValue for i in pred_df.index],
    "Predicted_Sales": pred_df["Predicted"]
})
results.to_csv(os.path.join(OUTPUT_DIR, "optimization_results.csv"), index=False)

print("Prescriptive Analytics completed.")
