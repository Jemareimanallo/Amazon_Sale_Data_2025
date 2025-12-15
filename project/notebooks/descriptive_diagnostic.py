import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
DATA_PATH = "data/amazon_sales_data 2025.csv"
OUTPUT_DIR = "outputs/descriptive_charts"
CLEANED_DATA_PATH = "outputs/cleaned_dataset.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# -------------------------
# Data Cleaning
# -------------------------
df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)
df.to_csv(CLEANED_DATA_PATH, index=False)

# -------------------------
# Descriptive Analytics
# -------------------------
numeric_cols = df.select_dtypes(include="number").columns

# Descriptive statistics
desc = df[numeric_cols].describe()
desc.to_csv(os.path.join(OUTPUT_DIR, "descriptive_stats.csv"))

# Histograms
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{col}_hist.png"))
    plt.close()

# Categorical value counts
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    df[col].value_counts().to_csv(os.path.join(OUTPUT_DIR, f"{col}_value_counts.csv"))

# -------------------------
# Diagnostic Analytics
# -------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

for i, col1 in enumerate(numeric_cols):
    for col2 in numeric_cols[i+1:]:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[col1], y=df[col2])
        plt.title(f"{col1} vs {col2}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{col1}_vs_{col2}.png"))
        plt.close()

print("Descriptive and Diagnostic Analytics completed.")
