import pandas as pd
import numpy as np

# ==========================
# Read Excel file
# ==========================

file_path = "../data/VIX_VVIX_clean.xlsx"
df = pd.read_excel(file_path, sheet_name="Data")

# ==========================
# Prepare data
# ==========================

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df.set_index("Date")

# ==========================
# Create new variables
# ==========================

df["dVIX"] = df["VIX"].diff()
df["dVVIX"] = df["VVIX"].diff()

df["rVIX"] = np.log(df["VIX"]).diff()
df["rVVIX"] = np.log(df["VVIX"]).diff()

# Remove first observation (contains NaN)
df = df.dropna()

# ==========================
# Print summary
# ==========================

print(df.head())

print("\n")
print(df.info())

print("\nMissing values:")
print(df.isna().sum())