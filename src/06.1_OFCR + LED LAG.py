from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

project_folder = Path(__file__).resolve().parent.parent

fsi_file = project_folder / "data" / "fsi.csv"
vix_vvix_file = project_folder / "data" / "VIX_VVIX_clean.xlsx"

output_folder = project_folder / "output"
tables_folder = output_folder / "tables"
figures_folder = output_folder / "figures" / "lead_lag"

tables_folder.mkdir(parents=True, exist_ok=True)
figures_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

STRESS_PERCENTILE = 0.90

# Lead horizons:
# Corr(ΔVVIX_t, ΔVIX_{t+k})
LEADS = [0, 1, 2, 3]


# ============================================================
# 3. LOAD VIX-VVIX DATA
# ============================================================

df = pd.read_excel(vix_vvix_file)

print("\nVIX-VVIX columns:")
print(df.columns.tolist())

# Standardise date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = (
    df.dropna(subset=["Date"])
    .sort_values("Date")
    .drop_duplicates(subset="Date")
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Detect VIX and VVIX columns
# ------------------------------------------------------------

def find_column(columns, exact_name):
    """
    Find a column ignoring upper/lower case and spaces.
    """
    clean_map = {
        str(col).strip().upper(): col
        for col in columns
    }

    key = exact_name.strip().upper()

    if key not in clean_map:
        raise ValueError(
            f"Could not find column '{exact_name}'. "
            f"Available columns: {list(columns)}"
        )

    return clean_map[key]


vix_col = find_column(df.columns, "VIX")
vvix_col = find_column(df.columns, "VVIX")

df[vix_col] = pd.to_numeric(df[vix_col], errors="coerce")
df[vvix_col] = pd.to_numeric(df[vvix_col], errors="coerce")


# ============================================================
# 4. FIRST DIFFERENCES
# ============================================================

# Following the lead-lag analysis based on innovations
df["dVIX"] = df[vix_col].diff()
df["dVVIX"] = df[vvix_col].diff()


# ============================================================
# 5. LOAD OFR FINANCIAL STRESS INDEX
# ============================================================

fsi = pd.read_csv(fsi_file)

print("\nOFR FSI columns:")
print(fsi.columns.tolist())

fsi["Date"] = pd.to_datetime(fsi["Date"], errors="coerce")

fsi = (
    fsi.dropna(subset=["Date"])
    .sort_values("Date")
    .drop_duplicates(subset="Date")
    .reset_index(drop=True)
)


# ============================================================
# 6. IDENTIFY THE OFR FSI VALUE COLUMN
# ============================================================

# Exclude Date and search among numeric columns.
possible_fsi_columns = [
    col for col in fsi.columns
    if col != "Date"
]

# Convert possible columns to numeric where possible
for col in possible_fsi_columns:
    fsi[col] = pd.to_numeric(fsi[col], errors="coerce")

numeric_fsi_columns = [
    col for col in possible_fsi_columns
    if fsi[col].notna().sum() > 0
]

if len(numeric_fsi_columns) == 0:
    raise ValueError(
        "No numeric OFR FSI column detected in fsi.csv."
    )

# If only one numeric column exists, use it automatically.
if len(numeric_fsi_columns) == 1:
    fsi_col = numeric_fsi_columns[0]

else:
    # Try to identify the most likely FSI column
    likely_names = [
        "OFR FSI",
        "OFR_FSI",
        "FSI",
        "Financial Stress Index",
        "financial_stress_index"
    ]

    fsi_col = None

    for candidate in likely_names:
        matches = [
            col for col in numeric_fsi_columns
            if str(col).strip().lower() == candidate.lower()
        ]

        if matches:
            fsi_col = matches[0]
            break

    if fsi_col is None:
        raise ValueError(
            "\nMultiple numeric columns found in fsi.csv:\n"
            f"{numeric_fsi_columns}\n\n"
            "Specify manually which one is the OFR FSI column."
        )


print(f"\nOFR FSI column used: {fsi_col}")


fsi = fsi[["Date", fsi_col]].copy()
fsi = fsi.rename(columns={fsi_col: "OFR_FSI"})


# ============================================================
# 7. MERGE VIX-VVIX WITH OFR FSI
# ============================================================

data = pd.merge(
    df[["Date", vix_col, vvix_col, "dVIX", "dVVIX"]],
    fsi,
    on="Date",
    how="inner"
)

data = (
    data.sort_values("Date")
    .reset_index(drop=True)
)

print("\nMerged sample:")
print(f"Start: {data['Date'].min().date()}")
print(f"End:   {data['Date'].max().date()}")
print(f"N:     {len(data)}")


# ============================================================
# 8. DEFINE HIGH-STRESS REGIME
# ============================================================

# Threshold calculated on the overlapping VIX-VVIX-OFR sample
stress_threshold = data["OFR_FSI"].quantile(STRESS_PERCENTILE)

data["High_Stress"] = data["OFR_FSI"] > stress_threshold


print("\n" + "=" * 60)
print("OFR HIGH-STRESS DEFINITION")
print("=" * 60)

print(
    f"{STRESS_PERCENTILE:.0%} percentile threshold: "
    f"{stress_threshold:.4f}"
)

print(
    f"High-stress observations: "
    f"{data['High_Stress'].sum()}"
)

print(
    f"Normal observations: "
    f"{(~data['High_Stress']).sum()}"
)

print(
    f"Share high-stress: "
    f"{data['High_Stress'].mean():.2%}"
)


# ============================================================
# 9. LEAD-LAG CORRELATION FUNCTION
# ============================================================

def lead_lag_correlations(sample, leads):
    """
    Computes:

        Corr(ΔVVIX_t, ΔVIX_{t+k})

    for each lead k.

    Positive k means VVIX is observed first and VIX later.
    """

    results = []

    for k in leads:

        temp = sample[["dVVIX", "dVIX"]].copy()

        # Future VIX change:
        # at row t we attach ΔVIX_{t+k}
        temp["future_dVIX"] = temp["dVIX"].shift(-k)

        temp = temp.dropna(
            subset=["dVVIX", "future_dVIX"]
        )

        correlation = temp["dVVIX"].corr(
            temp["future_dVIX"]
        )

        results.append({
            "k": k,
            "correlation": correlation,
            "N": len(temp)
        })

    return pd.DataFrame(results)


# ============================================================
# 10. FULL-SAMPLE LEAD-LAG
# ============================================================

full_results = lead_lag_correlations(
    data,
    LEADS
)

full_results["Sample"] = "Full sample"


# ============================================================
# 11. HIGH-STRESS LEAD-LAG
# ============================================================

stress_data = data.loc[
    data["High_Stress"]
].copy()

stress_results = lead_lag_correlations(
    stress_data,
    LEADS
)

stress_results["Sample"] = "OFR > P90"


# ============================================================
# 12. NON-STRESS LEAD-LAG
# ============================================================

normal_data = data.loc[
    ~data["High_Stress"]
].copy()

normal_results = lead_lag_correlations(
    normal_data,
    LEADS
)

normal_results["Sample"] = "OFR <= P90"


# ============================================================
# 13. COMBINE RESULTS
# ============================================================

results_long = pd.concat(
    [
        full_results,
        stress_results,
        normal_results
    ],
    ignore_index=True
)

results_long = results_long[
    ["Sample", "k", "correlation", "N"]
]


print("\n" + "=" * 60)
print("LEAD-LAG CORRELATIONS")
print("Corr(ΔVVIX_t, ΔVIX_{t+k})")
print("=" * 60)

print(
    results_long.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 14. CREATE THESIS-STYLE TABLE
# ============================================================

correlation_table = results_long.pivot(
    index="Sample",
    columns="k",
    values="correlation"
)

correlation_table.columns = [
    f"k={k}" for k in correlation_table.columns
]

# Desired ordering
correlation_table = correlation_table.reindex(
    [
        "Full sample",
        "OFR > P90",
        "OFR <= P90"
    ]
)

print("\n" + "=" * 60)
print("CORRELATION TABLE")
print("=" * 60)

print(
    correlation_table.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 15. SAVE TABLES
# ============================================================

results_long.to_csv(
    tables_folder / "ofr_stress_lead_lag_long.csv",
    index=False
)

correlation_table.to_csv(
    tables_folder / "ofr_stress_lead_lag_comparison.csv"
)

data[[
    "Date",
    vix_col,
    vvix_col,
    "dVIX",
    "dVVIX",
    "OFR_FSI",
    "High_Stress"
]].to_csv(
    tables_folder / "ofr_stress_classification.csv",
    index=False
)


# ============================================================
# 16. PLOT: FULL SAMPLE VS STRESS VS NON-STRESS
# ============================================================

plot_table = results_long.pivot(
    index="k",
    columns="Sample",
    values="correlation"
)

plot_table = plot_table[
    [
        "Full sample",
        "OFR > P90",
        "OFR <= P90"
    ]
]


x = np.arange(len(LEADS))
width = 0.25


fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    x - width,
    plot_table["Full sample"],
    width,
    label="Full sample"
)

ax.bar(
    x,
    plot_table["OFR > P90"],
    width,
    label="High stress: OFR > P90"
)

ax.bar(
    x + width,
    plot_table["OFR <= P90"],
    width,
    label="Non-stress: OFR <= P90"
)


ax.axhline(
    y=0,
    linewidth=0.8
)

ax.set_xticks(x)
ax.set_xticklabels(
    [f"k={k}" for k in LEADS]
)

ax.set_xlabel("Lead horizon")
ax.set_ylabel("Cross-correlation")

ax.set_title(
    "VIX-VVIX Lead-Lag Cross-Correlations Conditional on OFR Financial Stress"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    figures_folder / "ofr_stress_vix_vvix_lead_lag.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 17. LIST HIGH-STRESS DATES
# ============================================================

stress_dates = data.loc[
    data["High_Stress"],
    ["Date", "OFR_FSI", "dVIX", "dVVIX"]
].copy()

stress_dates = stress_dates.sort_values(
    "OFR_FSI",
    ascending=False
)

stress_dates.to_csv(
    tables_folder / "ofr_high_stress_dates.csv",
    index=False
)


print("\nTop 20 highest OFR stress observations:")

print(
    stress_dates.head(20).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\nAnalysis completed successfully.")