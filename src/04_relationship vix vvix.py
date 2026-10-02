from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from arch.unitroot import ADF, PhillipsPerron


# ============================================================
# 1. Project paths
# ============================================================

project_folder = Path(__file__).resolve().parent.parent

input_file = project_folder / "data" / "VIX_VVIX_clean.xlsx"

output_folder = project_folder / "output"
figures_folder = output_folder / "figures"
tables_folder = output_folder / "tables"

figures_folder.mkdir(parents=True, exist_ok=True)
tables_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Data loading and preparation
# ============================================================

df = pd.read_excel(
    input_file,
    sheet_name="Data"
)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df.drop_duplicates(subset="Date")

df = df.set_index("Date")

df = df.dropna(subset=["VIX", "VVIX"])


# ============================================================
# 3. Data validation
# ============================================================

if (df[["VIX", "VVIX"]] <= 0).any().any():
    raise ValueError(
        "VIX and VVIX must be strictly positive "
        "to compute logarithmic transformations."
    )


# ============================================================
# 4. Series transformations
# ============================================================

# First differences (innovations)
df["dVIX"] = df["VIX"].diff()
df["dVVIX"] = df["VVIX"].diff()

# Log returns
df["rVIX"] = np.log(df["VIX"]).diff()
df["rVVIX"] = np.log(df["VVIX"]).diff()

# Log levels
df["logVIX"] = np.log(df["VIX"])
df["logVVIX"] = np.log(df["VVIX"])


# ============================================================
# 5. Plot colors
# ============================================================

vix_color = "#2F6FB0"
vvix_color = "#4B2E83"


# ============================================================
# 6. Extreme movements
# ============================================================

# ------------------------------------------------------------
# 6.1 Percentile thresholds
# ------------------------------------------------------------

vix_q01 = df["rVIX"].quantile(0.01)
vix_q99 = df["rVIX"].quantile(0.99)

vvix_q01 = df["rVVIX"].quantile(0.01)
vvix_q99 = df["rVVIX"].quantile(0.99)


print("\n1% empirical tail thresholds")
print("------------------------------------")
print(f"VIX  1st percentile : {vix_q01:.4f} ({vix_q01 * 100:.2f}%)")
print(f"VIX 99th percentile : {vix_q99:.4f} ({vix_q99 * 100:.2f}%)")
print(f"VVIX 1st percentile : {vvix_q01:.4f} ({vvix_q01 * 100:.2f}%)")
print(f"VVIX 99th percentile: {vvix_q99:.4f} ({vvix_q99 * 100:.2f}%)")


# ------------------------------------------------------------
# 6.2 Extreme-movement indicators
# ------------------------------------------------------------

df["VIX_extreme_positive"] = df["rVIX"] > vix_q99
df["VIX_extreme_negative"] = df["rVIX"] < vix_q01

df["VVIX_extreme_positive"] = df["rVVIX"] > vvix_q99
df["VVIX_extreme_negative"] = df["rVVIX"] < vvix_q01


# General extreme indicator, irrespective of direction
df["VIX_extreme"] = (
    df["VIX_extreme_positive"]
    | df["VIX_extreme_negative"]
)

df["VVIX_extreme"] = (
    df["VVIX_extreme_positive"]
    | df["VVIX_extreme_negative"]
)


# ------------------------------------------------------------
# 6.3 Count extreme observations
# ------------------------------------------------------------

extreme_counts = pd.DataFrame(
    {
        "VIX": [
            df["VIX_extreme_positive"].sum(),
            df["VIX_extreme_negative"].sum(),
            df["VIX_extreme"].sum()
        ],
        "VVIX": [
            df["VVIX_extreme_positive"].sum(),
            df["VVIX_extreme_negative"].sum(),
            df["VVIX_extreme"].sum()
        ]
    },
    index=[
        "Positive extremes",
        "Negative extremes",
        "Total extremes"
    ]
)

print("\nNumber of extreme movements")
print("------------------------------------")
print(extreme_counts)


# ------------------------------------------------------------
# 6.4 Save thresholds and counts
# ------------------------------------------------------------

extreme_thresholds = pd.DataFrame(
    {
        "Lower_1pct": [
            vix_q01,
            vvix_q01
        ],
        "Upper_1pct": [
            vix_q99,
            vvix_q99
        ]
    },
    index=[
        "VIX log returns",
        "VVIX log returns"
    ]
)

extreme_thresholds.to_excel(
    tables_folder / "extreme_movement_thresholds.xlsx"
)

extreme_counts.to_excel(
    tables_folder / "extreme_movement_counts.xlsx"
)


# ============================================================
# 7. Plot extreme movements over time
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 9),
    sharex=True
)


# ------------------------------------------------------------
# VIX
# ------------------------------------------------------------

axes[0].plot(
    df.index,
    df["rVIX"] * 100,
    linewidth=0.8,
    color=vix_color,
    alpha=0.7,
    label="Daily log return"
)

axes[0].scatter(
    df.index[df["VIX_extreme_positive"]],
    df.loc[df["VIX_extreme_positive"], "rVIX"] * 100,
    s=35,
    marker="^",
    label="Upper 1% extreme",
    zorder=3
)

axes[0].scatter(
    df.index[df["VIX_extreme_negative"]],
    df.loc[df["VIX_extreme_negative"], "rVIX"] * 100,
    s=35,
    marker="v",
    label="Lower 1% extreme",
    zorder=3
)

axes[0].axhline(
    vix_q99 * 100,
    linestyle="--",
    linewidth=1
)

axes[0].axhline(
    vix_q01 * 100,
    linestyle="--",
    linewidth=1
)

axes[0].axhline(
    0,
    linewidth=0.8
)

axes[0].set_title(
    "Extreme Daily Movements in VIX Log Returns"
)

axes[0].set_ylabel("Log return (%)")

axes[0].legend()


# ------------------------------------------------------------
# VVIX
# ------------------------------------------------------------

axes[1].plot(
    df.index,
    df["rVVIX"] * 100,
    linewidth=0.8,
    color=vvix_color,
    alpha=0.7,
    label="Daily log return"
)

axes[1].scatter(
    df.index[df["VVIX_extreme_positive"]],
    df.loc[df["VVIX_extreme_positive"], "rVVIX"] * 100,
    s=35,
    marker="^",
    label="Upper 1% extreme",
    zorder=3
)

axes[1].scatter(
    df.index[df["VVIX_extreme_negative"]],
    df.loc[df["VVIX_extreme_negative"], "rVVIX"] * 100,
    s=35,
    marker="v",
    label="Lower 1% extreme",
    zorder=3
)

axes[1].axhline(
    vvix_q99 * 100,
    linestyle="--",
    linewidth=1
)

axes[1].axhline(
    vvix_q01 * 100,
    linestyle="--",
    linewidth=1
)

axes[1].axhline(
    0,
    linewidth=0.8
)

axes[1].set_title(
    "Extreme Daily Movements in VVIX Log Returns"
)

axes[1].set_ylabel("Log return (%)")
axes[1].set_xlabel("Date")

axes[1].legend()


plt.tight_layout()

jump_figure_path = (
    figures_folder
    / "extreme_movements_log_returns.png"
)

fig.savefig(
    jump_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 8. Extreme co-movements between VIX and VVIX
# ============================================================

# Same-direction co-movements
df["co_extreme_positive"] = (
    df["VIX_extreme_positive"]
    & df["VVIX_extreme_positive"]
)

df["co_extreme_negative"] = (
    df["VIX_extreme_negative"]
    & df["VVIX_extreme_negative"]
)

# Opposite-direction co-movements
df["co_extreme_opposite"] = (
    (df["VIX_extreme_positive"] & df["VVIX_extreme_negative"])
    |
    (df["VIX_extreme_negative"] & df["VVIX_extreme_positive"])
)

# Both indices are extreme, irrespective of direction
df["co_extreme_any"] = (
    df["VIX_extreme"]
    & df["VVIX_extreme"]
)


# ------------------------------------------------------------
# Counts
# ------------------------------------------------------------

n_vix_positive = int(df["VIX_extreme_positive"].sum())
n_vix_negative = int(df["VIX_extreme_negative"].sum())
n_vix_extreme = int(df["VIX_extreme"].sum())

n_vvix_positive = int(df["VVIX_extreme_positive"].sum())
n_vvix_negative = int(df["VVIX_extreme_negative"].sum())
n_vvix_extreme = int(df["VVIX_extreme"].sum())

n_co_positive = int(df["co_extreme_positive"].sum())
n_co_negative = int(df["co_extreme_negative"].sum())
n_co_opposite = int(df["co_extreme_opposite"].sum())
n_co_any = int(df["co_extreme_any"].sum())


# ------------------------------------------------------------
# Conditional co-occurrence rates
# ------------------------------------------------------------

pct_vix_with_vvix = (
    n_co_any / n_vix_extreme * 100
    if n_vix_extreme > 0 else np.nan
)

pct_vvix_with_vix = (
    n_co_any / n_vvix_extreme * 100
    if n_vvix_extreme > 0 else np.nan
)

pct_positive = (
    n_co_positive / n_vix_positive * 100
    if n_vix_positive > 0 else np.nan
)

pct_negative = (
    n_co_negative / n_vix_negative * 100
    if n_vix_negative > 0 else np.nan
)


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

print("\nExtreme co-movements: VIX and VVIX")
print("=" * 55)

print("\nIndividual extreme observations:")
print(f"VIX positive extremes:   {n_vix_positive}")
print(f"VIX negative extremes:   {n_vix_negative}")
print(f"VIX total extremes:      {n_vix_extreme}")
print(f"VVIX positive extremes:  {n_vvix_positive}")
print(f"VVIX negative extremes:  {n_vvix_negative}")
print(f"VVIX total extremes:     {n_vvix_extreme}")

print("\nCoincident extreme movements:")
print(f"Positive-positive:        {n_co_positive}")
print(f"Negative-negative:        {n_co_negative}")
print(f"Opposite direction:       {n_co_opposite}")
print(f"Total coincident:         {n_co_any}")

print("\nCo-occurrence rates:")
print(
    f"VIX extremes accompanied by a VVIX extreme: "
    f"{n_co_any}/{n_vix_extreme} "
    f"({pct_vix_with_vvix:.2f}%)"
)

print(
    f"VVIX extremes accompanied by a VIX extreme: "
    f"{n_co_any}/{n_vvix_extreme} "
    f"({pct_vvix_with_vix:.2f}%)"
)

print(
    f"VIX positive extremes accompanied by "
    f"VVIX positive extremes: "
    f"{n_co_positive}/{n_vix_positive} "
    f"({pct_positive:.2f}%)"
)

print(
    f"VIX negative extremes accompanied by "
    f"VVIX negative extremes: "
    f"{n_co_negative}/{n_vix_negative} "
    f"({pct_negative:.2f}%)"
)


# ------------------------------------------------------------
# Table containing all coincident extreme dates
# ------------------------------------------------------------

co_extreme_dates = df.loc[
    df["co_extreme_any"],
    [
        "rVIX",
        "rVVIX",
        "VIX_extreme_positive",
        "VIX_extreme_negative",
        "VVIX_extreme_positive",
        "VVIX_extreme_negative"
    ]
].copy()

co_extreme_dates["VIX log return (%)"] = (
    co_extreme_dates["rVIX"] * 100
)

co_extreme_dates["VVIX log return (%)"] = (
    co_extreme_dates["rVVIX"] * 100
)

co_extreme_dates["Type"] = np.select(
    [
        (
            co_extreme_dates["VIX_extreme_positive"]
            & co_extreme_dates["VVIX_extreme_positive"]
        ),
        (
            co_extreme_dates["VIX_extreme_negative"]
            & co_extreme_dates["VVIX_extreme_negative"]
        )
    ],
    [
        "Positive-positive",
        "Negative-negative"
    ],
    default="Opposite direction"
)

co_extreme_dates = co_extreme_dates[
    [
        "VIX log return (%)",
        "VVIX log return (%)",
        "Type"
    ]
]

print("\nDates with extreme movements in both indices:")
print("=" * 55)
print(co_extreme_dates.round(2).to_string())


# Save co-movement table
co_extreme_dates.to_excel(
    tables_folder / "extreme_co_movements.xlsx"
)


# ============================================================
# 9. Full extreme-event lists and annual summary
# ============================================================

# ------------------------------------------------------------
# 9.1 Complete list of VIX extreme observations
# ------------------------------------------------------------

vix_extreme_dates = df.loc[
    df["VIX_extreme"],
    [
        "rVIX",
        "rVVIX",
        "VIX_extreme_positive",
        "VIX_extreme_negative",
        "VVIX_extreme",
        "co_extreme_any"
    ]
].copy()

vix_extreme_dates["VIX log return (%)"] = (
    vix_extreme_dates["rVIX"] * 100
)

vix_extreme_dates["VVIX log return (%)"] = (
    vix_extreme_dates["rVVIX"] * 100
)

vix_extreme_dates["VIX Type"] = np.where(
    vix_extreme_dates["VIX_extreme_positive"],
    "Positive",
    "Negative"
)

vix_extreme_dates["VVIX also extreme?"] = np.where(
    vix_extreme_dates["VVIX_extreme"],
    "Yes",
    "No"
)

vix_extreme_dates = vix_extreme_dates[
    [
        "VIX log return (%)",
        "VVIX log return (%)",
        "VIX Type",
        "VVIX also extreme?"
    ]
]


# ------------------------------------------------------------
# 9.2 Complete list of VVIX extreme observations
# ------------------------------------------------------------

vvix_extreme_dates = df.loc[
    df["VVIX_extreme"],
    [
        "rVIX",
        "rVVIX",
        "VVIX_extreme_positive",
        "VVIX_extreme_negative",
        "VIX_extreme",
        "co_extreme_any"
    ]
].copy()

vvix_extreme_dates["VIX log return (%)"] = (
    vvix_extreme_dates["rVIX"] * 100
)

vvix_extreme_dates["VVIX log return (%)"] = (
    vvix_extreme_dates["rVVIX"] * 100
)

vvix_extreme_dates["VVIX Type"] = np.where(
    vvix_extreme_dates["VVIX_extreme_positive"],
    "Positive",
    "Negative"
)

vvix_extreme_dates["VIX also extreme?"] = np.where(
    vvix_extreme_dates["VIX_extreme"],
    "Yes",
    "No"
)

vvix_extreme_dates = vvix_extreme_dates[
    [
        "VIX log return (%)",
        "VVIX log return (%)",
        "VVIX Type",
        "VIX also extreme?"
    ]
]


# ------------------------------------------------------------
# 9.3 Annual counts
# ------------------------------------------------------------

annual_summary = pd.DataFrame(
    index=sorted(df.index.year.unique())
)

annual_summary.index.name = "Year"

annual_summary["VIX positive"] = (
    df["VIX_extreme_positive"]
    .groupby(df.index.year)
    .sum()
    .astype(int)
)

annual_summary["VIX negative"] = (
    df["VIX_extreme_negative"]
    .groupby(df.index.year)
    .sum()
    .astype(int)
)

annual_summary["VIX total"] = (
    df["VIX_extreme"]
    .groupby(df.index.year)
    .sum()
    .astype(int)
)

annual_summary["VVIX positive"] = (
    df["VVIX_extreme_positive"]
    .groupby(df.index.year)
    .sum()
    .astype(int)
)

annual_summary["VVIX negative"] = (
    df["VVIX_extreme_negative"]
    .groupby(df.index.year)
    .sum()
    .astype(int)
)

annual_summary["VVIX total"] = (
    df["VVIX_extreme"]
    .groupby(df.index.year)
    .sum()
    .astype(int)
)

annual_summary["Co-extreme positive"] = (
    df["co_extreme_positive"]
    .groupby(df.index.year)
    .sum()
    .astype(int)
)

annual_summary["Co-extreme negative"] = (
    df["co_extreme_negative"]
    .groupby(df.index.year)
    .sum()
    .astype(int)
)

annual_summary["Co-extreme total"] = (
    df["co_extreme_any"]
    .groupby(df.index.year)
    .sum()
    .astype(int)
)

annual_summary = annual_summary.fillna(0).astype(int)

# Add total row
annual_summary.loc["Total"] = annual_summary.sum()


# ------------------------------------------------------------
# 9.4 Print results
# ------------------------------------------------------------

print("\nComplete list of VIX extreme observations")
print("=" * 70)
print(vix_extreme_dates.round(2).to_string())

print("\nComplete list of VVIX extreme observations")
print("=" * 70)
print(vvix_extreme_dates.round(2).to_string())

print("\nAnnual frequency of extreme movements")
print("=" * 100)
print(annual_summary.to_string())


# ------------------------------------------------------------
# 9.5 Save results
# ------------------------------------------------------------

vix_extreme_dates.to_excel(
    tables_folder / "VIX_all_extreme_movements.xlsx"
)

vvix_extreme_dates.to_excel(
    tables_folder / "VVIX_all_extreme_movements.xlsx"
)

annual_summary.to_excel(
    tables_folder / "annual_extreme_movements_summary.xlsx"
)


# ============================================================
# 10. Rolling correlation between VIX and VVIX innovations
# ============================================================

rolling_window = 100

df["rolling_corr_dVIX_dVVIX"] = (
    df["dVIX"]
    .rolling(window=rolling_window)
    .corr(df["dVVIX"])
)


# ============================================================
# 11. Plot: 100-day rolling correlation
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.plot(
    df.index,
    df["rolling_corr_dVIX_dVVIX"],
    color="black",
    linewidth=1.5
)

ax.set_title(
    "100-Day Rolling Correlation between VIX and VVIX First Differences",
    fontsize=14
)

ax.set_xlabel("Date")
ax.set_ylabel("Correlation")

# Restricted axis for readability
ax.set_ylim(0.3, 1.0)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)

fig.tight_layout()

correlation_figure_path = (
    figures_folder
    / "rolling_correlation_vix_vvix_first_differences.png"
)

fig.savefig(
    correlation_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 12. Correlation summary
# ============================================================

# Correlation between index levels
corr_levels = df["VIX"].corr(
    df["VVIX"]
)

# Correlation between daily first differences
corr_differences = df["dVIX"].corr(
    df["dVVIX"]
)

# Summary statistics of the rolling correlation
rolling_corr_mean = (
    df["rolling_corr_dVIX_dVVIX"].mean()
)

rolling_corr_min = (
    df["rolling_corr_dVIX_dVVIX"].min()
)

rolling_corr_max = (
    df["rolling_corr_dVIX_dVVIX"].max()
)


# ------------------------------------------------------------
# 12.1 Print correlation results
# ------------------------------------------------------------

print("\nCorrelation analysis")
print("=" * 55)

print(
    f"Correlation between VIX and VVIX levels: "
    f"{corr_levels:.3f}"
)

print(
    f"Correlation between VIX and VVIX first differences: "
    f"{corr_differences:.3f}"
)

print(
    f"Average 100-day rolling correlation: "
    f"{rolling_corr_mean:.3f}"
)

print(
    f"Minimum 100-day rolling correlation: "
    f"{rolling_corr_min:.3f}"
)

print(
    f"Maximum 100-day rolling correlation: "
    f"{rolling_corr_max:.3f}"
)


# ------------------------------------------------------------
# 12.2 Create correlation summary table
# ------------------------------------------------------------

correlation_summary = pd.DataFrame(
    {
        "Statistic": [
            "Correlation between VIX and VVIX levels",
            "Correlation between VIX and VVIX first differences",
            "Average 100-day rolling correlation",
            "Minimum 100-day rolling correlation",
            "Maximum 100-day rolling correlation"
        ],
        "Value": [
            corr_levels,
            corr_differences,
            rolling_corr_mean,
            rolling_corr_min,
            rolling_corr_max
        ]
    }
)

correlation_summary["Value"] = (
    correlation_summary["Value"].round(3)
)


# ------------------------------------------------------------
# 12.3 Save correlation summary table
# ------------------------------------------------------------

correlation_summary.to_excel(
    tables_folder / "correlation_summary.xlsx",
    index=False
)


# ============================================================
# 13. Output confirmation
# ============================================================

print("\nFiles saved successfully")
print("=" * 55)

print("\nFigures:")
print(jump_figure_path)
print(correlation_figure_path)

print("\nTables:")
print(tables_folder / "extreme_movement_thresholds.xlsx")
print(tables_folder / "extreme_movement_counts.xlsx")
print(tables_folder / "extreme_co_movements.xlsx")
print(tables_folder / "VIX_all_extreme_movements.xlsx")
print(tables_folder / "VVIX_all_extreme_movements.xlsx")
print(tables_folder / "annual_extreme_movements_summary.xlsx")
print(tables_folder / "correlation_summary.xlsx")


# ============================================================
# 13. Investigation of minimum and maximum rolling correlation
# ============================================================

rolling_corr = df["rolling_corr_dVIX_dVVIX"].dropna()

# Dates and values of minimum and maximum rolling correlation
min_corr_date = rolling_corr.idxmin()
max_corr_date = rolling_corr.idxmax()

min_corr_value = rolling_corr.loc[min_corr_date]
max_corr_value = rolling_corr.loc[max_corr_date]


# ------------------------------------------------------------
# 13.1 Extract the corresponding 100-day windows
# ------------------------------------------------------------

min_position = df.index.get_loc(min_corr_date)
max_position = df.index.get_loc(max_corr_date)

min_window = df.iloc[
    min_position - rolling_window + 1:
    min_position + 1
][
    [
        "VIX",
        "VVIX",
        "dVIX",
        "dVVIX"
    ]
].copy()

max_window = df.iloc[
    max_position - rolling_window + 1:
    max_position + 1
][
    [
        "VIX",
        "VVIX",
        "dVIX",
        "dVVIX"
    ]
].copy()


# ------------------------------------------------------------
# 13.2 Verify correlations within the extracted windows
# ------------------------------------------------------------

min_window_corr = min_window["dVIX"].corr(
    min_window["dVVIX"]
)

max_window_corr = max_window["dVIX"].corr(
    max_window["dVVIX"]
)


# ------------------------------------------------------------
# 13.3 Print results
# ------------------------------------------------------------

print("\nMinimum and maximum rolling-correlation episodes")
print("=" * 65)

print("\nMINIMUM CORRELATION")
print("-" * 40)

print(
    f"Rolling correlation: "
    f"{min_corr_value:.3f}"
)

print(
    f"Date of minimum: "
    f"{min_corr_date.strftime('%Y-%m-%d')}"
)

print(
    f"Window start: "
    f"{min_window.index.min().strftime('%Y-%m-%d')}"
)

print(
    f"Window end: "
    f"{min_window.index.max().strftime('%Y-%m-%d')}"
)

print(
    f"Number of observations: "
    f"{len(min_window)}"
)

print(
    f"Verified window correlation: "
    f"{min_window_corr:.3f}"
)


print("\nMAXIMUM CORRELATION")
print("-" * 40)

print(
    f"Rolling correlation: "
    f"{max_corr_value:.3f}"
)

print(
    f"Date of maximum: "
    f"{max_corr_date.strftime('%Y-%m-%d')}"
)

print(
    f"Window start: "
    f"{max_window.index.min().strftime('%Y-%m-%d')}"
)

print(
    f"Window end: "
    f"{max_window.index.max().strftime('%Y-%m-%d')}"
)

print(
    f"Number of observations: "
    f"{len(max_window)}"
)

print(
    f"Verified window correlation: "
    f"{max_window_corr:.3f}"
)


# ------------------------------------------------------------
# 13.4 Largest movements in minimum-correlation window
# ------------------------------------------------------------

min_window["abs_dVIX"] = min_window["dVIX"].abs()
min_window["abs_dVVIX"] = min_window["dVVIX"].abs()

largest_min_vix = (
    min_window
    .sort_values("abs_dVIX", ascending=False)
    .head(10)
)

largest_min_vvix = (
    min_window
    .sort_values("abs_dVVIX", ascending=False)
    .head(10)
)

print("\nLargest VIX movements in minimum-correlation window")
print("=" * 65)

print(
    largest_min_vix[
        [
            "VIX",
            "VVIX",
            "dVIX",
            "dVVIX"
        ]
    ].round(3).to_string()
)

print("\nLargest VVIX movements in minimum-correlation window")
print("=" * 65)

print(
    largest_min_vvix[
        [
            "VIX",
            "VVIX",
            "dVIX",
            "dVVIX"
        ]
    ].round(3).to_string()
)


# ------------------------------------------------------------
# 13.5 Largest movements in maximum-correlation window
# ------------------------------------------------------------

max_window["abs_dVIX"] = max_window["dVIX"].abs()
max_window["abs_dVVIX"] = max_window["dVVIX"].abs()

largest_max_vix = (
    max_window
    .sort_values("abs_dVIX", ascending=False)
    .head(10)
)

largest_max_vvix = (
    max_window
    .sort_values("abs_dVVIX", ascending=False)
    .head(10)
)

print("\nLargest VIX movements in maximum-correlation window")
print("=" * 65)

print(
    largest_max_vix[
        [
            "VIX",
            "VVIX",
            "dVIX",
            "dVVIX"
        ]
    ].round(3).to_string()
)

print("\nLargest VVIX movements in maximum-correlation window")
print("=" * 65)

print(
    largest_max_vvix[
        [
            "VIX",
            "VVIX",
            "dVIX",
            "dVVIX"
        ]
    ].round(3).to_string()
)


# ------------------------------------------------------------
# 13.6 Save minimum and maximum windows
# ------------------------------------------------------------

min_window[
    [
        "VIX",
        "VVIX",
        "dVIX",
        "dVVIX"
    ]
].to_excel(
    tables_folder
    / "minimum_rolling_correlation_window.xlsx"
)

max_window[
    [
        "VIX",
        "VVIX",
        "dVIX",
        "dVVIX"
    ]
].to_excel(
    tables_folder
    / "maximum_rolling_correlation_window.xlsx"
)


# ------------------------------------------------------------
# 13.7 Save summary table
# ------------------------------------------------------------

rolling_correlation_extremes = pd.DataFrame(
    {
        "Episode": [
            "Minimum",
            "Maximum"
        ],
        "Correlation": [
            min_corr_value,
            max_corr_value
        ],
        "End date": [
            min_corr_date,
            max_corr_date
        ],
        "Window start": [
            min_window.index.min(),
            max_window.index.min()
        ],
        "Window end": [
            min_window.index.max(),
            max_window.index.max()
        ]
    }
)

rolling_correlation_extremes.to_excel(
    tables_folder
    / "rolling_correlation_extremes.xlsx",
    index=False
)

# Correlation including all 100 observations
corr_with_march13 = min_window["dVIX"].corr(
    min_window["dVVIX"]
)

# Remove March 13, 2020
min_window_without_march13 = min_window.drop(
    pd.Timestamp("2020-03-13")
)

# Recalculate correlation
corr_without_march13 = (
    min_window_without_march13["dVIX"]
    .corr(min_window_without_march13["dVVIX"])
)

print("\nSensitivity to March 13, 2020")
print("=" * 50)
print(f"With March 13:    {corr_with_march13:.3f}")
print(f"Without March 13: {corr_without_march13:.3f}")

# ============================================================
# 14. Local minima of the rolling correlation
# ============================================================

from scipy.signal import find_peaks


# Use the negative rolling correlation to identify local minima
rolling_corr = df["rolling_corr_dVIX_dVVIX"].dropna()

# Minimum distance between two minima:
# approximately 60 trading days
min_distance = 60

minima_positions, _ = find_peaks(
    -rolling_corr.values,
    distance=min_distance
)

local_minima = rolling_corr.iloc[minima_positions]


# ------------------------------------------------------------
# Select the lowest local minima
# ------------------------------------------------------------

lowest_local_minima = (
    local_minima
    .sort_values()
    .head(10)
)


# ------------------------------------------------------------
# Create summary table
# ------------------------------------------------------------

local_minima_table = pd.DataFrame(
    {
        "Date": lowest_local_minima.index,
        "Rolling correlation": lowest_local_minima.values
    }
)

local_minima_table["Rolling correlation"] = (
    local_minima_table["Rolling correlation"].round(3)
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\nLowest local minima of the 100-day rolling correlation")
print("=" * 60)

print(
    local_minima_table.to_string(
        index=False
    )
)

# ============================================================
# 14. Troughs of negative spikes in rolling correlation
# ============================================================

from scipy.signal import find_peaks


# ------------------------------------------------------------
# 14.1 Prepare rolling-correlation series
# ------------------------------------------------------------

rolling_corr = (
    df["rolling_corr_dVIX_dVVIX"]
    .dropna()
)


# ------------------------------------------------------------
# 14.2 Identify troughs of negative spikes
# ------------------------------------------------------------

# We apply find_peaks to the negative correlation series.
# Therefore, peaks in -rolling_corr correspond to troughs
# in the original rolling-correlation series.

# prominence:
# minimum depth of the spike relative to its surroundings
#
# distance:
# minimum number of trading days between two distinct troughs

min_prominence = 0.04
min_distance = 30

trough_positions, properties = find_peaks(
    -rolling_corr.values,
    prominence=min_prominence,
    distance=min_distance
)


# Dates and correlation values at the troughs
trough_dates = rolling_corr.index[trough_positions]
trough_values = rolling_corr.iloc[trough_positions]


# ------------------------------------------------------------
# 14.3 Create summary table
# ------------------------------------------------------------

correlation_troughs = pd.DataFrame(
    {
        "Date": trough_dates,
        "Rolling correlation": trough_values.values,
        "Prominence": properties["prominences"]
    }
)

# Sort chronologically
correlation_troughs = (
    correlation_troughs
    .sort_values("Date")
    .reset_index(drop=True)
)

correlation_troughs["Rolling correlation"] = (
    correlation_troughs["Rolling correlation"]
    .round(3)
)

correlation_troughs["Prominence"] = (
    correlation_troughs["Prominence"]
    .round(3)
)


# ------------------------------------------------------------
# 14.4 Print troughs
# ------------------------------------------------------------

print("\nTroughs of negative spikes in rolling correlation")
print("=" * 70)

print(
    correlation_troughs.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 14.5 Save trough table
# ------------------------------------------------------------

correlation_troughs.to_excel(
    tables_folder / "rolling_correlation_spike_troughs.xlsx",
    index=False
)


# ============================================================
# 15. Plot rolling correlation with identified troughs
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)

# Rolling correlation
ax.plot(
    df.index,
    df["rolling_corr_dVIX_dVVIX"],
    color="black",
    linewidth=1.5
)

# Mark the exact trough of every detected negative spike
ax.scatter(
    trough_dates,
    trough_values,
    s=40,
    marker="o",
    color="red",
    zorder=5,
    label="Spike trough"
)

ax.set_title(
    "100-Day Rolling Correlation between VIX and VVIX First Differences",
    fontsize=14
)

ax.set_xlabel("Date")
ax.set_ylabel("Correlation")

ax.set_ylim(0.3, 1.0)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)

ax.legend()

fig.tight_layout()


# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------

spike_figure_path = (
    figures_folder
    / "rolling_correlation_with_spike_troughs.png"
)

fig.savefig(
    spike_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 16. Directional and contribution analysis of correlation troughs
# ============================================================

# This section investigates WHY the 100-day rolling correlation
# reaches a local trough.
#
# For each identified trough:
# 1. Extract the corresponding 100-day window
# 2. Classify daily VIX/VVIX movements by direction
# 3. Count same-direction and opposite-direction observations
# 4. Standardize dVIX and dVVIX within the window
# 5. Compute each day's contribution to Pearson correlation
# 6. Perform a leave-one-out analysis
# 7. Save summary tables for all troughs


# ============================================================
# 16.1 Containers for results
# ============================================================

directional_results = []
all_contributions = []


# ============================================================
# 16.2 Loop over all identified troughs
# ============================================================

for trough_date in trough_dates:

    trough_position = df.index.get_loc(trough_date)

    # A complete rolling window must be available
    if trough_position < rolling_window - 1:
        continue

    # --------------------------------------------------------
    # Extract the 100-day window ending at the trough
    # --------------------------------------------------------

    trough_window = df.iloc[
        trough_position - rolling_window + 1:
        trough_position + 1
    ][
        [
            "dVIX",
            "dVVIX"
        ]
    ].dropna().copy()

    n_obs = len(trough_window)

    if n_obs < 2:
        continue


    # ========================================================
    # 16.3 Directional classification
    # ========================================================

    conditions = [
        (
            (trough_window["dVIX"] > 0)
            & (trough_window["dVVIX"] > 0)
        ),
        (
            (trough_window["dVIX"] < 0)
            & (trough_window["dVVIX"] < 0)
        ),
        (
            (trough_window["dVIX"] > 0)
            & (trough_window["dVVIX"] < 0)
        ),
        (
            (trough_window["dVIX"] < 0)
            & (trough_window["dVVIX"] > 0)
        )
    ]

    labels = [
        "Both positive",
        "Both negative",
        "VIX positive / VVIX negative",
        "VIX negative / VVIX positive"
    ]

    trough_window["Direction"] = np.select(
        conditions,
        labels,
        default="Zero movement"
    )


    # --------------------------------------------------------
    # Same-direction versus opposite-direction indicator
    #
    # pandas nullable Boolean dtype is used because standard
    # bool columns cannot contain missing values.
    # --------------------------------------------------------

    trough_window["Opposite_direction"] = (
        np.sign(trough_window["dVIX"])
        != np.sign(trough_window["dVVIX"])
    ).astype("boolean")


    # If one of the two innovations equals zero,
    # the observation is neither same-direction nor opposite.
    zero_mask = (
        (trough_window["dVIX"] == 0)
        | (trough_window["dVVIX"] == 0)
    )

    trough_window.loc[
        zero_mask,
        "Opposite_direction"
    ] = pd.NA


    # ========================================================
    # 16.4 Directional counts
    # ========================================================

    n_both_positive = int(
        trough_window["Direction"]
        .eq("Both positive")
        .sum()
    )

    n_both_negative = int(
        trough_window["Direction"]
        .eq("Both negative")
        .sum()
    )

    n_vix_pos_vvix_neg = int(
        trough_window["Direction"]
        .eq("VIX positive / VVIX negative")
        .sum()
    )

    n_vix_neg_vvix_pos = int(
        trough_window["Direction"]
        .eq("VIX negative / VVIX positive")
        .sum()
    )

    n_opposite = int(
        trough_window["Opposite_direction"]
        .fillna(False)
        .sum()
    )

    n_same = int(
        trough_window["Opposite_direction"]
        .eq(False)
        .fillna(False)
        .sum()
    )

    n_zero = int(
        trough_window["Opposite_direction"]
        .isna()
        .sum()
    )

    n_directional = (
        n_same
        + n_opposite
    )

    opposite_share = (
        n_opposite
        / n_directional
        * 100
        if n_directional > 0
        else np.nan
    )


    # ========================================================
    # 16.5 Actual correlation within the trough window
    # ========================================================

    full_corr = (
        trough_window["dVIX"]
        .corr(trough_window["dVVIX"])
    )


    # ========================================================
    # 16.6 Standardized innovations
    # ========================================================

    vix_std = trough_window["dVIX"].std(
        ddof=1
    )

    vvix_std = trough_window["dVVIX"].std(
        ddof=1
    )

    # Avoid division by zero in pathological cases
    if vix_std == 0 or vvix_std == 0:
        continue


    trough_window["z_dVIX"] = (
        trough_window["dVIX"]
        - trough_window["dVIX"].mean()
    ) / vix_std

    trough_window["z_dVVIX"] = (
        trough_window["dVVIX"]
        - trough_window["dVVIX"].mean()
    ) / vvix_std


    # ========================================================
    # 16.7 Contribution of each observation to correlation
    # ========================================================
    #
    # Pearson correlation:
    #
    # rho = sum(z_x * z_y) / (n - 1)
    #
    # Therefore each daily observation contributes:
    #
    # contribution_t = z_x,t * z_y,t / (n - 1)
    #
    # Negative values reduce the correlation.
    # Positive values increase the correlation.
    # ========================================================

    trough_window[
        "Correlation contribution"
    ] = (
        trough_window["z_dVIX"]
        * trough_window["z_dVVIX"]
    ) / (n_obs - 1)


    # ========================================================
    # 16.8 Check that contributions reproduce correlation
    # ========================================================

    sum_contributions = (
        trough_window[
            "Correlation contribution"
        ]
        .sum()
    )


    # ========================================================
    # 16.9 Leave-one-out analysis
    # ========================================================
    #
    # Remove each observation individually and recalculate
    # the correlation.
    #
    # If correlation rises considerably after removing day t,
    # that day was pushing the correlation downward.
    # ========================================================

    leave_one_out_corr = []

    for observation_date in trough_window.index:

        temp_window = trough_window.drop(
            index=observation_date
        )

        temp_corr = (
            temp_window["dVIX"]
            .corr(temp_window["dVVIX"])
        )

        leave_one_out_corr.append(
            temp_corr
        )


    trough_window[
        "Correlation without observation"
    ] = leave_one_out_corr


    trough_window[
        "Correlation increase if removed"
    ] = (
        trough_window[
            "Correlation without observation"
        ]
        - full_corr
    )


    # ========================================================
    # 16.10 Additional magnitude information
    # ========================================================
    #
    # This is useful to identify cases where VVIX reacts much
    # more strongly than VIX in standardized terms.
    # ========================================================

    trough_window[
        "Absolute z_dVIX"
    ] = (
        trough_window["z_dVIX"]
        .abs()
    )

    trough_window[
        "Absolute z_dVVIX"
    ] = (
        trough_window["z_dVVIX"]
        .abs()
    )

    trough_window[
        "VVIX relative standardized response"
    ] = (
        trough_window["Absolute z_dVVIX"]
        - trough_window["Absolute z_dVIX"]
    )


    # ========================================================
    # 16.11 Add identifying information
    # ========================================================

    trough_window[
        "Trough date"
    ] = trough_date

    trough_window[
        "Observation date"
    ] = trough_window.index

    trough_window[
        "Rolling correlation at trough"
    ] = full_corr


    # ========================================================
    # 16.12 Store directional summary
    # ========================================================

    directional_results.append(
        {
            "Trough date":
                trough_date,

            "Rolling correlation":
                full_corr,

            "Sum of daily contributions":
                sum_contributions,

            "N observations":
                n_obs,

            "Both positive":
                n_both_positive,

            "Both negative":
                n_both_negative,

            "VIX positive / VVIX negative":
                n_vix_pos_vvix_neg,

            "VIX negative / VVIX positive":
                n_vix_neg_vvix_pos,

            "Same-direction days":
                n_same,

            "Opposite-direction days":
                n_opposite,

            "Zero-movement days":
                n_zero,

            "Opposite-direction share (%)":
                opposite_share
        }
    )


    # Store the complete observation-level window
    all_contributions.append(
        trough_window
    )


# ============================================================
# 17. Summary across all troughs
# ============================================================

directional_summary = pd.DataFrame(
    directional_results
)


# ------------------------------------------------------------
# Round values
# ------------------------------------------------------------

directional_summary[
    "Rolling correlation"
] = (
    directional_summary[
        "Rolling correlation"
    ]
    .round(4)
)

directional_summary[
    "Sum of daily contributions"
] = (
    directional_summary[
        "Sum of daily contributions"
    ]
    .round(4)
)

directional_summary[
    "Opposite-direction share (%)"
] = (
    directional_summary[
        "Opposite-direction share (%)"
    ]
    .round(2)
)


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

print(
    "\nDirectional analysis of "
    "correlation-trough windows"
)

print("=" * 120)

print(
    directional_summary.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------

directional_summary.to_excel(
    tables_folder
    / "correlation_trough_directional_analysis.xlsx",
    index=False
)


# ============================================================
# 18. Combine all observation-level contributions
# ============================================================

all_contributions = pd.concat(
    all_contributions,
    axis=0,
    ignore_index=True
)


# ============================================================
# 19. Five strongest negative contributors for each trough
# ============================================================

top_negative_contributors = (
    all_contributions
    .sort_values(
        [
            "Trough date",
            "Correlation contribution"
        ],
        ascending=[
            True,
            True
        ]
    )
    .groupby(
        "Trough date",
        group_keys=False
    )
    .head(5)
    .copy()
)


top_negative_contributors = (
    top_negative_contributors[
        [
            "Trough date",
            "Observation date",
            "Rolling correlation at trough",
            "dVIX",
            "dVVIX",
            "Direction",
            "z_dVIX",
            "z_dVVIX",
            "Correlation contribution",
            "Correlation without observation",
            "Correlation increase if removed",
            "VVIX relative standardized response"
        ]
    ]
)


# ------------------------------------------------------------
# Round numerical columns
# ------------------------------------------------------------

columns_to_round = [
    "Rolling correlation at trough",
    "dVIX",
    "dVVIX",
    "z_dVIX",
    "z_dVVIX",
    "Correlation contribution",
    "Correlation without observation",
    "Correlation increase if removed",
    "VVIX relative standardized response"
]

top_negative_contributors[
    columns_to_round
] = (
    top_negative_contributors[
        columns_to_round
    ]
    .round(4)
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print(
    "\nFive strongest negative contributors "
    "for each correlation trough"
)

print("=" * 150)

print(
    top_negative_contributors
    .to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

top_negative_contributors.to_excel(
    tables_folder
    / "top_negative_contributors_by_trough.xlsx",
    index=False
)


# ============================================================
# 20. Observation with strongest negative contribution
#     for each trough
# ============================================================

main_negative_contributor = (
    all_contributions
    .sort_values(
        [
            "Trough date",
            "Correlation contribution"
        ]
    )
    .groupby(
        "Trough date",
        as_index=False
    )
    .first()
)


main_negative_contributor = (
    main_negative_contributor[
        [
            "Trough date",
            "Observation date",
            "Rolling correlation at trough",
            "dVIX",
            "dVVIX",
            "Direction",
            "z_dVIX",
            "z_dVVIX",
            "Correlation contribution",
            "Correlation increase if removed",
            "VVIX relative standardized response"
        ]
    ]
)


# ------------------------------------------------------------
# Round only numerical columns that actually exist
# ------------------------------------------------------------

main_round_columns = [
    col
    for col in columns_to_round
    if col in main_negative_contributor.columns
]

main_negative_contributor[
    main_round_columns
] = (
    main_negative_contributor[
        main_round_columns
    ]
    .round(4)
)


print(
    "\nMain negative contributor "
    "for each correlation trough"
)

print("=" * 140)

print(
    main_negative_contributor
    .to_string(
        index=False
    )
)


main_negative_contributor.to_excel(
    tables_folder
    / "main_negative_contributor_by_trough.xlsx",
    index=False
)

# ============================================================
# 21. Aggregate classification of main negative contributors
# ============================================================

direction_counts = (
    main_negative_contributor[
        "Direction"
    ]
    .value_counts()
    .rename_axis(
        "Direction"
    )
    .reset_index(
        name="Number of troughs"
    )
)


direction_counts[
    "Share (%)"
] = (
    direction_counts[
        "Number of troughs"
    ]
    / direction_counts[
        "Number of troughs"
    ].sum()
    * 100
).round(2)


print(
    "\nDirection of the main negative contributor "
    "across correlation troughs"
)

print("=" * 80)

print(
    direction_counts
    .to_string(
        index=False
    )
)


direction_counts.to_excel(
    tables_folder
    / "main_negative_contributor_direction_summary.xlsx",
    index=False
)


# ============================================================
# 22. Largest leave-one-out influence for each trough
# ============================================================
#
# This identifies the observation whose removal produces
# the largest increase in correlation.
# ============================================================

largest_leave_one_out_effect = (
    all_contributions
    .sort_values(
        [
            "Trough date",
            "Correlation increase if removed"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby(
        "Trough date",
        as_index=False
    )
    .first()
)


largest_leave_one_out_effect = (
    largest_leave_one_out_effect[
        [
            "Trough date",
            "Observation date",
            "Rolling correlation at trough",
            "dVIX",
            "dVVIX",
            "Direction",
            "Correlation contribution",
            "Correlation without observation",
            "Correlation increase if removed"
        ]
    ]
)


leave_one_out_round_columns = [
    "Rolling correlation at trough",
    "dVIX",
    "dVVIX",
    "Correlation contribution",
    "Correlation without observation",
    "Correlation increase if removed"
]

largest_leave_one_out_effect[
    leave_one_out_round_columns
] = (
    largest_leave_one_out_effect[
        leave_one_out_round_columns
    ]
    .round(4)
)


print(
    "\nObservation with largest leave-one-out "
    "effect for each trough"
)

print("=" * 140)

print(
    largest_leave_one_out_effect
    .to_string(
        index=False
    )
)


largest_leave_one_out_effect.to_excel(
    tables_folder
    / "largest_leave_one_out_effect_by_trough.xlsx",
    index=False
)


# ============================================================
# 23. Strongest standardized VVIX amplification
#     within each trough window
# ============================================================
#
# Positive values mean that VVIX moved more strongly than VIX
# relative to its own historical variability within the window.
# ============================================================

largest_vvix_relative_response = (
    all_contributions
    .sort_values(
        [
            "Trough date",
            "VVIX relative standardized response"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby(
        "Trough date",
        as_index=False
    )
    .first()
)


largest_vvix_relative_response = (
    largest_vvix_relative_response[
        [
            "Trough date",
            "Observation date",
            "Rolling correlation at trough",
            "dVIX",
            "dVVIX",
            "Direction",
            "z_dVIX",
            "z_dVVIX",
            "VVIX relative standardized response",
            "Correlation contribution"
        ]
    ]
)


vvix_response_round_columns = [
    "Rolling correlation at trough",
    "dVIX",
    "dVVIX",
    "z_dVIX",
    "z_dVVIX",
    "VVIX relative standardized response",
    "Correlation contribution"
]

largest_vvix_relative_response[
    vvix_response_round_columns
] = (
    largest_vvix_relative_response[
        vvix_response_round_columns
    ]
    .round(4)
)


print(
    "\nLargest relative VVIX standardized response "
    "within each trough window"
)

print("=" * 140)

print(
    largest_vvix_relative_response
    .to_string(
        index=False
    )
)


largest_vvix_relative_response.to_excel(
    tables_folder
    / "largest_vvix_relative_response_by_trough.xlsx",
    index=False
)


# ============================================================
# 24. Save complete observation-level dataset
# ============================================================

all_contributions.to_excel(
    tables_folder
    / "all_correlation_trough_contributions.xlsx",
    index=False
)


# ============================================================
# 25. Final confirmation
# ============================================================

print(
    "\nCorrelation-trough analysis completed successfully"
)

print("=" * 70)

print("\nTables saved:")

print(
    tables_folder
    / "correlation_trough_directional_analysis.xlsx"
)

print(
    tables_folder
    / "top_negative_contributors_by_trough.xlsx"
)

print(
    tables_folder
    / "main_negative_contributor_by_trough.xlsx"
)

print(
    tables_folder
    / "main_negative_contributor_direction_summary.xlsx"
)

print(
    tables_folder
    / "largest_leave_one_out_effect_by_trough.xlsx"
)

print(
    tables_folder
    / "largest_vvix_relative_response_by_trough.xlsx"
)

print(
    tables_folder
    / "all_correlation_trough_contributions.xlsx"
)

