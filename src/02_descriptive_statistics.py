from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

if (df[["VIX", "VVIX"]] <= 0).any().any():
    raise ValueError(
        "VIX and VVIX must be strictly positive "
        "to compute logarithmic returns."
    )


# ============================================================
# 3. Series transformations
# ============================================================

df["dVIX"] = df["VIX"].diff()
df["dVVIX"] = df["VVIX"].diff()

df["rVIX"] = np.log(df["VIX"]).diff()
df["rVVIX"] = np.log(df["VVIX"]).diff()


# ============================================================
# 4. Plot colors
# ============================================================

vix_color = "#2F6FB0"
vvix_color = "#4B2E83"


# ============================================================
# 5. Figure 1: VIX and VVIX levels
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 5.5)
)

ax.plot(
    df.index,
    df["VIX"],
    color=vix_color,
    linewidth=1.15,
    label="VIX"
)

ax.plot(
    df.index,
    df["VVIX"],
    color=vvix_color,
    linewidth=1.25,
    label="VVIX"
)

ax.set_title(
    "VIX and VVIX Price Series",
    fontsize=15,
    fontweight="bold",
    pad=14
)

ax.set_ylabel(
    "Index Level",
    fontsize=11
)

ax.set_xlabel("")

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

ax.tick_params(
    axis="both",
    labelsize=10
)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=False,
    fontsize=10
)

fig.tight_layout()

levels_figure_path = (
    figures_folder / "vix_vvix_levels.png"
)

fig.savefig(
    levels_figure_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)


# ============================================================
# 6. Figure 2: VIX and VVIX log returns
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(12, 7),
    sharex=True
)


# ------------------------------------------------------------
# 6.1 VIX log returns
# ------------------------------------------------------------

axes[0].plot(
    df.index,
    df["rVIX"],
    color=vix_color,
    linewidth=0.9
)

axes[0].axhline(
    y=0,
    color="black",
    linewidth=0.7,
    alpha=0.6
)

axes[0].set_title(
    "VIX Log Returns",
    fontsize=13,
    fontweight="bold",
    pad=10
)

axes[0].set_ylabel(
    "Log Return",
    fontsize=10
)

axes[0].grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)

axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[0].spines["left"].set_linewidth(0.8)
axes[0].spines["bottom"].set_linewidth(0.8)

axes[0].tick_params(
    axis="both",
    labelsize=9
)


# ------------------------------------------------------------
# 6.2 VVIX log returns
# ------------------------------------------------------------

axes[1].plot(
    df.index,
    df["rVVIX"],
    color=vvix_color,
    linewidth=0.9
)

axes[1].axhline(
    y=0,
    color="black",
    linewidth=0.7,
    alpha=0.6
)

axes[1].set_title(
    "VVIX Log Returns",
    fontsize=13,
    fontweight="bold",
    pad=10
)

axes[1].set_ylabel(
    "Log Return",
    fontsize=10
)

axes[1].set_xlabel("")

axes[1].grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)

axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

axes[1].spines["left"].set_linewidth(0.8)
axes[1].spines["bottom"].set_linewidth(0.8)

axes[1].tick_params(
    axis="both",
    labelsize=9
)


# ------------------------------------------------------------
# 6.3 Main title and layout
# ------------------------------------------------------------

fig.suptitle(
    "Daily Log Returns of the VIX and VVIX",
    fontsize=15,
    fontweight="bold",
    y=0.99
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

log_returns_figure_path = (
    figures_folder / "vix_vvix_log_returns.png"
)

fig.savefig(
    log_returns_figure_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)


# ============================================================
# 7. Figure 3: VIX and VVIX first differences
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(12, 7),
    sharex=True
)


# ------------------------------------------------------------
# 7.1 VIX first differences
# ------------------------------------------------------------

axes[0].plot(
    df.index,
    df["dVIX"],
    color=vix_color,
    linewidth=0.9
)

axes[0].axhline(
    y=0,
    color="black",
    linewidth=0.7,
    alpha=0.6
)

axes[0].set_title(
    "VIX First Differences",
    fontsize=13,
    fontweight="bold",
    pad=10
)

axes[0].set_ylabel(
    "Change in Index Points",
    fontsize=10
)

axes[0].grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)

axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[0].spines["left"].set_linewidth(0.8)
axes[0].spines["bottom"].set_linewidth(0.8)

axes[0].tick_params(
    axis="both",
    labelsize=9
)


# ------------------------------------------------------------
# 7.2 VVIX first differences
# ------------------------------------------------------------

axes[1].plot(
    df.index,
    df["dVVIX"],
    color=vvix_color,
    linewidth=0.9
)

axes[1].axhline(
    y=0,
    color="black",
    linewidth=0.7,
    alpha=0.6
)

axes[1].set_title(
    "VVIX First Differences",
    fontsize=13,
    fontweight="bold",
    pad=10
)

axes[1].set_ylabel(
    "Change in Index Points",
    fontsize=10
)

axes[1].set_xlabel("")

axes[1].grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)

axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

axes[1].spines["left"].set_linewidth(0.8)
axes[1].spines["bottom"].set_linewidth(0.8)

axes[1].tick_params(
    axis="both",
    labelsize=9
)


# ------------------------------------------------------------
# 7.3 Main title and layout
# ------------------------------------------------------------

fig.suptitle(
    "Daily First Differences of the VIX and VVIX",
    fontsize=15,
    fontweight="bold",
    y=0.99
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

first_differences_figure_path = (
    figures_folder / "vix_vvix_first_differences.png"
)

fig.savefig(
    first_differences_figure_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)


# ============================================================
# 8. Descriptive statistics function
# ============================================================

def calculate_statistics(series: pd.Series) -> dict:
    """
    Compute the main descriptive statistics for a time series.

    pandas.Series.kurt() returns excess kurtosis,
    so the benchmark for a normal distribution is zero.
    """

    clean_series = series.dropna()

    return {
        "Min.": clean_series.min(),
        "Mean": clean_series.mean(),
        "Median": clean_series.median(),
        "Max.": clean_series.max(),
        "Std.": clean_series.std(),
        "Skewness": clean_series.skew(),
        "Kurtosis": clean_series.kurt()
    }


# ============================================================
# 9. Summary statistics
# ============================================================

variables = {
    "VIX": df["VIX"],
    "VVIX": df["VVIX"],
    "r(VIX)": df["rVIX"],
    "r(VVIX)": df["rVVIX"],
    "ΔVIX": df["dVIX"],
    "ΔVVIX": df["dVVIX"]
}

summary = pd.DataFrame({
    variable_name: calculate_statistics(series)
    for variable_name, series in variables.items()
})

row_order = [
    "Min.",
    "Mean",
    "Median",
    "Max.",
    "Std.",
    "Skewness",
    "Kurtosis"
]

summary = summary.loc[row_order]

summary_rounded = summary.round(4)


# ============================================================
# 10. Print and save summary statistics
# ============================================================

print("\nSUMMARY STATISTICS\n")

print(summary_rounded)

csv_path = (
    tables_folder / "summary_statistics.csv"
)

excel_path = (
    tables_folder / "summary_statistics.xlsx"
)

summary_rounded.to_csv(
    csv_path
)

summary_rounded.to_excel(
    excel_path
)


# ============================================================
# 11. Final output
# ============================================================

print("\nFiles created successfully:")

print(
    f"Levels figure: {levels_figure_path}"
)

print(
    f"Log returns figure: {log_returns_figure_path}"
)

print(
    f"First differences figure: {first_differences_figure_path}"
)

print(
    f"CSV table: {csv_path}"
)

print(
    f"Excel table: {excel_path}"
)