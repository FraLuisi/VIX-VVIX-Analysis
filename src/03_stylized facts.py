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
# Used for the autocorrelation and stationarity analysis
# following Fernandes, Medeiros and Scharth (2014)
df["logVIX"] = np.log(df["VIX"])
df["logVVIX"] = np.log(df["VVIX"])


# ============================================================
# 5. Plot colors
# ============================================================

vix_color = "#2F6FB0"
vvix_color = "#4B2E83"


# ============================================================
# 6. Basic sample check
# ============================================================

print("\nSTYLIZED FACTS DATASET\n")

print(f"Start date:   {df.index.min().date()}")
print(f"End date:     {df.index.max().date()}")
print(f"Observations: {len(df)}")

print("\nMissing values after transformations:")

print(
    df[
        [
            "VIX",
            "VVIX",
            "logVIX",
            "logVVIX",
            "dVIX",
            "dVVIX",
            "rVIX",
            "rVVIX"
        ]
    ].isna().sum()
)


# ============================================================
# 7. Autocorrelation analysis
# ============================================================

max_lags = 500


# ============================================================
# 8. Figure: ACF and PACF of log(VIX) and log(VVIX)
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(14, 9)
)


# ------------------------------------------------------------
# 8.1 ACF of log(VIX)
# ------------------------------------------------------------

plot_acf(
    df["logVIX"].dropna(),
    lags=max_lags,
    alpha=0.05,
    zero=True,
    fft=True,
    ax=axes[0, 0]
)

axes[0, 0].set_title(
    "ACF of log(VIX)",
    fontsize=12,
    fontweight="bold"
)

axes[0, 0].set_xlabel("Lag (Trading Days)")
axes[0, 0].set_ylabel("Autocorrelation")


# ------------------------------------------------------------
# 8.2 PACF of log(VIX)
# ------------------------------------------------------------

plot_pacf(
    df["logVIX"].dropna(),
    lags=max_lags,
    alpha=0.05,
    zero=True,
    method="ywm",
    ax=axes[0, 1]
)

axes[0, 1].set_title(
    "PACF of log(VIX)",
    fontsize=12,
    fontweight="bold"
)

axes[0, 1].set_xlabel("Lag (Trading Days)")
axes[0, 1].set_ylabel("Partial Autocorrelation")


# ------------------------------------------------------------
# 8.3 ACF of log(VVIX)
# ------------------------------------------------------------

plot_acf(
    df["logVVIX"].dropna(),
    lags=max_lags,
    alpha=0.05,
    zero=True,
    fft=True,
    ax=axes[1, 0]
)

axes[1, 0].set_title(
    "ACF of log(VVIX)",
    fontsize=12,
    fontweight="bold"
)

axes[1, 0].set_xlabel("Lag (Trading Days)")
axes[1, 0].set_ylabel("Autocorrelation")


# ------------------------------------------------------------
# 8.4 PACF of log(VVIX)
# ------------------------------------------------------------

plot_pacf(
    df["logVVIX"].dropna(),
    lags=max_lags,
    alpha=0.05,
    zero=True,
    method="ywm",
    ax=axes[1, 1]
)

axes[1, 1].set_title(
    "PACF of log(VVIX)",
    fontsize=12,
    fontweight="bold"
)

axes[1, 1].set_xlabel("Lag (Trading Days)")
axes[1, 1].set_ylabel("Partial Autocorrelation")


# ------------------------------------------------------------
# 8.5 Figure formatting
# ------------------------------------------------------------

for ax in axes.flat:

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.30
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        labelsize=9
    )


fig.suptitle(
    "Autocorrelation Structure of the VIX and VVIX",
    fontsize=15,
    fontweight="bold",
    y=0.99
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)


acf_pacf_figure_path = (
    figures_folder / "acf_pacf_log_vix_vvix.png"
)

fig.savefig(
    acf_pacf_figure_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)


# ============================================================
# 9. Selected autocorrelation coefficients
# ============================================================

selected_lags = [
    1,
    5,
    21,
    63,
    125,
    250,
    500
]


acf_vix = acf(
    df["logVIX"].dropna(),
    nlags=max_lags,
    fft=True
)

acf_vvix = acf(
    df["logVVIX"].dropna(),
    nlags=max_lags,
    fft=True
)


# ============================================================
# 10. Autocorrelation table
# ============================================================

autocorrelation_table = pd.DataFrame(
    {
        "Lag": selected_lags,
        "VIX": [
            acf_vix[lag]
            for lag in selected_lags
        ],
        "VVIX": [
            acf_vvix[lag]
            for lag in selected_lags
        ]
    }
)

autocorrelation_table = (
    autocorrelation_table
    .set_index("Lag")
    .round(4)
)


# ============================================================
# 11. Print autocorrelation table
# ============================================================

print("\nSELECTED AUTOCORRELATIONS\n")

print(autocorrelation_table)


# ============================================================
# 12. Save autocorrelation table
# ============================================================

acf_csv_path = (
    tables_folder / "selected_autocorrelations.csv"
)

acf_excel_path = (
    tables_folder / "selected_autocorrelations.xlsx"
)

autocorrelation_table.to_csv(
    acf_csv_path
)

autocorrelation_table.to_excel(
    acf_excel_path
)


# ============================================================
# 13. Stationarity tests: ADF and Phillips-Perron
# ============================================================

def run_stationarity_tests(series: pd.Series) -> dict:
    """
    Run ADF and Phillips-Perron unit-root tests.

    Null hypothesis for both tests:
        H0 = the series contains a unit root.

    ADF lag length is selected using BIC,
    following Fernandes et al. (2014).
    """

    clean_series = series.dropna().astype(float)

    # --------------------------------------------------------
    # 13.1 Augmented Dickey-Fuller
    # --------------------------------------------------------

    adf_test = ADF(
        clean_series,
        trend="c",
        method="bic"
    )

    # --------------------------------------------------------
    # 13.2 Phillips-Perron
    # --------------------------------------------------------

    pp_test = PhillipsPerron(
        clean_series,
        trend="c",
        test_type="tau"
    )

    return {
        "ADF Statistic": adf_test.stat,
        "ADF p-value": adf_test.pvalue,
        "ADF Lags": adf_test.lags,

        "PP Statistic": pp_test.stat,
        "PP p-value": pp_test.pvalue,
        "PP Lags": pp_test.lags
    }


# ============================================================
# 14. Run stationarity tests
# ============================================================

vix_tests = run_stationarity_tests(
    df["logVIX"]
)

vvix_tests = run_stationarity_tests(
    df["logVVIX"]
)


# ============================================================
# 15. Stationarity test results table
# ============================================================

stationarity_results = pd.DataFrame(
    {
        "VIX": [
            vix_tests["ADF Statistic"],
            vix_tests["ADF p-value"],
            vix_tests["PP Statistic"],
            vix_tests["PP p-value"]
        ],

        "VVIX": [
            vvix_tests["ADF Statistic"],
            vvix_tests["ADF p-value"],
            vvix_tests["PP Statistic"],
            vvix_tests["PP p-value"]
        ]
    },

    index=[
        "ADF Statistic",
        "ADF p-value",
        "PP Statistic",
        "PP p-value"
    ]
)

stationarity_results_rounded = (
    stationarity_results.round(4)
)


# ============================================================
# 16. Print stationarity test results
# ============================================================

print(
    "\n"
    "==============================================="
)

print(
    "STATIONARITY TESTS"
)

print(
    "===============================================\n"
)

print(
    stationarity_results_rounded
)


print("\nADF lag selection (BIC):")

print(
    f"VIX:  {vix_tests['ADF Lags']}"
)

print(
    f"VVIX: {vvix_tests['ADF Lags']}"
)


print("\nPhillips-Perron bandwidth/lags:")

print(
    f"VIX:  {vix_tests['PP Lags']}"
)

print(
    f"VVIX: {vvix_tests['PP Lags']}"
)


# ============================================================
# 17. Save stationarity test results
# ============================================================

stationarity_csv_path = (
    tables_folder
    / "stationarity_tests.csv"
)

stationarity_excel_path = (
    tables_folder
    / "stationarity_tests.xlsx"
)

stationarity_results_rounded.to_csv(
    stationarity_csv_path
)

stationarity_results_rounded.to_excel(
    stationarity_excel_path
)


# ============================================================
# 18. Final output
# ============================================================

print(
    "\n"
    "==============================================="
)

print(
    "FILES CREATED SUCCESSFULLY"
)

print(
    "===============================================\n"
)

print(
    f"ACF/PACF figure: "
    f"{acf_pacf_figure_path}"
)

print(
    f"Autocorrelation CSV table: "
    f"{acf_csv_path}"
)

print(
    f"Autocorrelation Excel table: "
    f"{acf_excel_path}"
)

print(
    f"Stationarity CSV table: "
    f"{stationarity_csv_path}"
)

print(
    f"Stationarity Excel table: "
    f"{stationarity_excel_path}"
)

# ============================================================
# 19. Mean reversion: AR(1) and half-life
# ============================================================

def estimate_ar1_half_life(series: pd.Series) -> dict:
    """
    Estimate a simple AR(1) model:

        X_t = c + phi * X_{t-1} + epsilon_t

    and compute the half-life of mean reversion:

        HL = ln(0.5) / ln(phi)

    The half-life is expressed in trading days.

    Notes
    -----
    - 0 < phi < 1 implies mean reversion.
    - A phi closer to 1 implies slower mean reversion.
    - The long-run mean implied by the AR(1) model is:
          mu = c / (1 - phi)
    """

    clean_series = series.dropna().astype(float)

    ar_data = pd.DataFrame({
        "current": clean_series,
        "lagged": clean_series.shift(1)
    }).dropna()

    y = ar_data["current"]

    X = sm.add_constant(
        ar_data["lagged"]
    )

    model = sm.OLS(
        y,
        X
    ).fit()

    constant = model.params["const"]
    phi = model.params["lagged"]

    phi_pvalue = model.pvalues["lagged"]

    # --------------------------------------------------------
    # Half-life
    # --------------------------------------------------------

    if 0 < phi < 1:

        half_life = (
            np.log(0.5)
            / np.log(phi)
        )

        long_run_mean = (
            constant
            / (1 - phi)
        )

    else:

        half_life = np.nan
        long_run_mean = np.nan

    return {
        "Constant": constant,
        "AR(1) coefficient": phi,
        "AR(1) p-value": phi_pvalue,
        "Half-life": half_life,
        "Long-run mean": long_run_mean,
        "R-squared": model.rsquared
    }


# ============================================================
# 20. Estimate AR(1) models
# ============================================================

vix_ar1 = estimate_ar1_half_life(
    df["VIX"]
)

vvix_ar1 = estimate_ar1_half_life(
    df["VVIX"]
)


# ============================================================
# 21. Mean-reversion results table
# ============================================================

mean_reversion_results = pd.DataFrame(
    {
        "VIX": [
            vix_ar1["AR(1) coefficient"],
            vix_ar1["AR(1) p-value"],
            vix_ar1["Half-life"],
            vix_ar1["Long-run mean"],
            vix_ar1["R-squared"]
        ],

        "VVIX": [
            vvix_ar1["AR(1) coefficient"],
            vvix_ar1["AR(1) p-value"],
            vvix_ar1["Half-life"],
            vvix_ar1["Long-run mean"],
            vvix_ar1["R-squared"]
        ]
    },

    index=[
        "AR(1) coefficient",
        "AR(1) p-value",
        "Half-life (trading days)",
        "Implied long-run mean",
        "R-squared"
    ]
)

mean_reversion_results_rounded = (
    mean_reversion_results.round(4)
)


# ============================================================
# 22. Print mean-reversion results
# ============================================================

print(
    "\n"
    "==============================================="
)

print(
    "MEAN REVERSION AND HALF-LIFE"
)

print(
    "===============================================\n"
)

print(
    mean_reversion_results_rounded
)

print(
    "\nInterpretation:"
)

print(
    f"VIX half-life: "
    f"{vix_ar1['Half-life']:.2f} trading days"
)

print(
    f"VVIX half-life: "
    f"{vvix_ar1['Half-life']:.2f} trading days"
)


# ============================================================
# 23. Save mean-reversion results
# ============================================================

mean_reversion_csv_path = (
    tables_folder
    / "mean_reversion_half_life.csv"
)

mean_reversion_excel_path = (
    tables_folder
    / "mean_reversion_half_life.xlsx"
)

mean_reversion_results_rounded.to_csv(
    mean_reversion_csv_path
)

mean_reversion_results_rounded.to_excel(
    mean_reversion_excel_path
)


# ============================================================
# 24. Percentile-based mean-reversion paths
# ============================================================

forecast_horizon = 300


def percentile_mean_reversion_paths(
    series: pd.Series,
    horizon: int = 300
):
    """
    Construct average subsequent index paths following
    extreme initial observations.

    Percentile groups follow the structure used in the
    mean-reversion figure of Albers (2023):

        99%-100%
        95%-99%
        90%-95%
        5%-10%
        0%-5%

    For every observation belonging to one of these groups,
    the subsequent path from t to t+horizon is collected.

    Only starting observations with a complete future
    horizon are retained.
    """

    clean_series = series.dropna().astype(float)

    # --------------------------------------------------------
    # Percentile thresholds
    # --------------------------------------------------------

    q00 = clean_series.quantile(0.00)
    q05 = clean_series.quantile(0.05)
    q10 = clean_series.quantile(0.10)

    q90 = clean_series.quantile(0.90)
    q95 = clean_series.quantile(0.95)
    q99 = clean_series.quantile(0.99)
    q100 = clean_series.quantile(1.00)


    percentile_groups = {
        "99%-100%": (
            q99,
            q100,
            True
        ),

        "95%-99%": (
            q95,
            q99,
            False
        ),

        "90%-95%": (
            q90,
            q95,
            False
        ),

        "5%-10%": (
            q05,
            q10,
            False
        ),

        "0%-5%": (
            q00,
            q05,
            True
        )
    }


    average_paths = {}
    observation_counts = {}


    # --------------------------------------------------------
    # Construct average future path for each percentile group
    # --------------------------------------------------------

    for group_name, (
        lower,
        upper,
        include_boundary
    ) in percentile_groups.items():

        paths = []

        for i in range(
            len(clean_series) - horizon
        ):

            initial_value = clean_series.iloc[i]

            if group_name == "99%-100%":

                belongs_to_group = (
                    initial_value >= lower
                    and
                    initial_value <= upper
                )

            elif group_name == "0%-5%":

                belongs_to_group = (
                    initial_value >= lower
                    and
                    initial_value <= upper
                )

            else:

                belongs_to_group = (
                    initial_value >= lower
                    and
                    initial_value < upper
                )


            if belongs_to_group:

                future_path = (
                    clean_series
                    .iloc[
                        i:
                        i + horizon + 1
                    ]
                    .to_numpy()
                )

                if len(future_path) == horizon + 1:

                    paths.append(
                        future_path
                    )


        observation_counts[group_name] = len(paths)


        if len(paths) > 0:

            paths_array = np.vstack(
                paths
            )

            average_paths[group_name] = (
                paths_array.mean(axis=0)
            )

        else:

            average_paths[group_name] = (
                np.full(
                    horizon + 1,
                    np.nan
                )
            )


    return (
        average_paths,
        observation_counts,
        clean_series.mean()
    )


# ============================================================
# 25. Construct VIX and VVIX mean-reversion paths
# ============================================================

(
    vix_paths,
    vix_counts,
    vix_mean
) = percentile_mean_reversion_paths(
    df["VIX"],
    horizon=forecast_horizon
)


(
    vvix_paths,
    vvix_counts,
    vvix_mean
) = percentile_mean_reversion_paths(
    df["VVIX"],
    horizon=forecast_horizon
)


# ============================================================
# 26. Figure: Mean reversion after extreme observations
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(14, 6),
    sharex=True
)

horizons = np.arange(
    forecast_horizon + 1
)


# ------------------------------------------------------------
# 26.1 VIX
# ------------------------------------------------------------

for group_name, path in vix_paths.items():

    axes[0].plot(
        horizons,
        path,
        linewidth=1.5,
        label=group_name
    )


axes[0].axhline(
    y=vix_mean,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Full-sample mean"
)

axes[0].set_title(
    "Mean Reversion of the VIX",
    fontsize=13,
    fontweight="bold"
)

axes[0].set_xlabel(
    "Trading Days after Initial Observation"
)

axes[0].set_ylabel(
    "VIX Level"
)

axes[0].grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.30
)

axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)


# ------------------------------------------------------------
# 26.2 VVIX
# ------------------------------------------------------------

for group_name, path in vvix_paths.items():

    axes[1].plot(
        horizons,
        path,
        linewidth=1.5,
        label=group_name
    )


axes[1].axhline(
    y=vvix_mean,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Full-sample mean"
)

axes[1].set_title(
    "Mean Reversion of the VVIX",
    fontsize=13,
    fontweight="bold"
)

axes[1].set_xlabel(
    "Trading Days after Initial Observation"
)

axes[1].set_ylabel(
    "VVIX Level"
)

axes[1].grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.30
)

axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)


# ------------------------------------------------------------
# 26.3 Legend and formatting
# ------------------------------------------------------------

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02)
)

fig.suptitle(
    "Mean Reversion Following Extreme VIX and VVIX Levels",
    fontsize=15,
    fontweight="bold",
    y=0.98
)

fig.tight_layout(
    rect=[0, 0.08, 1, 0.95]
)


mean_reversion_figure_path = (
    figures_folder
    / "mean_reversion_percentile_paths.png"
)

fig.savefig(
    mean_reversion_figure_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)


# ============================================================
# 27. Percentile-group observation counts
# ============================================================

percentile_counts = pd.DataFrame(
    {
        "VIX": vix_counts,
        "VVIX": vvix_counts
    }
)

print(
    "\n"
    "==============================================="
)

print(
    "PERCENTILE MEAN-REVERSION SAMPLE COUNTS"
)

print(
    "===============================================\n"
)

print(
    percentile_counts
)


percentile_counts_csv_path = (
    tables_folder
    / "mean_reversion_percentile_counts.csv"
)

percentile_counts_excel_path = (
    tables_folder
    / "mean_reversion_percentile_counts.xlsx"
)

percentile_counts.to_csv(
    percentile_counts_csv_path
)

percentile_counts.to_excel(
    percentile_counts_excel_path
)


# ============================================================
# 28. Final mean-reversion output
# ============================================================

print(
    "\n"
    "==============================================="
)

print(
    "MEAN-REVERSION FILES CREATED SUCCESSFULLY"
)

print(
    "===============================================\n"
)

print(
    f"Mean-reversion figure: "
    f"{mean_reversion_figure_path}"
)

print(
    f"Half-life CSV table: "
    f"{mean_reversion_csv_path}"
)

print(
    f"Half-life Excel table: "
    f"{mean_reversion_excel_path}"
)

print(
    f"Percentile-count CSV table: "
    f"{percentile_counts_csv_path}"
)

print(
    f"Percentile-count Excel table: "
    f"{percentile_counts_excel_path}"
)

