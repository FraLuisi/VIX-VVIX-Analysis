# ============================================================
# VIX-VVIX GRANGER CAUSALITY ANALYSIS
#
# Full sample
#
# Step 1:
# Select the lag order of a bivariate VAR using information
# criteria (AIC, BIC, HQIC, FPE).
#
# Step 2:
# Use the BIC-selected lag order as the main specification.
#
# Step 3:
# Test Granger causality at economically relevant lag orders:
#
# p = 1  -> 1 trading day
# p = 5  -> approximately 1 trading week
# p = 10 -> approximately 2 trading weeks
# p = 22 -> approximately 1 trading month
#
# Directions:
#
# dVVIX -> dVIX
# dVIX  -> dVVIX
#
# H0:
# Lagged values of the predictor do not provide additional
# predictive information for the dependent variable.
# ============================================================


# ============================================================
# 1. Packages
# ============================================================

from pathlib import Path

import pandas as pd

from statsmodels.tsa.api import VAR


# ============================================================
# 2. Project paths
# ============================================================

project_folder = Path(__file__).resolve().parent.parent

input_file = (
    project_folder
    / "data"
    / "VIX_VVIX_clean.xlsx"
)

output_folder = (
    project_folder
    / "output"
)

tables_folder = (
    output_folder
    / "tables"
    / "granger"
)

tables_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. Load data
# ============================================================

df = pd.read_excel(
    input_file,
    sheet_name="Data"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = (
    df
    .sort_values("Date")
    .drop_duplicates(subset="Date")
    .set_index("Date")
)

df = df.dropna(
    subset=[
        "VIX",
        "VVIX"
    ]
)


# ============================================================
# 4. First differences
# ============================================================

df["dVIX"] = (
    df["VIX"]
    .diff()
)

df["dVVIX"] = (
    df["VVIX"]
    .diff()
)


granger_data = (
    df[
        [
            "dVIX",
            "dVVIX"
        ]
    ]
    .dropna()
    .copy()
)


print(
    "\nFull-sample Granger causality analysis"
)

print(
    "=" * 75
)

print(
    f"Period: "
    f"{granger_data.index.min().date()} "
    f"to "
    f"{granger_data.index.max().date()}"
)

print(
    f"Number of observations: "
    f"{len(granger_data)}"
)


# ============================================================
# 5. VAR lag-order selection
# ============================================================
#
# Maximum candidate lag order = 22 trading days.
#
# BIC is used as the primary criterion because it penalises
# model complexity more strongly and therefore favours
# a more parsimonious specification.
# ============================================================

max_lag_selection = 22

var_model = VAR(
    granger_data
)

lag_selection = (
    var_model
    .select_order(
        maxlags=max_lag_selection
    )
)


print(
    "\nVAR lag-order selection"
)

print(
    "=" * 75
)

print(
    lag_selection.summary()
)


# ============================================================
# 5.1 Selected lag orders
# ============================================================

selected_aic = lag_selection.aic
selected_bic = lag_selection.bic
selected_hqic = lag_selection.hqic
selected_fpe = lag_selection.fpe


print(
    "\nSelected lag orders"
)

print(
    "=" * 75
)

print(
    f"AIC:  {selected_aic}"
)

print(
    f"BIC:  {selected_bic}"
)

print(
    f"HQIC: {selected_hqic}"
)

print(
    f"FPE:  {selected_fpe}"
)


# ============================================================
# 6. Main BIC specification
# ============================================================

selected_lag = int(
    selected_bic
)


if selected_lag < 1:
    raise ValueError(
        "BIC selected lag 0. "
        "Granger causality requires at least one lag."
    )


print(
    "\nMain Granger specification"
)

print(
    "=" * 75
)

print(
    f"BIC-selected lag order: "
    f"{selected_lag}"
)


# ============================================================
# 7. Alternative lag specifications
# ============================================================
#
# p = 1  -> 1 trading day
# p = 5  -> approximately 1 trading week
# p = 10 -> approximately 2 trading weeks
# p = 22 -> approximately 1 trading month
#
# ============================================================

lag_orders = [
    1,
    5,
    10,
    22
]


lag_labels = {
    1: "1 trading day",
    5: "Approx. 1 trading week",
    10: "Approx. 2 trading weeks",
    22: "Approx. 1 trading month"
}


# ============================================================
# 8. Run Granger causality tests
# ============================================================

granger_results = []


for p in lag_orders:

    # --------------------------------------------------------
    # 8.1 Estimate VAR(p)
    # --------------------------------------------------------

    var_results = (
        var_model
        .fit(p)
    )


    print(
        f"\nVAR({p}) estimation"
    )

    print(
        "=" * 75
    )

    print(
        f"Lag interpretation: "
        f"{lag_labels[p]}"
    )

    print(
        f"Effective observations: "
        f"{var_results.nobs}"
    )


    # --------------------------------------------------------
    # 8.2 Granger causality:
    # dVVIX -> dVIX
    # --------------------------------------------------------

    vvix_to_vix_test = (
        var_results
        .test_causality(
            caused="dVIX",
            causing=["dVVIX"],
            kind="f"
        )
    )


    granger_results.append(
        {
            "Direction":
                "dVVIX -> dVIX",

            "VAR lag order":
                p,

            "Lag interpretation":
                lag_labels[p],

            "F-statistic":
                vvix_to_vix_test.test_statistic,

            "p-value":
                vvix_to_vix_test.pvalue,

            "Reject H0 at 5%":
                vvix_to_vix_test.pvalue < 0.05,

            "BIC-selected":
                p == selected_lag
        }
    )


    # --------------------------------------------------------
    # 8.3 Granger causality:
    # dVIX -> dVVIX
    # --------------------------------------------------------

    vix_to_vvix_test = (
        var_results
        .test_causality(
            caused="dVVIX",
            causing=["dVIX"],
            kind="f"
        )
    )


    granger_results.append(
        {
            "Direction":
                "dVIX -> dVVIX",

            "VAR lag order":
                p,

            "Lag interpretation":
                lag_labels[p],

            "F-statistic":
                vix_to_vvix_test.test_statistic,

            "p-value":
                vix_to_vvix_test.pvalue,

            "Reject H0 at 5%":
                vix_to_vvix_test.pvalue < 0.05,

            "BIC-selected":
                p == selected_lag
        }
    )


# ============================================================
# 9. Convert results to DataFrame
# ============================================================

granger_results = pd.DataFrame(
    granger_results
)


# ============================================================
# 10. Print all results
# ============================================================

print(
    "\nGranger causality across alternative lag orders"
)

print(
    "=" * 110
)

print(
    granger_results
    .round(
        {
            "F-statistic": 4,
            "p-value": 6
        }
    )
    .to_string(index=False)
)


# ============================================================
# 11. Main direction: dVVIX -> dVIX
# ============================================================

vvix_to_vix_table = (
    granger_results[
        granger_results["Direction"]
        == "dVVIX -> dVIX"
    ]
    .copy()
)


print(
    "\nMain direction: dVVIX -> dVIX"
)

print(
    "=" * 95
)

print(
    vvix_to_vix_table
    .round(
        {
            "F-statistic": 4,
            "p-value": 6
        }
    )
    .to_string(index=False)
)


# ============================================================
# 12. Reverse direction: dVIX -> dVVIX
# ============================================================

vix_to_vvix_table = (
    granger_results[
        granger_results["Direction"]
        == "dVIX -> dVVIX"
    ]
    .copy()
)


print(
    "\nReverse direction: dVIX -> dVVIX"
)

print(
    "=" * 95
)

print(
    vix_to_vvix_table
    .round(
        {
            "F-statistic": 4,
            "p-value": 6
        }
    )
    .to_string(index=False)
)


# ============================================================
# 13. Lag-selection table
# ============================================================

lag_selection_table = pd.DataFrame(
    {
        "Criterion": [
            "AIC",
            "BIC",
            "HQIC",
            "FPE"
        ],

        "Selected lag": [
            selected_aic,
            selected_bic,
            selected_hqic,
            selected_fpe
        ]
    }
)


print(
    "\nInformation-criterion lag selection"
)

print(
    "=" * 75
)

print(
    lag_selection_table
    .to_string(index=False)
)


# ============================================================
# 14. Save results
# ============================================================

output_file = (
    tables_folder
    / "vix_vvix_granger_multiple_lags.xlsx"
)


with pd.ExcelWriter(
    output_file
) as writer:

    granger_results.to_excel(
        writer,
        sheet_name="All Results",
        index=False
    )

    vvix_to_vix_table.to_excel(
        writer,
        sheet_name="VVIX to VIX",
        index=False
    )

    vix_to_vvix_table.to_excel(
        writer,
        sheet_name="VIX to VVIX",
        index=False
    )

    lag_selection_table.to_excel(
        writer,
        sheet_name="Lag Selection",
        index=False
    )


print(
    "\nGranger causality analysis completed successfully."
)

print(
    f"\nResults saved to:\n"
    f"{output_file}"
)