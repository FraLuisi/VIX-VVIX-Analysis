import pandas as pd
import numpy as np
import statsmodels.api as sm

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
# Create auxiliary variables
# ==========================

df["dVIX"] = df["VIX"].diff()
df["dVVIX"] = df["VVIX"].diff()

df["rVIX"] = np.log(df["VIX"]).diff()
df["rVVIX"] = np.log(df["VVIX"]).diff()

# IMPORTANT:
# Do NOT apply df.dropna() here.
# The first NaN created by diff() is irrelevant for the HAR regressions.
# Missing observations will be handled locally for each regression sample.

# ============================================================
# H1: IN-SAMPLE INCREMENTAL INFORMATION OF THE VVIX
# Robustness specification: HAC maxlags = h - 1
# ============================================================

# ============================================================
# 1. Log levels
# ============================================================

df["logVIX"] = np.log(df["VIX"])
df["logVVIX"] = np.log(df["VVIX"])

# ============================================================
# 2. HAR components
# ============================================================

har_windows = [1, 5, 10, 22, 66]

for k in har_windows:

    if k == 1:
        df[f"logVIX_{k}"] = df["logVIX"]

    else:
        df[f"logVIX_{k}"] = (
            df["logVIX"]
            .rolling(
                window=k,
                min_periods=k
            )
            .mean()
        )

# ============================================================
# 3. Future VIX targets
# ============================================================

forecast_horizons = [1, 5, 10, 22]

for h in forecast_horizons:

    # Direct forecast:
    # variables observed at t are matched with log(VIX) at t+h
    df[f"logVIX_target_{h}"] = df["logVIX"].shift(-h)

# ============================================================
# 4. Regressors and containers
# ============================================================

har_regressors = [
    "logVIX_1",
    "logVIX_5",
    "logVIX_10",
    "logVIX_22",
    "logVIX_66"
]

vvix_regressor = "logVVIX"

h1_results = []

har_models = {}
har_vvix_models = {}

# ============================================================
# 5. Estimate HAR and HAR-VVIX for each forecast horizon
# ============================================================

for h in forecast_horizons:

    print("\n" + "=" * 70)
    print(f"HORIZON: {h} TRADING DAY(S)")
    print("=" * 70)

    target = f"logVIX_target_{h}"

    required_columns = (
        [target]
        + har_regressors
        + [vvix_regressor]
    )

    # --------------------------------------------------------
    # SAME estimation sample for HAR and HAR-VVIX
    # --------------------------------------------------------

    regression_data = (
        df[required_columns]
        .dropna()
        .copy()
    )

    y = regression_data[target]

    # --------------------------------------------------------
    # Robustness HAC bandwidth: L = h - 1
    # --------------------------------------------------------

    hac_lags = h - 1

    print(f"Observations: {len(regression_data)}")
    print(f"HAC maxlags (h - 1): {hac_lags}")

    # --------------------------------------------------------
    # Benchmark HAR model
    # --------------------------------------------------------

    X_har = sm.add_constant(
        regression_data[har_regressors]
    )

    har_model = sm.OLS(
        y,
        X_har
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": hac_lags,
            "kernel": "bartlett"
        }
    )

    har_models[h] = har_model

    # --------------------------------------------------------
    # HAR-VVIX model
    # --------------------------------------------------------

    X_har_vvix = sm.add_constant(
        regression_data[
            har_regressors + [vvix_regressor]
        ]
    )

    har_vvix_model = sm.OLS(
        y,
        X_har_vvix
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": hac_lags,
            "kernel": "bartlett"
        }
    )

    har_vvix_models[h] = har_vvix_model

    # ========================================================
    # 6. VVIX coefficient: gamma^(h)
    # ========================================================

    gamma = har_vvix_model.params[vvix_regressor]
    gamma_se = har_vvix_model.bse[vvix_regressor]
    gamma_t = har_vvix_model.tvalues[vvix_regressor]
    gamma_pvalue = har_vvix_model.pvalues[vvix_regressor]

    # ========================================================
    # 7. Partial R-squared of the VVIX
    # ========================================================

    ssr_har = har_model.ssr
    ssr_har_vvix = har_vvix_model.ssr

    partial_r2 = (
        ssr_har - ssr_har_vvix
    ) / ssr_har

    # ========================================================
    # 8. Adjusted R-squared
    # ========================================================

    adj_r2_har = har_model.rsquared_adj
    adj_r2_har_vvix = har_vvix_model.rsquared_adj

    delta_adj_r2 = (
        adj_r2_har_vvix
        - adj_r2_har
    )

    # ========================================================
    # 9. Save results
    # ========================================================

    h1_results.append({
        "Horizon": h,
        "Observations": int(har_vvix_model.nobs),
        "Gamma_VVIX": gamma,
        "HAC_SE": gamma_se,
        "t_stat": gamma_t,
        "p_value": gamma_pvalue,
        "Partial_R2_VVIX": partial_r2,
        "Adj_R2_HAR": adj_r2_har,
        "Adj_R2_HAR_VVIX": adj_r2_har_vvix,
        "Delta_Adj_R2": delta_adj_r2,
        "HAC_maxlags": hac_lags
    })

# ============================================================
# 10. Final H1 results table
# ============================================================

h1_results_df = pd.DataFrame(h1_results)

print("\n" + "=" * 90)
print("H1: IN-SAMPLE INCREMENTAL INFORMATION OF THE VVIX")
print("=" * 90)

print(
    h1_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ============================================================
# 11. Compact thesis table
# ============================================================

h1_thesis_table = h1_results_df[
    [
        "Horizon",
        "Gamma_VVIX",
        "HAC_SE",
        "t_stat",
        "p_value",
        "Partial_R2_VVIX",
        "Adj_R2_HAR",
        "Adj_R2_HAR_VVIX"
    ]
].copy()

print("\nCOMPACT H1 TABLE")
print("-" * 90)

print(
    h1_thesis_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)