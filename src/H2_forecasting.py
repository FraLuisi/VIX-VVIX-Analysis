import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm, t as student_t

# ============================================================
# H2: PSEUDO-OUT-OF-SAMPLE FORECASTING
# ROLLING WINDOW = 2500 OBSERVATIONS (Fernandes et al., 2014)
#
# Design:
# - Rolling estimation window of 2500 usable regression pairs
# - Parameters re-estimated at every forecast origin
# - Direct forecasts at h = 1, 5, 10, 22
# - Same HAR / HAR-VVIX specification as H1
# - Same HAC lag rule as H1 (see hac_lags below)
# - Strict no-look-ahead rule:
#       a training pair (X_s, y_{s+h}) is usable at forecast
#       origin t only if s + h <= t
# - Same forecast-origin period across all horizons
# - Evaluation: RMSE, MAE, percentage improvements
# - Clark-West test (nested models, one-sided)
# - Diebold-Mariano test with HLN correction (squared and
#   absolute loss, two-sided), reported as a complement
# ============================================================


# ============================================================
# 0. Settings shared with H1
# ============================================================

ROLLING_WINDOW = 2500

har_windows = [1, 5, 10, 22, 66]
forecast_horizons = [1, 5, 10, 22]

har_regressors = [
    "logVIX_1",
    "logVIX_5",
    "logVIX_10",
    "logVIX_22",
    "logVIX_66",
]

vvix_regressor = "logVVIX"

# HAC lag rule. Must be IDENTICAL in H1 and H2.
#   "horizon":   L = floor(1.5 h) for h > 1, L = 1 for h = 1
#   "automatic": L = floor(4 (T/100)^(2/9)), statsmodels default
HAC_RULE = "automatic"


def hac_lags(h, nobs, rule=HAC_RULE):
    """
    Newey-West truncation lag.

    Direct h-step forecasts on daily data induce an MA(h-1)
    structure in regression residuals and in forecast-loss
    differentials. The "horizon" rule keeps a margin above h-1.
    The "automatic" rule is the statsmodels default and does not
    depend on h.
    """
    if rule == "horizon":
        return int(np.floor(1.5 * h)) if h > 1 else 1
    if rule == "automatic":
        return int(np.floor(4 * (nobs / 100.0) ** (2.0 / 9.0)))
    if rule == "h_minus_1":
        # Theoretical minimum. For h = 1 this is L = 0 (White).
        return h - 1
    raise ValueError(f"Unknown HAC rule: {rule}")


# ============================================================
# 1. Read data
# ============================================================

file_path = "../data/VIX_VVIX_clean.xlsx"

df = pd.read_excel(file_path, sheet_name="Data")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")

# Ordered trading-day position
df["_pos"] = np.arange(len(df))

# ============================================================
# 2. Log variables
# ============================================================

df["logVIX"] = np.log(df["VIX"])
df["logVVIX"] = np.log(df["VVIX"])

# ============================================================
# 3. HAR components (same definitions as H1)
# ============================================================

for k in har_windows:
    if k == 1:
        df[f"logVIX_{k}"] = df["logVIX"]
    else:
        df[f"logVIX_{k}"] = (
            df["logVIX"]
            .rolling(window=k, min_periods=k)
            .mean()
        )

# ============================================================
# 4. Direct forecast targets (same as H1)
# ============================================================

for h in forecast_horizons:
    df[f"logVIX_target_{h}"] = df["logVIX"].shift(-h)


# ============================================================
# 5. Common forecast-origin period
# ============================================================

def get_common_forecast_bounds(data, window_size, horizons):
    """
    First common origin: earliest t at which every horizon has at
    least `window_size` usable pairs (s + h <= t).
    Last common origin: leaves room for the longest horizon.
    """
    first_origins = []

    for h in horizons:
        target = f"logVIX_target_{h}"
        required = [target] + har_regressors + [vvix_regressor, "_pos"]

        valid_pairs = data[required].dropna().copy()
        valid_pairs["_target_pos"] = valid_pairs["_pos"] + h

        target_positions = np.sort(valid_pairs["_target_pos"].to_numpy())

        if len(target_positions) < window_size:
            raise ValueError(
                f"Not enough usable observations for h={h} "
                f"and rolling window={window_size}."
            )

        first_origins.append(int(target_positions[window_size - 1]))

    common_start = max(first_origins)
    common_end = len(data) - 1 - max(horizons)

    if common_start > common_end:
        raise ValueError(
            "No common out-of-sample period is available "
            f"with rolling window={window_size}."
        )

    return common_start, common_end


test_start_pos, test_end_pos = get_common_forecast_bounds(
    df, ROLLING_WINDOW, forecast_horizons
)

print("\n" + "=" * 100)
print("ROLLING-WINDOW FORECAST DESIGN")
print("=" * 100)
print(f"Total observations: {len(df)}")
print(f"Rolling estimation window: {ROLLING_WINDOW}")
print(f"HAC lag rule: {HAC_RULE}")
print("First common forecast origin:", df.index[test_start_pos].date())
print("Last common forecast origin:", df.index[test_end_pos].date())
print("Common OOS forecast origins:", test_end_pos - test_start_pos + 1)


# ============================================================
# 6. Forecast comparison tests
# ============================================================

def _hac_mean_tstat(series, maxlags):
    """t-statistic of the mean of `series` with Newey-West HAC s.e."""
    series = np.asarray(series, dtype=float)
    X = np.ones((len(series), 1))
    model = sm.OLS(series, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags, "kernel": "bartlett"},
    )
    return float(model.tvalues[0])


def clark_west_test(actual, f_har, f_hv, h, lags=None):
    """
    Clark and West (2007) MSPE-adjusted test for nested models.
    H0: equal MSPE (benchmark = HAR).
    H1: HAR-VVIX has lower MSPE (one-sided).
    """
    actual = np.asarray(actual, dtype=float)
    f0 = np.asarray(f_har, dtype=float)
    f1 = np.asarray(f_hv, dtype=float)

    e0 = actual - f0
    e1 = actual - f1

    cw_diff = e0 ** 2 - e1 ** 2 + (f0 - f1) ** 2

    if lags is None:
        lags = hac_lags(h, len(cw_diff))
    stat = _hac_mean_tstat(cw_diff, lags)
    p_one_sided = 1.0 - norm.cdf(stat)

    return stat, p_one_sided, lags


def diebold_mariano_test(actual, f_har, f_hv, h, loss="squared", lags=None):
    """
    Diebold-Mariano (1995) test with the Harvey-Leybourne-Newbold
    (1997) small-sample correction. Two-sided, Student-t with
    n-1 degrees of freedom.

    d_t = L(e_HAR) - L(e_HAR_VVIX); positive mean favours HAR-VVIX.

    Reported as a complement to Clark-West: for nested models the
    DM statistic is conservative under H0, so failure to reject is
    not by itself informative, while rejection is.
    """
    actual = np.asarray(actual, dtype=float)
    e0 = actual - np.asarray(f_har, dtype=float)
    e1 = actual - np.asarray(f_hv, dtype=float)

    if loss == "squared":
        d = e0 ** 2 - e1 ** 2
    elif loss == "absolute":
        d = np.abs(e0) - np.abs(e1)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    n = len(d)
    if lags is None:
        lags = hac_lags(h, n)
    stat = _hac_mean_tstat(d, lags)

    # HLN correction
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat_hln = stat * hln
    p_two_sided = 2.0 * (1.0 - student_t.cdf(abs(stat_hln), df=n - 1))

    return stat_hln, p_two_sided, lags


# ============================================================
# 7. Rolling-window pseudo-OOS forecasts
# ============================================================

forecast_rows = []
needed_predictors = har_regressors + [vvix_regressor]

for h in forecast_horizons:

    target = f"logVIX_target_{h}"
    required_columns = [target] + har_regressors + [vvix_regressor, "_pos"]

    # Row s contains predictors X_s and future target y_{s+h}
    pair_data = df[required_columns].dropna().copy()
    pair_data["_target_pos"] = pair_data["_pos"] + h

    print("\n" + "-" * 100)
    print(f"GENERATING ROLLING FORECASTS: h = {h}")
    print("-" * 100)

    for t_pos in range(test_start_pos, test_end_pos + 1):

        current_row = df.iloc[t_pos]

        if current_row[needed_predictors].isna().any():
            continue

        actual_logvix = df.iloc[t_pos + h]["logVIX"]
        if pd.isna(actual_logvix):
            continue

        # Strict real-time information set: target already observed
        eligible = pair_data[pair_data["_target_pos"] <= t_pos]
        if len(eligible) < ROLLING_WINDOW:
            continue

        train = eligible.tail(ROLLING_WINDOW)
        y_train = train[target]

        # HAR benchmark
        X_har_train = sm.add_constant(
            train[har_regressors], has_constant="add"
        )
        har_model = sm.OLS(y_train, X_har_train).fit()

        # HAR-VVIX
        X_hv_train = sm.add_constant(
            train[har_regressors + [vvix_regressor]], has_constant="add"
        )
        hv_model = sm.OLS(y_train, X_hv_train).fit()

        # Regressors observed at t, in estimation-matrix column order
        x_har_t = np.array(
            [1.0] + [current_row[v] for v in har_regressors]
        )
        x_hv_t = np.array(
            [1.0]
            + [current_row[v] for v in har_regressors]
            + [current_row[vvix_regressor]]
        )

        forecast_har = float(x_har_t @ har_model.params.to_numpy())
        forecast_hv = float(x_hv_t @ hv_model.params.to_numpy())

        forecast_rows.append({
            "Horizon": h,
            "Forecast_Origin": df.index[t_pos],
            "Target_Date": df.index[t_pos + h],
            "Training_Observations": len(train),
            "Actual_logVIX": float(actual_logvix),
            "Forecast_HAR": forecast_har,
            "Forecast_HAR_VVIX": forecast_hv,
            "Error_HAR": float(actual_logvix - forecast_har),
            "Error_HAR_VVIX": float(actual_logvix - forecast_hv),
            "Gamma_rolling": float(hv_model.params[vvix_regressor]),
        })

forecasts = pd.DataFrame(forecast_rows)

# ============================================================
# 8. Forecast evaluation
# ============================================================

results = []

for h in forecast_horizons:

    d = forecasts[forecasts["Horizon"] == h].dropna().copy()

    if len(d) == 0:
        raise ValueError(f"No out-of-sample forecasts generated for h={h}.")

    e_har = d["Error_HAR"].to_numpy()
    e_hv = d["Error_HAR_VVIX"].to_numpy()

    rmse_har = np.sqrt(np.mean(e_har ** 2))
    rmse_hv = np.sqrt(np.mean(e_hv ** 2))
    mae_har = np.mean(np.abs(e_har))
    mae_hv = np.mean(np.abs(e_hv))

    # Positive = HAR-VVIX performs better
    rmse_improvement = 100 * (rmse_har - rmse_hv) / rmse_har
    mae_improvement = 100 * (mae_har - mae_hv) / mae_har

    cw_stat, cw_p, cw_lags = clark_west_test(
        d["Actual_logVIX"], d["Forecast_HAR"], d["Forecast_HAR_VVIX"], h
    )

    dm_sq_stat, dm_sq_p, _ = diebold_mariano_test(
        d["Actual_logVIX"], d["Forecast_HAR"], d["Forecast_HAR_VVIX"],
        h, loss="squared",
    )

    dm_abs_stat, dm_abs_p, _ = diebold_mariano_test(
        d["Actual_logVIX"], d["Forecast_HAR"], d["Forecast_HAR_VVIX"],
        h, loss="absolute",
    )

    results.append({
        "Horizon": h,
        "OOS_Observations": len(d),
        "Rolling_Window": ROLLING_WINDOW,
        "RMSE_HAR": rmse_har,
        "RMSE_HAR_VVIX": rmse_hv,
        "RMSE_Improvement_pct": rmse_improvement,
        "MAE_HAR": mae_har,
        "MAE_HAR_VVIX": mae_hv,
        "MAE_Improvement_pct": mae_improvement,
        "CW_stat": cw_stat,
        "CW_pvalue_one_sided": cw_p,
        "DM_sq_stat_HLN": dm_sq_stat,
        "DM_sq_pvalue": dm_sq_p,
        "DM_abs_stat_HLN": dm_abs_stat,
        "DM_abs_pvalue": dm_abs_p,
        "HAC_maxlags": cw_lags,
    })

results_df = pd.DataFrame(results)

# ============================================================
# 9. Output
# ============================================================

print("\n" + "=" * 120)
print("H2: ROLLING-WINDOW PSEUDO-OUT-OF-SAMPLE FORECASTING")
print(f"ROLLING WINDOW = {ROLLING_WINDOW}, HAC RULE = {HAC_RULE}")
print(
    "COMMON FORECAST-ORIGIN PERIOD:",
    df.index[test_start_pos].date(), "to", df.index[test_end_pos].date(),
)
print("=" * 120)

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

thesis_table = results_df[
    [
        "Horizon",
        "OOS_Observations",
        "RMSE_HAR",
        "RMSE_HAR_VVIX",
        "RMSE_Improvement_pct",
        "MAE_HAR",
        "MAE_HAR_VVIX",
        "MAE_Improvement_pct",
        "CW_stat",
        "CW_pvalue_one_sided",
        "DM_sq_stat_HLN",
        "DM_sq_pvalue",
        "DM_abs_stat_HLN",
        "DM_abs_pvalue",
    ]
].copy()

print("\n" + "=" * 120)
print("COMPACT H2 TABLE")
print("=" * 120)
print(thesis_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================================================
# 10. Sensitivity of OOS tests to the HAC truncation lag
#     Forecasts and point statistics are unchanged; only the
#     covariance of the loss differentials varies.
# ============================================================

HAC_RULES_SENSITIVITY = ["h_minus_1", "horizon", "automatic"]

sensitivity_rows = []

for h in forecast_horizons:

    d = forecasts[forecasts["Horizon"] == h].dropna().copy()
    n = len(d)

    for rule in HAC_RULES_SENSITIVITY:
        L = hac_lags(h, n, rule=rule)

        cw_s, cw_p, _ = clark_west_test(
            d["Actual_logVIX"], d["Forecast_HAR"], d["Forecast_HAR_VVIX"],
            h, lags=L,
        )
        dm_sq_s, dm_sq_p, _ = diebold_mariano_test(
            d["Actual_logVIX"], d["Forecast_HAR"], d["Forecast_HAR_VVIX"],
            h, loss="squared", lags=L,
        )
        dm_abs_s, dm_abs_p, _ = diebold_mariano_test(
            d["Actual_logVIX"], d["Forecast_HAR"], d["Forecast_HAR_VVIX"],
            h, loss="absolute", lags=L,
        )

        sensitivity_rows.append({
            "Horizon": h,
            "HAC_rule": rule,
            "L": L,
            "CW_stat": cw_s,
            "CW_p": cw_p,
            "DM_sq_stat": dm_sq_s,
            "DM_sq_p": dm_sq_p,
            "DM_abs_stat": dm_abs_s,
            "DM_abs_p": dm_abs_p,
        })

sensitivity_df = pd.DataFrame(sensitivity_rows)

print("\n" + "=" * 120)
print("SENSITIVITY OF OOS TESTS TO THE HAC TRUNCATION LAG")
print("(h_minus_1: L = h-1, White for h = 1; horizon: floor(1.5h), 1 for h = 1; "
      "automatic: floor(4(T/100)^(2/9)))")
print("=" * 120)
print(sensitivity_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Compact wide view: p-values only, one row per horizon
wide = sensitivity_df.pivot(
    index="Horizon", columns="HAC_rule",
    values=["L", "CW_p", "DM_sq_p", "DM_abs_p"],
)
wide = wide.reindex(columns=HAC_RULES_SENSITIVITY, level=1)

print("\nCOMPACT SENSITIVITY TABLE (p-values)")
print("-" * 120)
print(wide.to_string(float_format=lambda x: f"{x:.3f}"))

# ============================================================
# 11. Exports (needed for regime / subsample analysis)
# ============================================================

forecasts.to_csv(f"H2_rolling_{ROLLING_WINDOW}_forecasts.csv", index=False)
results_df.to_csv(f"H2_rolling_{ROLLING_WINDOW}_results.csv", index=False)
sensitivity_df.to_csv(f"H2_rolling_{ROLLING_WINDOW}_hac_sensitivity.csv", index=False)
