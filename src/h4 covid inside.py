import pandas as pd
import numpy as np
import statsmodels.api as sm

# ============================================================
# H4 ROBUSTNESS:
# EVENT-STUDY FORECASTING AROUND ADDITIONAL
# VIX-VVIX CORRELATION TROUGHS ASSOCIATED WITH OFR STRESS
#
# Models:
#   HAR
#   HAR-VVIX
#
# Rolling estimation window:
#   1000 trading observations
#
# Forecast horizons:
#   h = 1, 2, 3, 5 trading days
#
# Event windows:
#   +/- 1 calendar month around each correlation trough
#
# Main H4 events are excluded.
# 2008-2009 events are excluded because there is insufficient
# pre-event history for a common 1000-observation rolling window.
#
# Output:
#   Forecast performance tables only
# ============================================================


# ============================================================
# 1. READ DATA
# ============================================================

file_path = "../data/VIX_VVIX_clean.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="Data"
)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df = df.set_index("Date")


# ============================================================
# 2. LOG LEVELS
# ============================================================

df["logVIX"] = np.log(df["VIX"])
df["logVVIX"] = np.log(df["VVIX"])


# ============================================================
# 3. HAR COMPONENTS
# Same specification as H1-H3
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
# 4. FORECAST HORIZONS
# ============================================================

forecast_horizons = [1, 2, 3, 5]

date_series = pd.Series(
    df.index,
    index=df.index
)

for h in forecast_horizons:

    # Direct forecast target:
    # information available at t predicts log(VIX) at t+h
    df[f"logVIX_target_{h}"] = (
        df["logVIX"].shift(-h)
    )

    # Actual date corresponding to t+h
    df[f"target_date_{h}"] = (
        date_series.shift(-h)
    )


# ============================================================
# 5. MODEL SPECIFICATION
# ============================================================

har_regressors = [
    "logVIX_1",
    "logVIX_5",
    "logVIX_10",
    "logVIX_22",
    "logVIX_66"
]

vvix_regressor = "logVVIX"

ROLLING_WINDOW = 1000


# ============================================================
# 6. ROBUSTNESS EVENT DEFINITIONS
#
# Additional correlation troughs associated with OFR
# financial stress.
#
# Excluded:
#   - 2008-2009 events:
#       insufficient history for W = 1000
#
#   - Main H4 events:
#       2012-04-05
#       2016-06-09
#       2020-03-13
#       2023-03-06
#
# The same methodology is applied to all robustness events.
# ============================================================


events = {

    "COVID Feb 2020":
        pd.Timestamp("2020-02-14"),

    "COVID Late Feb 2020":
        pd.Timestamp("2020-02-28"),

    "COVID Mar 2020":
        pd.Timestamp("2020-03-13"),

    "COVID Ape 2020":
        pd.Timestamp("2020-04-13"),

    "COVID June 2020":
        pd.Timestamp("2020-06-01"),

    "COVID July 2020":
    pd.Timestamp("2020-07-01"),
}

# ============================================================
# 7. CREATE +/- 1 MONTH EVENT WINDOWS
# ============================================================

event_windows = {}

for event_name, anchor_date in events.items():

    event_start = (
        anchor_date
        - pd.DateOffset(months=1)
    )

    event_end = (
        anchor_date
        + pd.DateOffset(months=1)
    )

    event_windows[event_name] = {
        "Anchor": anchor_date,
        "Start": event_start,
        "End": event_end
    }


# ============================================================
# 8. PRINT EVENT WINDOWS
# ============================================================

print("\n" + "=" * 110)
print("H4 ROBUSTNESS EVENT WINDOWS")
print("=" * 110)

for event_name, dates in event_windows.items():

    print(
        f"{event_name:<35} "
        f"| Start: {dates['Start'].date()} "
        f"| Trough: {dates['Anchor'].date()} "
        f"| End: {dates['End'].date()}"
    )


# ============================================================
# 9. CONTAINER FOR RESULTS
# ============================================================

results = []


# ============================================================
# 10. LOOP OVER EVENTS
# ============================================================

for event_name, dates in event_windows.items():

    EVENT_START = dates["Start"]
    EVENT_END = dates["End"]
    ANCHOR_DATE = dates["Anchor"]

    # ========================================================
    # LOOP OVER FORECAST HORIZONS
    # ========================================================

    for h in forecast_horizons:

        target = f"logVIX_target_{h}"
        target_date_col = f"target_date_{h}"

        required_columns = (
            [target, target_date_col]
            + har_regressors
            + [vvix_regressor]
        )

        model_data = (
            df[required_columns]
            .dropna()
            .copy()
        )

        forecast_records = []

        # ====================================================
        # ROLLING PSEUDO-OUT-OF-SAMPLE FORECASTING
        # ====================================================

        for i in range(
            ROLLING_WINDOW,
            len(model_data)
        ):

            forecast_origin = (
                model_data.index[i]
            )

            target_date = pd.Timestamp(
                model_data.iloc[i][target_date_col]
            )

            # ------------------------------------------------
            # Evaluate only targets inside the event window
            # ------------------------------------------------

            if not (
                EVENT_START
                <= target_date
                <= EVENT_END
            ):
                continue

            # ------------------------------------------------
            # Previous 1000 usable observations
            # ------------------------------------------------

            train = model_data.iloc[
                i - ROLLING_WINDOW:i
            ].copy()

            current = model_data.iloc[
                [i]
            ].copy()

            y_train = train[target]


            # =================================================
            # HAR
            # =================================================

            X_train_har = sm.add_constant(
                train[har_regressors],
                has_constant="add"
            )

            har_model = sm.OLS(
                y_train,
                X_train_har
            ).fit()

            X_current_har = sm.add_constant(
                current[har_regressors],
                has_constant="add"
            )

            forecast_har_log = float(
                har_model.predict(
                    X_current_har
                ).iloc[0]
            )


            # =================================================
            # HAR-VVIX
            # =================================================

            X_train_vvix = sm.add_constant(
                train[
                    har_regressors
                    + [vvix_regressor]
                ],
                has_constant="add"
            )

            har_vvix_model = sm.OLS(
                y_train,
                X_train_vvix
            ).fit()

            X_current_vvix = sm.add_constant(
                current[
                    har_regressors
                    + [vvix_regressor]
                ],
                has_constant="add"
            )

            forecast_vvix_log = float(
                har_vvix_model.predict(
                    X_current_vvix
                ).iloc[0]
            )


            # =================================================
            # ACTUAL VIX
            # =================================================

            actual_log = float(
                current[target].iloc[0]
            )

            # Convert forecasts and actual values
            # back to VIX levels
            actual_vix = np.exp(
                actual_log
            )

            forecast_har = np.exp(
                forecast_har_log
            )

            forecast_har_vvix = np.exp(
                forecast_vvix_log
            )


            # =================================================
            # FORECAST ERRORS
            # =================================================

            error_har = (
                actual_vix
                - forecast_har
            )

            error_vvix = (
                actual_vix
                - forecast_har_vvix
            )


            forecast_records.append({

                "Forecast_Origin":
                    forecast_origin,

                "Target_Date":
                    target_date,

                "Actual_VIX":
                    actual_vix,

                "Forecast_HAR":
                    forecast_har,

                "Forecast_HAR_VVIX":
                    forecast_har_vvix,

                "Error_HAR":
                    error_har,

                "Error_HAR_VVIX":
                    error_vvix
            })


        # ====================================================
        # 11. FORECAST ACCURACY
        # ====================================================

        fc = pd.DataFrame(
            forecast_records
        )

        if fc.empty:

            print(
                f"WARNING: No forecasts for "
                f"{event_name}, h={h}. "
                f"Insufficient pre-event observations "
                f"for W={ROLLING_WINDOW}."
            )

            continue


        # ----------------------------------------------------
        # RMSE
        # ----------------------------------------------------

        rmse_har = np.sqrt(
            np.mean(
                fc["Error_HAR"] ** 2
            )
        )

        rmse_vvix = np.sqrt(
            np.mean(
                fc["Error_HAR_VVIX"] ** 2
            )
        )


        # ----------------------------------------------------
        # MAE
        # ----------------------------------------------------

        mae_har = np.mean(
            np.abs(
                fc["Error_HAR"]
            )
        )

        mae_vvix = np.mean(
            np.abs(
                fc["Error_HAR_VVIX"]
            )
        )


        # ====================================================
        # 12. PERCENTAGE IMPROVEMENTS
        #
        # Positive:
        # HAR-VVIX performs better than HAR
        #
        # Negative:
        # HAR performs better than HAR-VVIX
        # ====================================================

        rmse_improvement = (
            (rmse_har - rmse_vvix)
            / rmse_har
            * 100
        )

        mae_improvement = (
            (mae_har - mae_vvix)
            / mae_har
            * 100
        )


        # ====================================================
        # 13. SAVE RESULTS
        # ====================================================

        results.append({

            "Event":
                event_name,

            "Trough_Date":
                ANCHOR_DATE.date(),

            "Horizon":
                h,

            "N":
                len(fc),

            "RMSE_HAR":
                rmse_har,

            "RMSE_HAR_VVIX":
                rmse_vvix,

            "Delta_RMSE_pct":
                rmse_improvement,

            "MAE_HAR":
                mae_har,

            "MAE_HAR_VVIX":
                mae_vvix,

            "Delta_MAE_pct":
                mae_improvement
        })


# ============================================================
# 14. CONVERT RESULTS TO DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 15. FULL RESULTS TABLE
# ============================================================

print("\n" + "=" * 140)
print(
    "H4 ROBUSTNESS: ADDITIONAL STRESS-EVENT "
    "OUT-OF-SAMPLE FORECAST PERFORMANCE"
)
print(f"Rolling window = {ROLLING_WINDOW}")
print(
    "Forecast horizons = "
    + ", ".join(
        str(h) for h in forecast_horizons
    )
    + " trading days"
)
print(
    "Event window = +/- 1 calendar month "
    "around correlation trough"
)
print("=" * 140)


if results_df.empty:

    print(
        "\nNo robustness events have sufficient "
        f"pre-event history for W={ROLLING_WINDOW}."
    )

else:

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# 16. COMPACT DELTA RMSE TABLE
# ============================================================

if not results_df.empty:

    delta_rmse_table = (
        results_df
        .pivot(
            index="Event",
            columns="Horizon",
            values="Delta_RMSE_pct"
        )
        .reindex(events.keys())
    )

    delta_rmse_table.columns = [
        f"h={h}"
        for h in delta_rmse_table.columns
    ]

    print("\n" + "=" * 100)
    print("H4 ROBUSTNESS: DELTA RMSE (%)")
    print(
        "Positive values = "
        "HAR-VVIX improves upon HAR"
    )
    print("=" * 100)

    print(
        delta_rmse_table.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )


# ============================================================
# 17. COMPACT DELTA MAE TABLE
# ============================================================

if not results_df.empty:

    delta_mae_table = (
        results_df
        .pivot(
            index="Event",
            columns="Horizon",
            values="Delta_MAE_pct"
        )
        .reindex(events.keys())
    )

    delta_mae_table.columns = [
        f"h={h}"
        for h in delta_mae_table.columns
    ]

    print("\n" + "=" * 100)
    print("H4 ROBUSTNESS: DELTA MAE (%)")
    print(
        "Positive values = "
        "HAR-VVIX improves upon HAR"
    )
    print("=" * 100)

    print(
        delta_mae_table.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )


# ============================================================
# 18. SINGLE COMPACT ROBUSTNESS TABLE
# ============================================================

if not results_df.empty:

    compact_table = (
        results_df[
            [
                "Event",
                "Trough_Date",
                "Horizon",
                "Delta_RMSE_pct",
                "Delta_MAE_pct"
            ]
        ]
        .copy()
    )

    compact_table["Delta_RMSE_pct"] = (
        compact_table["Delta_RMSE_pct"]
        .round(3)
    )

    compact_table["Delta_MAE_pct"] = (
        compact_table["Delta_MAE_pct"]
        .round(3)
    )

    print("\n" + "=" * 100)
    print("COMPACT H4 ROBUSTNESS TABLE")
    print("=" * 100)

    print(
        compact_table.to_string(
            index=False
        )
    )


# ============================================================
# 19. SIMPLE SUMMARY BY HORIZON
#
# This does NOT replace formal statistical testing.
# It simply reports how often HAR-VVIX improves upon HAR
# across the additional stress events.
# ============================================================

if not results_df.empty:

    horizon_summary = (
        results_df
        .groupby("Horizon")
        .agg(
            Events=("Event", "count"),

            Positive_Delta_RMSE=(
                "Delta_RMSE_pct",
                lambda x: (x > 0).sum()
            ),

            Mean_Delta_RMSE_pct=(
                "Delta_RMSE_pct",
                "mean"
            ),

            Positive_Delta_MAE=(
                "Delta_MAE_pct",
                lambda x: (x > 0).sum()
            ),

            Mean_Delta_MAE_pct=(
                "Delta_MAE_pct",
                "mean"
            )
        )
        .reset_index()
    )

    print("\n" + "=" * 100)
    print("ROBUSTNESS SUMMARY BY FORECAST HORIZON")
    print("=" * 100)

    print(
        horizon_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )