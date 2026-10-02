# ============================================================
# VIX-VVIX LEAD-LAG ANALYSIS
#
# Samples:
# 1. Full sample
# 2. +/- 21 trading days around 13 March 2020
# 3. +/- 21 trading days around 2 May 2008
#
# Cross-correlations:
#
# k = 0 -> Corr(dVVIX_t, dVIX_t)
# k = 1 -> Corr(dVVIX_t, dVIX_t+1)
# k = 2 -> Corr(dVVIX_t, dVIX_t+2)
# k = 3 -> Corr(dVVIX_t, dVIX_t+3)
#
# Convention:
# Positive lag = VVIX leads VIX
#
# This analysis is descriptive.
# No confidence bounds or significance tests are applied.
# ============================================================


# ============================================================
# 1. Packages
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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

figures_folder = (
    output_folder
    / "figures"
    / "lead_lag"
)

tables_folder = (
    output_folder
    / "tables"
    / "lead_lag"
)

figures_folder.mkdir(
    parents=True,
    exist_ok=True
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


# ============================================================
# 5. Function to extract symmetric trading-day window
# ============================================================
#
# The window contains:
#
# 21 trading days before the event
# Event day
# 21 trading days after the event
#
# Total = 43 trading days
# ============================================================

def get_trading_day_window(
    data,
    event_date,
    days_before=21,
    days_after=21
):

    event_date = pd.Timestamp(
        event_date
    )

    if event_date not in data.index:
        raise ValueError(
            f"{event_date.date()} is not available "
            f"in the dataset."
        )

    event_position = (
        data.index
        .get_loc(event_date)
    )

    start_position = (
        event_position
        - days_before
    )

    end_position = (
        event_position
        + days_after
        + 1
    )

    if start_position < 0:
        raise ValueError(
            "Not enough observations before "
            f"{event_date.date()}."
        )

    if end_position > len(data):
        raise ValueError(
            "Not enough observations after "
            f"{event_date.date()}."
        )

    window = (
        data.iloc[
            start_position:
            end_position
        ]
        .copy()
    )

    return window


# ============================================================
# 6. Lead-lag analysis function
# ============================================================
#
# Corr(dVVIX_t, dVIX_{t+k})
#
# k = 0 -> contemporaneous
# k = 1 -> VVIX leads VIX by 1 trading day
# k = 2 -> VVIX leads VIX by 2 trading days
# k = 3 -> VVIX leads VIX by 3 trading days
#
# ============================================================

def lead_lag_analysis(
    data,
    label,
    file_prefix,
    y_limits=(-0.5, 0.8)
):

    # --------------------------------------------------------
    # 6.1 Prepare sample
    # --------------------------------------------------------

    sample = (
        data[
            [
                "dVIX",
                "dVVIX"
            ]
        ]
        .dropna()
        .copy()
    )

    n_obs = len(sample)

    lags = [
        0,
        1,
        2,
        3
    ]


    # --------------------------------------------------------
    # 6.2 Cross-correlations
    # --------------------------------------------------------

    results = []

    for k in lags:

        shifted_vix = (
            sample["dVIX"]
            .shift(-k)
        )

        valid_pairs = pd.concat(
            [
                sample["dVVIX"],
                shifted_vix
            ],
            axis=1
        ).dropna()

        valid_pairs.columns = [
            "dVVIX",
            "future_dVIX"
        ]

        n_effective = len(
            valid_pairs
        )

        correlation = (
            valid_pairs["dVVIX"]
            .corr(
                valid_pairs["future_dVIX"]
            )
        )

        results.append(
            {
                "Lag":
                    k,

                "Interpretation": (
                    "Contemporaneous"
                    if k == 0
                    else f"VVIX leads VIX by {k} day(s)"
                ),

                "Cross-correlation":
                    correlation,

                "N effective":
                    n_effective
            }
        )


    results_table = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # 6.3 Print results
    # --------------------------------------------------------

    print(
        f"\n{label}"
    )

    print(
        "=" * 80
    )

    print(
        f"Period: "
        f"{sample.index.min().date()} "
        f"to "
        f"{sample.index.max().date()}"
    )

    print(
        f"Number of observations: "
        f"{n_obs}"
    )

    print(
        "Convention: "
        "positive lag = VVIX leads VIX"
    )

    print(
        "\nCross-correlations"
    )

    print(
        "-" * 80
    )

    print(
        results_table
        .round(
            {
                "Cross-correlation": 4
            }
        )
        .to_string(index=False)
    )


    # --------------------------------------------------------
    # 6.4 Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        results_table["Lag"],
        results_table[
            "Cross-correlation"
        ],
        width=0.6
    )


    # Zero-correlation reference line
    ax.axhline(
        0,
        linewidth=0.8
    )


    # Same y-axis scale for all three figures
    ax.set_ylim(
        y_limits
    )


    ax.set_title(
        label
    )

    ax.set_xlabel(
        "Lag k (positive = VVIX leads VIX)"
    )

    ax.set_ylabel(
        "Cross-correlation"
    )

    ax.set_xticks(
        lags
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.35
    )

    fig.tight_layout()


    # --------------------------------------------------------
    # 6.5 Save figure
    # --------------------------------------------------------

    figure_path = (
        figures_folder
        / f"{file_prefix}.png"
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


    # --------------------------------------------------------
    # 6.6 Save individual table
    # --------------------------------------------------------

    table_path = (
        tables_folder
        / f"{file_prefix}.xlsx"
    )

    results_table.to_excel(
        table_path,
        index=False
    )


    # --------------------------------------------------------
    # 6.7 Return results for comparison
    # --------------------------------------------------------

    return {
        "Period":
            label,

        "Start":
            sample.index.min(),

        "End":
            sample.index.max(),

        "N":
            n_obs,

        "Corr k=0":
            results_table.loc[
                results_table["Lag"] == 0,
                "Cross-correlation"
            ].iloc[0],

        "Corr k=1":
            results_table.loc[
                results_table["Lag"] == 1,
                "Cross-correlation"
            ].iloc[0],

        "Corr k=2":
            results_table.loc[
                results_table["Lag"] == 2,
                "Cross-correlation"
            ].iloc[0],

        "Corr k=3":
            results_table.loc[
                results_table["Lag"] == 3,
                "Cross-correlation"
            ].iloc[0]
    }


# ============================================================
# 7. Full sample
# ============================================================

full_sample_results = lead_lag_analysis(
    data=df,

    label=(
        "VIX-VVIX Lead-Lag Cross-Correlation: "
        "Full Sample"
    ),

    file_prefix=(
        "lead_lag_full_sample"
    ),

    y_limits=(-0.5, 0.8)
)


# ============================================================
# 8. Window around 13 March 2020
# ============================================================
#
# 21 trading days before
# 13 March 2020
# 21 trading days after
# ============================================================

covid_event_date = (
    "2020-03-13"
)

covid_window = (
    get_trading_day_window(
        data=df,
        event_date=covid_event_date,
        days_before=21,
        days_after=21
    )
)


print(
    "\nCOVID trough window"
)

print(
    "=" * 60
)

print(
    f"Event date: "
    f"{covid_event_date}"
)

print(
    f"Window start: "
    f"{covid_window.index.min().date()}"
)

print(
    f"Window end: "
    f"{covid_window.index.max().date()}"
)

print(
    f"Trading days: "
    f"{len(covid_window)}"
)


covid_results = lead_lag_analysis(
    data=covid_window,

    label=(
        "VIX-VVIX Lead-Lag Cross-Correlation: "
        "Window around 13 March 2020"
    ),

    file_prefix=(
        "lead_lag_21days_around_2020_03_13"
    ),

    y_limits=(-0.5, 0.8)
)


# ============================================================
# 9. Window around 2 May 2008
# ============================================================
#
# 21 trading days before
# 2 May 2008
# 21 trading days after
# ============================================================

may_2008_event_date = (
    "2008-05-02"
)

may_2008_window = (
    get_trading_day_window(
        data=df,
        event_date=may_2008_event_date,
        days_before=21,
        days_after=21
    )
)


print(
    "\n2008 trough window"
)

print(
    "=" * 60
)

print(
    f"Event date: "
    f"{may_2008_event_date}"
)

print(
    f"Window start: "
    f"{may_2008_window.index.min().date()}"
)

print(
    f"Window end: "
    f"{may_2008_window.index.max().date()}"
)

print(
    f"Trading days: "
    f"{len(may_2008_window)}"
)


may_2008_results = lead_lag_analysis(
    data=may_2008_window,

    label=(
        "VIX-VVIX Lead-Lag Cross-Correlation: "
        "Window around 2 May 2008"
    ),

    file_prefix=(
        "lead_lag_21days_around_2008_05_02"
    ),

    y_limits=(-0.5, 0.8)
)


# ============================================================
# 10. Comparison across periods
# ============================================================

comparison_table = pd.DataFrame(
    [
        full_sample_results,
        covid_results,
        may_2008_results
    ]
)


correlation_columns = [
    "Corr k=0",
    "Corr k=1",
    "Corr k=2",
    "Corr k=3"
]


comparison_table[
    correlation_columns
] = (
    comparison_table[
        correlation_columns
    ]
    .round(4)
)


print(
    "\nComparison of lead-lag evidence across periods"
)

print(
    "=" * 120
)

print(
    comparison_table
    .to_string(index=False)
)


# ============================================================
# 11. Save comparison table
# ============================================================

comparison_table.to_excel(
    tables_folder
    / "lead_lag_period_comparison.xlsx",
    index=False
)


print(
    "\nAll lead-lag analyses completed successfully."
)
# ============================================================
# 12. Summary comparison figure
# ============================================================

summary_plot = pd.DataFrame(
    {
        "Lag": [0, 1, 2, 3],

        "Full sample": [
            full_sample_results["Corr k=0"],
            full_sample_results["Corr k=1"],
            full_sample_results["Corr k=2"],
            full_sample_results["Corr k=3"]
        ],

        "COVID trough": [
            covid_results["Corr k=0"],
            covid_results["Corr k=1"],
            covid_results["Corr k=2"],
            covid_results["Corr k=3"]
        ],

        "2008 trough": [
            may_2008_results["Corr k=0"],
            may_2008_results["Corr k=1"],
            may_2008_results["Corr k=2"],
            may_2008_results["Corr k=3"]
        ]
    }
)


# ============================================================
# 12.1 Plot grouped bars
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

x = np.arange(
    len(summary_plot["Lag"])
)

bar_width = 0.24


ax.bar(
    x - bar_width,
    summary_plot["Full sample"],
    width=bar_width,
    label="Full sample"
)

ax.bar(
    x,
    summary_plot["COVID trough"],
    width=bar_width,
    label="COVID trough"
)

ax.bar(
    x + bar_width,
    summary_plot["2008 trough"],
    width=bar_width,
    label="2008 trough"
)


# Zero line
ax.axhline(
    0,
    linewidth=0.8
)


# Same scale used in previous figures
ax.set_ylim(
    -0.5,
    0.8
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        "k=0",
        "k=1",
        "k=2",
        "k=3"
    ]
)


ax.set_xlabel(
    "Lead horizon"
)

ax.set_ylabel(
    "Cross-correlation"
)

ax.set_title(
    "Comparison of VIX-VVIX Lead-Lag Cross-Correlations"
)


ax.legend()


ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.35
)


fig.tight_layout()


# ============================================================
# 12.2 Save figure
# ============================================================

comparison_figure_path = (
    figures_folder
    / "lead_lag_comparison_full_covid_2008.png"
)

fig.savefig(
    comparison_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(
    "\nLead-lag comparison figure saved:"
)

print(
    comparison_figure_path
)