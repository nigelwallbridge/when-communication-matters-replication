"""
Parameter sweeps for:

    Wallbridge & Wallbridge
    "When Communication Matters: Quantifying Communicative Value"

This program runs two systematic Monte Carlo experiments using the
scientific model in WCM_model.py and the Monte Carlo engine in
WCM_monte_carlo.py:

1. Source-accuracy sweep
2. Initial-utility (prior-wealth) sweep

One row of summary statistics is written for each parameter condition.

Output:
    WCM_sweep_results.csv
"""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path

from WCM_model import (
    SimulationParameters,
    calculate_Derived_receiver_accuracy,
)

from WCM_monte_carlo import (
    ExperimentParameters,
    run_monte_carlo,
)


# ---------------------------------------------------------------------
# BASELINE SCIENTIFIC PARAMETERS
# ---------------------------------------------------------------------

BASE_MODEL_PARAMS = SimulationParameters(
    n_repetitions=50,
    true_probability_favourable=0.60,
    initial_utility=5.0,
    prior_probability_favourable=0.60,
    A=6.0,
    B=-4.0,
    C=-1.0,
    D=2.0,
    source_accuracy=0.70,
    subjective_p_favourable_signal_given_favourable=0.55,
    subjective_p_favourable_signal_given_unfavourable=0.45,
)

EXPERIMENT_PARAMS = ExperimentParameters(
    n_simulations=10_000,
    master_seed=20260730,
    output_csv="outputs/WCM_monte_carlo_results.csv",
)


# ---------------------------------------------------------------------
# PARAMETER VALUES TO SWEEP
# ---------------------------------------------------------------------

# Refined sweep around chance accuracy (0.5), plus broader values.
SOURCE_ACCURACIES = [
    0.30,
    0.40,
    0.45,
    0.475,
    0.49,
    0.50,
    0.51,
    0.525,
    0.55,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

# Preliminary prior-wealth sweep.
#
# These values are intended to reveal the broad shape first. After seeing
# the result, the sweep can be made denser around any interesting transition.
INITIAL_UTILITIES = [
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
    15.0,
    20.0,
]

# Source accuracy held constant during the prior-wealth experiment.
# Change here if a different fixed source quality is preferred.
SOURCE_ACCURACY_FOR_WEALTH_SWEEP = 0.70

OUTPUT_CSV = Path("outputs/WCM_sweep_results.csv")


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def difference(a: float, b: float) -> float:
    """Return a - b, preserving NaN where appropriate."""
    if math.isnan(a) or math.isnan(b):
        return math.nan
    return a - b


def summarise_one_condition(
    *,
    experiment_name: str,
    sweep_parameter: str,
    sweep_value: float,
    model_params: SimulationParameters,
) -> dict[str, float | int | str]:
    """Run one Monte Carlo condition and return one summary row."""

    summary, _records = run_monte_carlo(
        model_params,
        EXPERIMENT_PARAMS,
    )

    without = summary.without_communication
    with_comm = summary.with_communication
    paired = summary.paired
    accuracy = summary.accuracy

    derived_receiver_accuracy = (
        calculate_Derived_receiver_accuracy(model_params)
    )

    survivor_utility_gain = difference(
        with_comm.mean_final_utility_if_survived,
        without.mean_final_utility_if_survived,
    )

    overall_utility_gain = (
        with_comm.mean_final_utility
        - without.mean_final_utility
    )

    return {
        # Experiment identification
        "experiment": experiment_name,
        "sweep_parameter": sweep_parameter,
        "sweep_value": sweep_value,

        # Scientific parameters needed to interpret each row
        "initial_utility": model_params.initial_utility,
        "source_accuracy": model_params.source_accuracy,
        "prior_probability_favourable":
            model_params.prior_probability_favourable,
        "true_probability_favourable":
            model_params.true_probability_favourable,
        "Derived_receiver_accuracy":
            derived_receiver_accuracy,

        # Monte Carlo settings
        "n_simulations": EXPERIMENT_PARAMS.n_simulations,
        "n_repetitions": model_params.n_repetitions,
        "master_seed": EXPERIMENT_PARAMS.master_seed,

        # Survival
        "survival_rate_without_communication":
            without.survival_rate,
        "survival_rate_with_communication":
            with_comm.survival_rate,
        "survival_gain":
            paired.survival_gain,
        "se_survival_gain":
            paired.standard_error_survival_gain,

        # Utility among survivors
        "mean_utility_survivors_without_communication":
            without.mean_final_utility_if_survived,
        "se_mean_utility_survivors_without_communication":
            without.standard_error_if_survived,
        "mean_utility_survivors_with_communication":
            with_comm.mean_final_utility_if_survived,
        "se_mean_utility_survivors_with_communication":
            with_comm.standard_error_if_survived,
        "mean_utility_survivors_gain":
            survivor_utility_gain,

        # Utility across all runs
        "mean_final_utility_without_communication":
            without.mean_final_utility,
        "mean_final_utility_with_communication":
            with_comm.mean_final_utility,
        "mean_final_utility_gain":
            overall_utility_gain,

        # Paired utility comparison
        "mean_paired_utility_difference":
            paired.mean_difference,
        "se_paired_utility_difference":
            paired.standard_error_difference,

        # Decisive communication
        "mean_decisive_events_per_run":
            paired.mean_decisive_events,
        "rate_runs_with_decisive_event":
            paired.rate_with_any_decisive_event,

        # Realised predictive accuracy
        "pooled_realised_sender_accuracy":
            accuracy.pooled_realised_sender_accuracy,
        "pooled_realised_receiver_accuracy":
            accuracy.pooled_realised_receiver_accuracy,
        "pooled_realised_accuracy_difference":
            accuracy.pooled_realised_accuracy_difference,
    }


# ---------------------------------------------------------------------
# SOURCE-ACCURACY SWEEP
# ---------------------------------------------------------------------

def run_source_accuracy_sweep() -> list[dict[str, float | int | str]]:
    """Vary source accuracy while holding other parameters constant."""

    rows: list[dict[str, float | int | str]] = []
    total = len(SOURCE_ACCURACIES)

    print("\nSOURCE-ACCURACY SWEEP")
    print("-" * 60)

    for number, source_accuracy in enumerate(
        SOURCE_ACCURACIES,
        start=1,
    ):
        print(
            f"Condition {number}/{total}: "
            f"source_accuracy = {source_accuracy:.3f}"
        )

        model_params = replace(
            BASE_MODEL_PARAMS,
            initial_utility=5.0,
            source_accuracy=source_accuracy,
        )

        row = summarise_one_condition(
            experiment_name="source_accuracy",
            sweep_parameter="source_accuracy",
            sweep_value=source_accuracy,
            model_params=model_params,
        )

        rows.append(row)

        print(
            f"  Survival gain:              "
            f"{row['survival_gain']:+.4f} "
            f"(SE {row['se_survival_gain']:.4f})"
        )
        print(
            f"  Mean survivor utility gain: "
            f"{row['mean_utility_survivors_gain']:+.3f}"
        )

    return rows


# ---------------------------------------------------------------------
# PRIOR-WEALTH / INITIAL-UTILITY SWEEP
# ---------------------------------------------------------------------

def run_initial_utility_sweep() -> list[dict[str, float | int | str]]:
    """Vary initial utility while holding source accuracy constant."""

    rows: list[dict[str, float | int | str]] = []
    total = len(INITIAL_UTILITIES)

    print("\nPRIOR-WEALTH / INITIAL-UTILITY SWEEP")
    print("-" * 60)
    print(
        f"Fixed source accuracy: "
        f"{SOURCE_ACCURACY_FOR_WEALTH_SWEEP:.3f}"
    )

    for number, initial_utility in enumerate(
        INITIAL_UTILITIES,
        start=1,
    ):
        print(
            f"Condition {number}/{total}: "
            f"initial_utility = {initial_utility:.3f}"
        )

        model_params = replace(
            BASE_MODEL_PARAMS,
            initial_utility=initial_utility,
            source_accuracy=SOURCE_ACCURACY_FOR_WEALTH_SWEEP,
        )

        row = summarise_one_condition(
            experiment_name="initial_utility",
            sweep_parameter="initial_utility",
            sweep_value=initial_utility,
            model_params=model_params,
        )

        rows.append(row)

        print(
            f"  Survival gain:              "
            f"{row['survival_gain']:+.4f} "
            f"(SE {row['se_survival_gain']:.4f})"
        )
        print(
            f"  Mean survivor utility gain: "
            f"{row['mean_utility_survivors_gain']:+.3f}"
        )

    return rows


# ---------------------------------------------------------------------
# CSV OUTPUT
# ---------------------------------------------------------------------

def write_summary_csv(
    rows: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    """Write all sweep conditions to one CSV file."""

    if not rows:
        raise ValueError("No sweep results were generated.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------

def main() -> None:
    print("WCM PARAMETER SWEEPS")
    print("=" * 60)
    print(
        f"Monte Carlo simulations per condition: "
        f"{EXPERIMENT_PARAMS.n_simulations:,}"
    )
    print(
        f"Decisions per simulation: "
        f"{BASE_MODEL_PARAMS.n_repetitions}"
    )
    print(
        f"Master seed: "
        f"{EXPERIMENT_PARAMS.master_seed}"
    )

    rows: list[dict[str, float | int | str]] = []

    rows.extend(
        run_source_accuracy_sweep()
    )

    rows.extend(
        run_initial_utility_sweep()
    )

    write_summary_csv(
        rows,
        OUTPUT_CSV,
    )

    print("\n" + "=" * 60)
    print("Both sweeps complete.")
    print(
        f"Summary results written to:\n"
        f"{OUTPUT_CSV.resolve()}"
    )
    print(
        f"Total parameter conditions: "
        f"{len(rows)}"
    )


if __name__ == "__main__":
    main()