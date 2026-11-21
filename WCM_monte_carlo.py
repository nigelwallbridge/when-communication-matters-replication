"""
Monte Carlo experiment for:

    Wallbridge & Wallbridge
    "When Communication Matters: Quantifying Communicative Value"

This program imports the shared scientific model from
``WCM_model.py`` and repeatedly compares:

1. a receiver acting without communication; and
2. the same receiver acting after Bayesian interpretation of communication.

Within each replication, both conditions face the same environmental sequence.
The experiment now also reports the direct prediction-accuracy comparison added
to the revised standalone program:

- ``source_accuracy``: the sender's configured probability of matching the
  environmental state;
- ``Derived_receiver_accuracy``: the receiver's baseline prediction accuracy,
  derived from its prior and the true environmental probability;
- realised sender and receiver accuracies over active communication rounds.

Run from the directory containing both files:

    WCM_monte_carlo.py and WCM_model.py

Requirements:

    pip install numpy
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from WCM_model import (
    SimulationParameters,
    calculate_Derived_receiver_accuracy,
    receiver_baseline_prediction,
    run_paired_simulation,
)


@dataclass(frozen=True)
class ExperimentParameters:
    """Settings controlling the repeated Monte Carlo experiment."""

    n_simulations: int = 10_000
    master_seed: int = 20260730
    output_csv: str = "outputs/WCM_monte_carlo_results.csv"


@dataclass(frozen=True)
class RunRecord:
    """Run-level outcomes retained for aggregation and export."""

    run: int
    final_without: float
    final_with: float
    survived_without: bool
    survived_with: bool
    decisive_events: int
    active_prediction_rounds: int
    realised_sender_accuracy: float
    realised_receiver_accuracy: float

    @property
    def paired_difference(self) -> float:
        return self.final_with - self.final_without

    @property
    def realised_accuracy_difference(self) -> float:
        """Sender minus receiver realised prediction accuracy."""
        return self.realised_sender_accuracy - self.realised_receiver_accuracy


@dataclass(frozen=True)
class ConditionSummary:
    n_runs: int
    n_survived: int
    survival_rate: float
    mean_final_utility: float
    median_final_utility: float
    standard_deviation: float
    standard_error: float
    mean_final_utility_if_survived: float
    standard_deviation_if_survived: float
    standard_error_if_survived: float


@dataclass(frozen=True)
class PairedSummary:
    mean_difference: float
    median_difference: float
    standard_error_difference: float
    survival_gain: float
    standard_error_survival_gain: float
    communication_higher_count: int
    communication_lower_count: int
    equal_count: int
    communication_higher_rate: float
    communication_lower_rate: float
    equal_rate: float
    mean_decisive_events: float
    runs_with_any_decisive_event: int
    rate_with_any_decisive_event: float


@dataclass(frozen=True)
class AccuracySummary:
    """Aggregated retrospective prediction-accuracy results."""

    total_active_predictions: int
    pooled_realised_sender_accuracy: float
    pooled_realised_receiver_accuracy: float
    pooled_realised_accuracy_difference: float
    mean_run_sender_accuracy: float
    mean_run_receiver_accuracy: float
    mean_run_accuracy_difference: float
    sender_more_accurate_runs: int
    receiver_more_accurate_runs: int
    equal_accuracy_runs: int


@dataclass(frozen=True)
class MonteCarloSummary:
    """Aggregated summaries returned by one Monte Carlo experiment."""

    without_communication: ConditionSummary
    with_communication: ConditionSummary
    paired: PairedSummary
    accuracy: AccuracySummary


def run_experiment(
    model_params: SimulationParameters,
    experiment_params: ExperimentParameters,
) -> list[RunRecord]:
    """Generate all paired Monte Carlo replications reproducibly."""
    master_seed = np.random.SeedSequence(experiment_params.master_seed)
    run_seeds = master_seed.spawn(experiment_params.n_simulations)
    records: list[RunRecord] = []

    for run_index, run_seed in enumerate(run_seeds, start=1):
        result = run_paired_simulation(model_params, run_seed)
        communication_result = result.with_communication

        records.append(
            RunRecord(
                run=run_index,
                final_without=result.without_communication.final_utility,
                final_with=communication_result.final_utility,
                survived_without=result.without_communication.survived,
                survived_with=communication_result.survived,
                decisive_events=communication_result.decisive_event_count,
                active_prediction_rounds=communication_result.active_prediction_count,
                realised_sender_accuracy=communication_result.realised_sender_accuracy,
                realised_receiver_accuracy=communication_result.realised_receiver_accuracy,
            )
        )

    return records


def summarise_condition(
    final_utilities: np.ndarray,
    survived: np.ndarray,
) -> ConditionSummary:
    """Calculate descriptive statistics for one condition."""
    n_runs = int(final_utilities.size)
    n_survived = int(np.sum(survived))
    survival_rate = n_survived / n_runs
    standard_deviation = float(np.std(final_utilities, ddof=1))
    standard_error = standard_deviation / math.sqrt(n_runs)
    if n_survived:
        survivor_utilities = final_utilities[survived]
        mean_if_survived = float(np.mean(survivor_utilities))
        if n_survived > 1:
            sd_if_survived = float(np.std(survivor_utilities, ddof=1))
            se_if_survived = sd_if_survived / math.sqrt(n_survived)
        else:
            sd_if_survived = math.nan
            se_if_survived = math.nan
    else:
        mean_if_survived = math.nan
        sd_if_survived = math.nan
        se_if_survived = math.nan

    return ConditionSummary(
        n_runs=n_runs,
        n_survived=n_survived,
        survival_rate=survival_rate,
        mean_final_utility=float(np.mean(final_utilities)),
        median_final_utility=float(np.median(final_utilities)),
        standard_deviation=standard_deviation,
        standard_error=standard_error,
        mean_final_utility_if_survived=mean_if_survived,
        standard_deviation_if_survived=sd_if_survived,
        standard_error_if_survived=se_if_survived,
    )


def summarise_pairs(records: Iterable[RunRecord]) -> PairedSummary:
    """Summarise within-run utility differences."""
    records = list(records)
    differences = np.asarray([r.paired_difference for r in records], dtype=float)
    decisive_counts = np.asarray([r.decisive_events for r in records], dtype=int)
    survival_differences = np.asarray(
        [
            int(r.survived_with) - int(r.survived_without)
            for r in records
        ],
        dtype=float,
    )
    n_runs = len(records)

    higher = int(np.sum(differences > 0.0))
    lower = int(np.sum(differences < 0.0))
    equal = int(np.sum(differences == 0.0))
    any_decisive = int(np.sum(decisive_counts > 0))

    survival_gain = float(np.mean(survival_differences))
    survival_gain_se = (
        float(np.std(survival_differences, ddof=1)) / math.sqrt(n_runs)
        if n_runs > 1
        else math.nan
    )

    return PairedSummary(
        mean_difference=float(np.mean(differences)),
        median_difference=float(np.median(differences)),
        standard_error_difference=float(np.std(differences, ddof=1))
        / math.sqrt(n_runs),
        survival_gain=survival_gain,
        standard_error_survival_gain=survival_gain_se,
        communication_higher_count=higher,
        communication_lower_count=lower,
        equal_count=equal,
        communication_higher_rate=higher / n_runs,
        communication_lower_rate=lower / n_runs,
        equal_rate=equal / n_runs,
        mean_decisive_events=float(np.mean(decisive_counts)),
        runs_with_any_decisive_event=any_decisive,
        rate_with_any_decisive_event=any_decisive / n_runs,
    )


def summarise_accuracy(records: Iterable[RunRecord]) -> AccuracySummary:
    """Aggregate sender and receiver realised predictive accuracy.

    Pooled accuracy weights each active prediction equally. Mean run accuracy
    gives every simulation run equal weight. Both are reported because runs may
    contain different numbers of active rounds when receivers are ruined early.
    """
    records = list(records)
    active = np.asarray([r.active_prediction_rounds for r in records], dtype=int)
    sender = np.asarray([r.realised_sender_accuracy for r in records], dtype=float)
    receiver = np.asarray([r.realised_receiver_accuracy for r in records], dtype=float)

    valid = (~np.isnan(sender)) & (~np.isnan(receiver)) & (active > 0)
    if not np.any(valid):
        return AccuracySummary(0, *(math.nan,) * 6, 0, 0, 0)

    active_valid = active[valid]
    sender_valid = sender[valid]
    receiver_valid = receiver[valid]
    differences = sender_valid - receiver_valid
    total_active = int(np.sum(active_valid))

    pooled_sender = float(np.sum(sender_valid * active_valid) / total_active)
    pooled_receiver = float(np.sum(receiver_valid * active_valid) / total_active)

    return AccuracySummary(
        total_active_predictions=total_active,
        pooled_realised_sender_accuracy=pooled_sender,
        pooled_realised_receiver_accuracy=pooled_receiver,
        pooled_realised_accuracy_difference=pooled_sender - pooled_receiver,
        mean_run_sender_accuracy=float(np.mean(sender_valid)),
        mean_run_receiver_accuracy=float(np.mean(receiver_valid)),
        mean_run_accuracy_difference=float(np.mean(differences)),
        sender_more_accurate_runs=int(np.sum(differences > 0.0)),
        receiver_more_accurate_runs=int(np.sum(differences < 0.0)),
        equal_accuracy_runs=int(np.sum(differences == 0.0)),
    )


def write_run_records(records: Iterable[RunRecord], output_path: Path) -> None:
    """Write one row per replication for transparency and later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "run",
                "final_utility_without_communication",
                "final_utility_with_communication",
                "survived_without_communication",
                "survived_with_communication",
                "paired_final_utility_difference",
                "decisive_communication_events",
                "active_prediction_rounds",
                "realised_sender_accuracy",
                "realised_receiver_accuracy",
                "realised_accuracy_difference_sender_minus_receiver",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.run,
                    f"{record.final_without:.10g}",
                    f"{record.final_with:.10g}",
                    int(record.survived_without),
                    int(record.survived_with),
                    f"{record.paired_difference:.10g}",
                    record.decisive_events,
                    record.active_prediction_rounds,
                    f"{record.realised_sender_accuracy:.10g}",
                    f"{record.realised_receiver_accuracy:.10g}",
                    f"{record.realised_accuracy_difference:.10g}",
                ]
            )


def run_monte_carlo(
    model_params: SimulationParameters,
    experiment_params: ExperimentParameters,
) -> tuple[MonteCarloSummary, list[RunRecord]]:
    """Run one full Monte Carlo condition and return summaries plus run records.

    This function contains the programmatic interface used by parameter-sweep
    scripts. It preserves the existing scientific model and randomisation while
    making the calculated summaries available without parsing terminal output.
    """
    records = run_experiment(model_params, experiment_params)

    final_without = np.asarray([r.final_without for r in records], dtype=float)
    final_with = np.asarray([r.final_with for r in records], dtype=float)
    survived_without = np.asarray([r.survived_without for r in records], dtype=bool)
    survived_with = np.asarray([r.survived_with for r in records], dtype=bool)

    summary = MonteCarloSummary(
        without_communication=summarise_condition(
            final_without, survived_without
        ),
        with_communication=summarise_condition(
            final_with, survived_with
        ),
        paired=summarise_pairs(records),
        accuracy=summarise_accuracy(records),
    )

    return summary, records


def print_configured_accuracy_comparison(params: SimulationParameters) -> None:
    """Print the direct by-design comparison requested for the revised model."""
    derived_receiver_accuracy = calculate_Derived_receiver_accuracy(params)
    difference = params.source_accuracy - derived_receiver_accuracy

    print("\nCONFIGURED / DERIVED PREDICTION-ACCURACY COMPARISON")
    print("-" * 68)
    print(f"Receiver baseline prediction:          {receiver_baseline_prediction(params)}")
    print(f"Configured source_accuracy:            {params.source_accuracy:10.3f}")
    print(f"Derived_receiver_accuracy:             {derived_receiver_accuracy:10.3f}")
    print(f"Difference (source - receiver):        {difference:+10.3f}")
    if difference > 0.0:
        print("By design, the sender has higher baseline predictive accuracy.")
    elif difference < 0.0:
        print("By design, the receiver has higher baseline predictive accuracy.")
    else:
        print("By design, sender and receiver have equal baseline predictive accuracy.")
    print("-" * 68)


def print_condition_table(
    without: ConditionSummary,
    with_communication: ConditionSummary,
) -> None:
    print("\nCONDITION-LEVEL RESULTS")
    print("-" * 126)
    print(
        f"{'Condition':<24} {'Runs':>8} {'Survived':>10} {'Survival':>10} "
        f"{'Mean final':>12} {'Median':>12} {'SD':>12} {'SE':>12} "
        f"{'Mean if survived':>18} {'SE survivors':>14}"
    )
    print("-" * 126)

    for label, summary in (
        ("Without communication", without),
        ("With communication", with_communication),
    ):
        print(
            f"{label:<24} {summary.n_runs:>8d} {summary.n_survived:>10d} "
            f"{summary.survival_rate:>10.3f} {summary.mean_final_utility:>12.3f} "
            f"{summary.median_final_utility:>12.3f} "
            f"{summary.standard_deviation:>12.3f} {summary.standard_error:>12.3f} "
            f"{summary.mean_final_utility_if_survived:>18.3f} "
            f"{summary.standard_error_if_survived:>14.3f}"
        )

    print("-" * 126)
    print(
        f"{'Difference (with - without)':<24} {'':>8} "
        f"{with_communication.n_survived - without.n_survived:>+10d} "
        f"{with_communication.survival_rate - without.survival_rate:>+10.3f} "
        f"{with_communication.mean_final_utility - without.mean_final_utility:>+12.3f} "
        f"{with_communication.median_final_utility - without.median_final_utility:>+12.3f}"
    )


def print_paired_summary(summary: PairedSummary) -> None:
    print("\nPAIRED WITHIN-RUN UTILITY COMPARISON")
    print("-" * 68)
    print(f"Mean final-utility difference:          {summary.mean_difference:10.3f}")
    print(f"Median final-utility difference:        {summary.median_difference:10.3f}")
    print(f"SE of paired difference:                {summary.standard_error_difference:10.3f}")
    print(f"Paired survival-rate gain:              {summary.survival_gain:+10.4f}")
    print(f"SE of paired survival-rate gain:        {summary.standard_error_survival_gain:10.4f}")
    print("-" * 68)
    print(
        f"Communication produced higher utility: {summary.communication_higher_count:8d} "
        f"({summary.communication_higher_rate:.3f})"
    )
    print(
        f"Communication produced lower utility:  {summary.communication_lower_count:8d} "
        f"({summary.communication_lower_rate:.3f})"
    )
    print(
        f"Final utilities were equal:             {summary.equal_count:8d} "
        f"({summary.equal_rate:.3f})"
    )
    print("-" * 68)
    print(f"Mean decisive events per run:           {summary.mean_decisive_events:10.3f}")
    print(
        f"Runs with at least one decisive event:  {summary.runs_with_any_decisive_event:8d} "
        f"({summary.rate_with_any_decisive_event:.3f})"
    )
    print("-" * 68)


def print_accuracy_summary(summary: AccuracySummary) -> None:
    """Print direct retrospective source-versus-receiver comparison."""
    print("\nREALISED PREDICTION-ACCURACY COMPARISON")
    print("-" * 68)
    print(f"Total active prediction rounds:         {summary.total_active_predictions:10d}")
    print(f"Pooled realised source accuracy:        {summary.pooled_realised_sender_accuracy:10.3f}")
    print(f"Pooled realised receiver accuracy:      {summary.pooled_realised_receiver_accuracy:10.3f}")
    print(f"Pooled difference (source - receiver):  {summary.pooled_realised_accuracy_difference:+10.3f}")
    print("-" * 68)
    print(f"Mean run source accuracy:               {summary.mean_run_sender_accuracy:10.3f}")
    print(f"Mean run receiver accuracy:             {summary.mean_run_receiver_accuracy:10.3f}")
    print(f"Mean run difference:                    {summary.mean_run_accuracy_difference:+10.3f}")
    print("-" * 68)
    print(f"Runs source was more accurate:          {summary.sender_more_accurate_runs:10d}")
    print(f"Runs receiver was more accurate:        {summary.receiver_more_accurate_runs:10d}")
    print(f"Runs with equal realised accuracy:      {summary.equal_accuracy_runs:10d}")
    print("-" * 68)


def main() -> None:
    """Configure, run, summarise, and save the Monte Carlo experiment."""
    model_params = SimulationParameters(
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

    experiment_params = ExperimentParameters(
        n_simulations=10_000,
        master_seed=20260730,
        output_csv="outputs/WCM_monte_carlo_results.csv",
    )

    print("Running paired Monte Carlo experiment...")
    print(f"Replications: {experiment_params.n_simulations:,}")
    print(f"Decisions per replication: {model_params.n_repetitions}")
    print(f"Master seed: {experiment_params.master_seed}")

    print_configured_accuracy_comparison(model_params)
    summary, records = run_monte_carlo(model_params, experiment_params)

    print_condition_table(
        summary.without_communication,
        summary.with_communication,
    )
    print_paired_summary(summary.paired)
    print_accuracy_summary(summary.accuracy)

    output_path = Path(experiment_params.output_csv)
    write_run_records(records, output_path)
    print(f"Run-level results written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()