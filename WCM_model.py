"""
Core model for the simulations accompanying:

    Wallbridge & Wallbridge
    "When Communication Matters: Quantifying Communicative Value"

This module contains the scientific model only. It does not print tables, draw
figures, or run the Monte Carlo experiment. Keeping the model separate from the
reporting code ensures that the single-run demonstration and repeated experiment
use exactly the same assumptions and decision rules.

The model compares two otherwise identical receivers:

1. A receiver that acts from a fixed prior and receives no communication.
2. A receiver that observes a sender's expectation, updates by Bayes' theorem,
   and chooses a strategy from the resulting posterior.

Both receivers face the same environmental sequence within each paired run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class SimulationParameters:
    """Parameters defining one simulation run.

    The defaults are the baseline parameters used in the paper (Table 1 and
    Section 4). They are illustrative rather than empirically estimated.
    Objective parameters describe the simulated world as specified by the
    modeller. Subjective parameters describe the receiver's beliefs.
    """

    # Number of sequential decisions made during one run.
    n_repetitions: int = 50

    # Objective probability that an environment is favourable.
    true_probability_favourable: float = 0.60

    # Initial resources, corresponding to N in the manuscript.
    initial_utility: float = 5.0

    # Receiver's fixed prior probability of a favourable environment.
    # The posterior for one round does not become the next round's prior.
    prior_probability_favourable: float = 0.60

    # Payoffs from the two strategies in the two environmental states.
    A: float = 6.0   # Expansive in a favourable environment
    B: float = -4.0  # Expansive in an unfavourable environment
    C: float = -1.0  # Cautious in a favourable environment
    D: float = 2.0   # Cautious in an unfavourable environment

    # Objective probability that the sender's communicated expectation matches
    # the true environmental state. This is assigned by the modeller and is not
    # directly observed by the receiver. It is NOT the probability that the
    # sender says "Favourable".
    #
    # Examples:
    #   0.50 -> the sender is correct half the time.
    #   0.70 -> the sender matches the environment about 70% of the time.
    #   0.30 -> the sender is wrong about 70% of the time and is therefore
    #           systematically misleading.
    source_accuracy: float = 0.70

    # Receiver's subjective communication model. These likelihoods are used in
    # Bayesian updating and need not match the sender's objective accuracy.
    # P(sender says Favourable | environment is Favourable)
    subjective_p_favourable_signal_given_favourable: float = 0.55

    # P(sender says Favourable | environment is Unfavourable)
    subjective_p_favourable_signal_given_unfavourable: float = 0.45


@dataclass
class RunResult:
    """Results from one receiver over one environmental sequence."""

    utility_history: List[float]
    strategy_history: List[str]

    @property
    def final_utility(self) -> float:
        """Resources remaining after the final round."""
        return self.utility_history[-1]

    @property
    def survived(self) -> bool:
        """Whether the receiver finishes with strictly positive resources."""
        return self.final_utility > 0.0


@dataclass
class CommunicationRunResult(RunResult):
    """Additional histories recorded for the communication condition."""

    baseline_strategy_history: List[str]
    sender_expectation_history: List[str]
    posterior_belief_history: List[float]
    decisive_information_history: List[bool]
    receiver_prediction_history: List[str]
    receiver_prediction_correct_history: List[bool]
    sender_prediction_correct_history: List[bool]
    prediction_active_history: List[bool]

    @property
    def decisive_event_count(self) -> int:
        """Number of rounds in which communication changed the strategy."""
        return sum(self.decisive_information_history)

    @property
    def active_prediction_count(self) -> int:
        """Rounds on which both sender and receiver predictions were recorded."""
        return sum(self.prediction_active_history)

    @property
    def realised_sender_accuracy(self) -> float:
        """Sender accuracy over active communication rounds in this run."""
        active = [
            correct
            for correct, is_active in zip(
                self.sender_prediction_correct_history,
                self.prediction_active_history,
            )
            if is_active
        ]
        return float(np.mean(active)) if active else math.nan

    @property
    def realised_receiver_accuracy(self) -> float:
        """Receiver baseline-prediction accuracy over the same active rounds."""
        active = [
            correct
            for correct, is_active in zip(
                self.receiver_prediction_correct_history,
                self.prediction_active_history,
            )
            if is_active
        ]
        return float(np.mean(active)) if active else math.nan


@dataclass
class PairedRunResult:
    """One fair comparison using a shared environmental sequence."""

    environments: np.ndarray
    without_communication: RunResult
    with_communication: CommunicationRunResult


# -----------------------------------------------------------------------------
# Probability, utility, prediction, and decision functions
# -----------------------------------------------------------------------------


def clamp_probability(p: float, epsilon: float = 1e-9) -> float:
    """Keep a probability inside (0, 1) to avoid numerical edge cases."""
    return max(epsilon, min(1.0 - epsilon, p))


def log_utility(value: float) -> float:
    """Return logarithmic utility, with non-positive resources treated as ruin."""
    if value <= 0.0:
        return -math.inf
    return math.log(value)


def expected_log_utility_expansive(
    current_utility: float,
    probability_favourable: float,
    params: SimulationParameters,
) -> float:
    """Expected logarithmic utility of the Expansive strategy."""
    p = clamp_probability(probability_favourable)
    return (
        p * log_utility(current_utility + params.A)
        + (1.0 - p) * log_utility(current_utility + params.B)
    )


def expected_log_utility_cautious(
    current_utility: float,
    probability_favourable: float,
    params: SimulationParameters,
) -> float:
    """Expected logarithmic utility of the Cautious strategy."""
    p = clamp_probability(probability_favourable)
    return (
        p * log_utility(current_utility + params.C)
        + (1.0 - p) * log_utility(current_utility + params.D)
    )


def choose_strategy(
    current_utility: float,
    probability_favourable: float,
    params: SimulationParameters,
) -> str:
    """Choose the strategy with the greater expected logarithmic utility."""
    expansive = expected_log_utility_expansive(
        current_utility, probability_favourable, params
    )
    cautious = expected_log_utility_cautious(
        current_utility, probability_favourable, params
    )
    return "Expansive" if expansive > cautious else "Cautious"


def realised_payoff(
    strategy: str,
    environment_is_favourable: bool,
    params: SimulationParameters,
) -> float:
    """Return the payoff realised by a strategy in the actual environment."""
    if strategy == "Expansive":
        return params.A if environment_is_favourable else params.B
    if strategy == "Cautious":
        return params.C if environment_is_favourable else params.D
    raise ValueError(f"Unknown strategy: {strategy}")


def receiver_baseline_prediction(params: SimulationParameters) -> str:
    """Return the receiver's pre-communication environmental prediction.

    The receiver has no private observation in this model. Its prediction is
    therefore the state assigned the greater probability by its fixed prior.
    This prediction is distinct from its strategy: strategy choice also depends
    on resources, payoffs, and risk.
    """
    if params.prior_probability_favourable >= 0.50:
        return "Favourable"
    return "Unfavourable"


def calculate_Derived_receiver_accuracy(params: SimulationParameters) -> float:
    """Calculate ``Derived_receiver_accuracy`` from existing parameters.

    This is a derived diagnostic, not a separately specified model parameter.
    It uses:

    1. ``prior_probability_favourable`` to identify the receiver's fixed
       baseline prediction; and
    2. ``true_probability_favourable`` to calculate how often that prediction
       is expected to match the environment.

    If the receiver predicts Favourable, the value equals
    ``true_probability_favourable``. If it predicts Unfavourable, it equals
    ``1 - true_probability_favourable``.

    The result is directly comparable with ``source_accuracy`` because both are
    probabilities of matching the true environmental state.
    """
    if receiver_baseline_prediction(params) == "Favourable":
        return params.true_probability_favourable
    return 1.0 - params.true_probability_favourable


# -----------------------------------------------------------------------------
# Environment and communication generation
# -----------------------------------------------------------------------------


def generate_environment_sequence(
    params: SimulationParameters,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one sequence of favourable/unfavourable environments."""
    return rng.random(params.n_repetitions) < params.true_probability_favourable


def generate_sender_expectation(
    environment_is_favourable: bool,
    params: SimulationParameters,
    rng: np.random.Generator,
) -> str:
    """Generate the sender's expectation for the current environment."""
    signal_is_correct = rng.random() < clamp_probability(params.source_accuracy)
    if signal_is_correct:
        return "Favourable" if environment_is_favourable else "Unfavourable"
    return "Unfavourable" if environment_is_favourable else "Favourable"


def bayesian_update_after_communication(
    prior_probability_favourable: float,
    sender_expectation: str,
    params: SimulationParameters,
) -> float:
    """Update the receiver's belief after observing the sender's expectation."""
    prior = clamp_probability(prior_probability_favourable)
    p_message_f_given_f = clamp_probability(
        params.subjective_p_favourable_signal_given_favourable
    )
    p_message_f_given_u = clamp_probability(
        params.subjective_p_favourable_signal_given_unfavourable
    )

    if sender_expectation == "Favourable":
        likelihood_given_favourable = p_message_f_given_f
        likelihood_given_unfavourable = p_message_f_given_u
    elif sender_expectation == "Unfavourable":
        likelihood_given_favourable = 1.0 - p_message_f_given_f
        likelihood_given_unfavourable = 1.0 - p_message_f_given_u
    else:
        raise ValueError(f"Unknown sender expectation: {sender_expectation}")

    numerator = likelihood_given_favourable * prior
    denominator = numerator + likelihood_given_unfavourable * (1.0 - prior)
    if denominator <= 0.0:
        raise ValueError("Bayesian update is undefined for this observation.")

    return clamp_probability(numerator / denominator)


# -----------------------------------------------------------------------------
# Single-run simulation functions
# -----------------------------------------------------------------------------


def run_without_communication(
    environments: np.ndarray,
    params: SimulationParameters,
) -> RunResult:
    """Run a receiver that always acts from its fixed prior belief."""
    current_utility = params.initial_utility
    utility_history = [current_utility]
    strategy_history: List[str] = []

    for environment_is_favourable in environments:
        if current_utility <= 0.0:
            utility_history.append(0.0)
            strategy_history.append("Ruined")
            continue

        strategy = choose_strategy(
            current_utility,
            params.prior_probability_favourable,
            params,
        )
        current_utility = max(
            0.0,
            current_utility
            + realised_payoff(strategy, bool(environment_is_favourable), params),
        )
        utility_history.append(current_utility)
        strategy_history.append(strategy)

    return RunResult(utility_history, strategy_history)


def run_with_communication(
    environments: np.ndarray,
    params: SimulationParameters,
    communication_rng: np.random.Generator,
) -> CommunicationRunResult:
    """Run a receiver that updates after one communication on every active round.

    Prediction accuracy is recorded only while the receiver is active. The
    realised sender and receiver accuracies therefore use exactly the same set
    of rounds, making their retrospective comparison internally consistent.
    """
    current_utility = params.initial_utility
    utility_history = [current_utility]
    strategy_history: List[str] = []
    baseline_strategy_history: List[str] = []
    sender_expectation_history: List[str] = []
    posterior_belief_history: List[float] = []
    decisive_information_history: List[bool] = []
    receiver_prediction_history: List[str] = []
    receiver_prediction_correct_history: List[bool] = []
    sender_prediction_correct_history: List[bool] = []
    prediction_active_history: List[bool] = []

    receiver_prediction = receiver_baseline_prediction(params)

    for environment_is_favourable_raw in environments:
        environment_is_favourable = bool(environment_is_favourable_raw)

        if current_utility <= 0.0:
            utility_history.append(0.0)
            strategy_history.append("Ruined")
            baseline_strategy_history.append("Ruined")
            sender_expectation_history.append("Not applicable")
            posterior_belief_history.append(params.prior_probability_favourable)
            decisive_information_history.append(False)
            receiver_prediction_history.append("Not applicable")
            receiver_prediction_correct_history.append(False)
            sender_prediction_correct_history.append(False)
            prediction_active_history.append(False)
            continue

        baseline_strategy = choose_strategy(
            current_utility,
            params.prior_probability_favourable,
            params,
        )

        sender_expectation = generate_sender_expectation(
            environment_is_favourable, params, communication_rng
        )
        posterior_belief = bayesian_update_after_communication(
            params.prior_probability_favourable,
            sender_expectation,
            params,
        )
        informed_strategy = choose_strategy(
            current_utility,
            posterior_belief,
            params,
        )
        decisive_information = informed_strategy != baseline_strategy

        true_environment = (
            "Favourable" if environment_is_favourable else "Unfavourable"
        )
        receiver_correct = receiver_prediction == true_environment
        sender_correct = sender_expectation == true_environment

        current_utility = max(
            0.0,
            current_utility
            + realised_payoff(informed_strategy, environment_is_favourable, params),
        )

        utility_history.append(current_utility)
        strategy_history.append(informed_strategy)
        baseline_strategy_history.append(baseline_strategy)
        sender_expectation_history.append(sender_expectation)
        posterior_belief_history.append(posterior_belief)
        decisive_information_history.append(decisive_information)
        receiver_prediction_history.append(receiver_prediction)
        receiver_prediction_correct_history.append(receiver_correct)
        sender_prediction_correct_history.append(sender_correct)
        prediction_active_history.append(True)

    return CommunicationRunResult(
        utility_history=utility_history,
        strategy_history=strategy_history,
        baseline_strategy_history=baseline_strategy_history,
        sender_expectation_history=sender_expectation_history,
        posterior_belief_history=posterior_belief_history,
        decisive_information_history=decisive_information_history,
        receiver_prediction_history=receiver_prediction_history,
        receiver_prediction_correct_history=receiver_prediction_correct_history,
        sender_prediction_correct_history=sender_prediction_correct_history,
        prediction_active_history=prediction_active_history,
    )


def run_paired_simulation(
    params: SimulationParameters,
    seed_sequence: np.random.SeedSequence,
) -> PairedRunResult:
    """Run both conditions against one shared environmental sequence."""
    environment_seed, communication_seed = seed_sequence.spawn(2)
    environment_rng = np.random.default_rng(environment_seed)
    communication_rng = np.random.default_rng(communication_seed)

    environments = generate_environment_sequence(params, environment_rng)
    without = run_without_communication(environments, params)
    with_communication = run_with_communication(
        environments, params, communication_rng
    )

    return PairedRunResult(environments, without, with_communication)