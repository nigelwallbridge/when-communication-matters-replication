"""
Exact (Markov-chain) analysis of repeated decisions for:

    Wallbridge & Wallbridge, "When Communication Matters"

With integer payoffs and integer initial resources, the receiver's resources N
form a Markov chain on {0, 1, 2, ...}, with 0 absorbing (ruin). Every quantity
estimated by Monte Carlo in WCM_monte_carlo.py can therefore be computed
exactly by propagating the probability distribution over N. The decision rule
and Bayesian update are imported from WCM_model.py, so the policy is identical
to the one used in the simulations.

Quantities computed:

1. Finite-horizon (T = 50) survival, survivor resources and growth rate,
   with and without communication.
2. An exact decomposition of the communication gain in expected final
   resources into
     - a DIRECT term: the sum, along the receiver's trajectory, of the
       single-decision gain from communication at the state it is in; and
     - an INDIRECT term: the gain from being in different states, i.e. the
       part a single-decision analysis cannot see.
3. Springboard and safety-net measures from absorbing-chain first-passage
   problems (infinite horizon).

Run:
    python WCM_exact.py
"""

from __future__ import annotations

import csv
import math
from numbers import Integral
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from WCM_model import (  # noqa: E402
    SimulationParameters,
    bayesian_update_after_communication,
    choose_strategy,
    realised_payoff,
)

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

BASE = SimulationParameters(
    n_repetitions=50,
    true_probability_favourable=0.60,
    initial_utility=5.0,
    prior_probability_favourable=0.60,
    A=6.0, B=-4.0, C=-1.0, D=2.0,
    source_accuracy=0.70,
    subjective_p_favourable_signal_given_favourable=0.55,
    subjective_p_favourable_signal_given_unfavourable=0.45,
)

# Escape threshold for first-passage measures: the first integer state above
# the decisive window (5.18, 9.51), where communication no longer changes the
# decision under the baseline interpretation.
ESCAPE = 10
SLOW_ZONE_TOP = 5


# -----------------------------------------------------------------------------
# Policy and transition structure
# -----------------------------------------------------------------------------


def policies(N: int, p: SimulationParameters) -> tuple[str, str, str]:
    """Strategy without communication, after 'Favourable', after 'Unfavourable'."""
    post_f = bayesian_update_after_communication(
        p.prior_probability_favourable, "Favourable", p)
    post_u = bayesian_update_after_communication(
        p.prior_probability_favourable, "Unfavourable", p)
    return (choose_strategy(N, p.prior_probability_favourable, p),
            choose_strategy(N, post_f, p),
            choose_strategy(N, post_u, p))


MECHANISMS = ("none", "springboard", "safety_net", "both")


def outcomes(N: int, p: SimulationParameters, communication: bool | str):
    """List of (probability, payoff) pairs for one decision at state N.

    ``communication`` is True/"both" (receiver uses the source at every state),
    False/"none" (never), "springboard" (only where its default strategy is
    Cautious, i.e. below the prior crossover, where the only possible switch is
    Cautious -> Expansive) or "safety_net" (only where its default is Expansive,
    where the only possible switch is Expansive -> Cautious). In the manuscript
    these are the risk-taking and risk-reducing responses respectively."""
    q, R = p.true_probability_favourable, p.source_accuracy
    s0, sf, su = policies(N, p)
    if communication == "both":
        communication = True
    elif communication == "none":
        communication = False
    elif communication in ("springboard", "safety_net"):
        default = "Cautious" if communication == "springboard" else "Expansive"
        communication = s0 == default
    out = []
    for env_f, pe in ((True, q), (False, 1 - q)):
        if not communication:
            out.append((pe, realised_payoff(s0, env_f, p)))
            continue
        # Source reports the true state with probability R.
        p_says_f = R if env_f else 1 - R
        out.append((pe * p_says_f, realised_payoff(sf, env_f, p)))
        out.append((pe * (1 - p_says_f), realised_payoff(su, env_f, p)))
    return out


def transition_matrix(p: SimulationParameters, communication: bool | str, M: int,
                      ruin_level: int = 0) -> np.ndarray:
    """Row-stochastic matrix on states 0..M. Any state <= ruin_level counts as
    ruin and is sent to the absorbing state 0 (ruin_level = 0 is the model of
    the main text). States that would exceed M are clipped to M (choose M large
    enough that this never matters for the horizon used)."""
    P = np.zeros((M + 1, M + 1))
    for N in range(0, ruin_level + 1):
        P[N, 0] = 1.0
    for N in range(ruin_level + 1, M + 1):
        for prob, pay in outcomes(N, p, communication):
            nxt = int(min(M, N + pay))
            P[N, 0 if nxt <= ruin_level else nxt] += prob
    return P


def drift(p: SimulationParameters, communication: bool, M: int,
          ruin_level: int = 0) -> np.ndarray:
    """Expected one-step change in resources at each state, including the
    loss of any remaining resources on ruin (0 once ruined)."""
    P = transition_matrix(p, communication, M, ruin_level)
    states = np.arange(M + 1)
    g = P @ states - states
    g[: ruin_level + 1] = 0.0
    return g


def one_step_ruin(p: SimulationParameters, communication: bool, M: int,
                  ruin_level: int = 0) -> np.ndarray:
    r = np.zeros(M + 1)
    for N in range(ruin_level + 1, M + 1):
        r[N] = sum(pr for pr, pay in outcomes(N, p, communication) if N + pay <= ruin_level)
    return r


# -----------------------------------------------------------------------------
# Finite-horizon analysis
# -----------------------------------------------------------------------------


def initial_distribution(N0: int, T: int, ruin_level: int) -> np.ndarray:
    """Initial mass for an integer resource process with an absorbing ruin state."""
    if any(not isinstance(value, Integral) or value < 0 for value in (N0, T, ruin_level)):
        raise ValueError("N0, T and ruin_level must be nonnegative integers")
    M = max(N0 + 6 * T + 1, ruin_level)
    pi = np.zeros(M + 1)
    pi[0 if N0 <= ruin_level else N0] = 1.0
    return pi


def finite_horizon(p: SimulationParameters, N0: int, T: int | None = None,
                   ruin_level: int = 0) -> dict:
    T = p.n_repetitions if T is None else T
    pw = initial_distribution(N0, T, ruin_level)
    M = len(pw) - 1  # unreachable ceiling under the manuscript payoffs
    states = np.arange(M + 1)
    logN = np.where(states > 0, np.log(np.maximum(states, 1)), 0.0)

    Pw, Po = transition_matrix(p, True, M, ruin_level), transition_matrix(p, False, M, ruin_level)
    gw, go = drift(p, True, M, ruin_level), drift(p, False, M, ruin_level)

    po = pw.copy()
    direct = indirect = 0.0
    surv_w_t, surv_o_t = [1 - pw[0]], [1 - po[0]]
    for _ in range(T):
        direct += pw @ (gw - go)
        indirect += (pw - po) @ go
        pw, po = pw @ Pw, po @ Po
        surv_w_t.append(1 - pw[0]); surv_o_t.append(1 - po[0])

    def summary(pi):
        S = 1 - pi[0]
        mean_surv = (states[1:] @ pi[1:]) / S if S > 0 else math.nan
        log_growth = ((logN[1:] @ pi[1:]) / S - math.log(N0)) / T if S > 0 and T else math.nan
        return S, mean_surv, log_growth, states @ pi

    Sw, mw, lw, Ew = summary(pw)
    So, mo, lo, Eo = summary(po)
    assert abs((Ew - Eo) - (direct + indirect)) < 1e-8, "decomposition identity failed"
    return dict(
        N0=N0, R=p.source_accuracy,
        interp=p.subjective_p_favourable_signal_given_favourable,
        survival_with=Sw, survival_without=So, survival_gain=Sw - So,
        survivor_mean_with=mw, survivor_mean_without=mo,
        log_growth_with=lw, log_growth_without=lo, log_growth_gain=lw - lo,
        expected_N_with=Ew, expected_N_without=Eo, expected_N_gain=Ew - Eo,
        direct=direct, indirect=indirect,
        surv_w_t=surv_w_t, surv_o_t=surv_o_t,
    )


def mechanisms(p: SimulationParameters, N0: int, T: int | None = None,
               ruin_level: int = 0) -> dict:
    """Final survival and expected resources for receivers that use
    communication never, only below the crossover (springboard), only above it
    (safety net), or everywhere; plus a Shapley split of the survival and
    resource gains between the two mechanisms. With two mechanisms the Shapley
    share of the springboard is the average of its marginal contribution when
    added first and when added second; the shares sum to the total gain."""
    T = p.n_repetitions if T is None else T
    pi0 = initial_distribution(N0, T, ruin_level)
    M = len(pi0) - 1
    states = np.arange(M + 1)
    out = {"N0": N0}
    for mech in MECHANISMS:
        pi = pi0.copy()
        P = transition_matrix(p, mech, M, ruin_level)
        for _ in range(T):
            pi = pi @ P
        out[f"survival_{mech}"] = 1 - pi[0]
        out[f"expected_N_{mech}"] = states @ pi
    for key in ("survival", "expected_N"):
        none, sp, sn, both = (out[f"{key}_{m}"] for m in MECHANISMS)
        share_sp = 0.5 * ((sp - none) + (both - sn))
        out[f"{key}_gain"] = both - none
        out[f"{key}_shapley_springboard"] = share_sp
        out[f"{key}_shapley_safety_net"] = (both - none) - share_sp
    return out


def survival_curve(p: SimulationParameters, N0: int, T: int, mechanism: bool | str = True,
                   ruin_level: int = 0) -> np.ndarray:
    """Probability of not yet being ruined after t = 0..T decisions."""
    pi = initial_distribution(N0, T, ruin_level)
    P = transition_matrix(p, mechanism, len(pi) - 1, ruin_level)
    S = [1 - pi[0]]
    for _ in range(T):
        pi = pi @ P
        S.append(1 - pi[0])
    return np.array(S)


def direct_share_by_horizon(p: SimulationParameters, N0: int, horizons) -> dict:
    """Share of the expected-resource gain captured by the direct term (S23)."""
    out = {}
    for T in horizons:
        r = finite_horizon(p, N0, T=T)
        out[T] = r["direct"] / r["expected_N_gain"] if r["expected_N_gain"] else math.nan
    return out


# -----------------------------------------------------------------------------
# First-passage (infinite-horizon) measures
# -----------------------------------------------------------------------------


def absorbing(P: np.ndarray, transient: list[int], targets: list[int]):
    """For a chain with the given transient states, return
    (P(hit targets before any other absorbing state), E[time to absorption],
     E[time | hit targets]) for each transient start."""
    Q = P[np.ix_(transient, transient)]
    I = np.eye(len(transient))
    Nfund = np.linalg.inv(I - Q)
    b = P[np.ix_(transient, targets)].sum(axis=1)
    h = Nfund @ b                                  # hitting probability
    t = Nfund @ np.ones(len(transient))            # expected absorption time
    th = Nfund @ h                                 # E[tau * 1{hit}]  (h-transform)
    return h, t, np.where(h > 0, th / np.maximum(h, 1e-300), np.nan)


def first_passage(p: SimulationParameters, ruin_level: int = 0) -> dict:
    M = 200
    out = {}
    for comm in (True, False):
        P = transition_matrix(p, comm, M, ruin_level)
        # Springboard: from below the window, reach N >= ESCAPE before ruin.
        transient = list(range(1, ESCAPE))
        h, t, th = absorbing(P, transient, list(range(ESCAPE, M + 1)))
        # Safety net: from inside the window (6..9), fall into the slow zone
        # (N <= 5) before escaping upward.
        win = list(range(SLOW_ZONE_TOP + 1, ESCAPE))
        hs, _, _ = absorbing(P, win, list(range(0, SLOW_ZONE_TOP + 1)))
        # Approximate eventual ruin by treating N >= 1480 as escape (see README).
        Mbig = 1500
        Pb = transition_matrix(p, comm, Mbig, ruin_level)
        tr = list(range(1, Mbig - 20))
        ruin, _, _ = absorbing(Pb, tr, [0])
        out["with" if comm else "without"] = dict(
            escape_prob=dict(zip(transient, h)),
            escape_time=dict(zip(transient, th)),
            fall_prob=dict(zip(win, hs)),
            ultimate_ruin={N: ruin[N - 1] for N in range(1, 21)},
        )
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    M = 30
    print("PER-STATE POLICY AND DRIFT (baseline: R=0.70, interpretation 0.55/0.45)")
    print(" N   no-comm  after-F   after-U   drift_without  drift_with  ruin1_w  ruin1_o")
    gw, go = drift(BASE, True, M), drift(BASE, False, M)
    rw, ro = one_step_ruin(BASE, True, M), one_step_ruin(BASE, False, M)
    for N in range(1, 16):
        s0, sf, su = policies(N, BASE)
        print(f"{N:2d}   {s0:9s} {sf:9s} {su:9s} {go[N]:+8.2f}       {gw[N]:+8.2f}    "
              f"{rw[N]:.2f}     {ro[N]:.2f}")
    assert np.allclose(rw, ro), "communication changes one-step ruin somewhere"
    print("One-step ruin probability is identical with and without communication "
          "at every state:\nthe entire survival effect is mediated by state.\n")

    rows = []
    print("INITIAL-STATE SWEEP (T=50, exact)")
    print("N0  S_w    S_o    dS     g_w     g_o     dg      E[N]gain direct indirect  %indirect")
    for N0 in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
        r = finite_horizon(replace(BASE, initial_utility=float(N0)), N0)
        rows.append(dict(experiment="initial_state", **r))
        share = r["indirect"] / r["expected_N_gain"] if r["expected_N_gain"] else math.nan
        print(f"{N0:2d}  {r['survival_with']:.3f}  {r['survival_without']:.3f}  "
              f"{r['survival_gain']:+.3f}  {r['log_growth_with']:.4f}  "
              f"{r['log_growth_without']:.4f}  {r['log_growth_gain']:+.4f}  "
              f"{r['expected_N_gain']:+7.2f}  {r['direct']:+6.2f}  {r['indirect']:+6.2f}   "
              f"{100 * share:5.1f}%")

    print("\nSOURCE-ACCURACY SWEEP at N0=5 (receiver interpretation fixed 0.55/0.45; "
          "and calibrated receiver)")
    print("R      dS_fixed  dg_fixed   dS_calib  dg_calib")
    for R in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.99]:
        fx = finite_horizon(replace(BASE, source_accuracy=R), 5)
        ca = finite_horizon(replace(
            BASE, source_accuracy=R,
            subjective_p_favourable_signal_given_favourable=R,
            subjective_p_favourable_signal_given_unfavourable=1 - R), 5)
        rows.append(dict(experiment="accuracy_fixed_interp", **fx))
        rows.append(dict(experiment="accuracy_calibrated", **ca))
        print(f"{R:.2f}   {fx['survival_gain']:+.3f}    {fx['log_growth_gain']:+.4f}    "
              f"{ca['survival_gain']:+.3f}    {ca['log_growth_gain']:+.4f}")

    fp = first_passage(BASE)
    print("\nSPRINGBOARD: P(reach N>=10 before ruin) and E[time | escape]")
    print(" N0   P_w    P_o    T_w    T_o")
    for N in range(1, ESCAPE):
        w, o = fp["with"], fp["without"]
        print(f"{N:3d}  {w['escape_prob'][N]:.3f}  {o['escape_prob'][N]:.3f}  "
              f"{w['escape_time'][N]:5.1f}  {o['escape_time'][N]:5.1f}")
    print("\nSAFETY NET: P(fall to N<=5 before reaching N>=10), starting in the window")
    for N in range(SLOW_ZONE_TOP + 1, ESCAPE):
        print(f"{N:3d}  with {fp['with']['fall_prob'][N]:.3f}   "
              f"without {fp['without']['fall_prob'][N]:.3f}")
    print("\nULTIMATE RUIN PROBABILITY (upper escape boundary N >= 1480)")
    for N in [1, 3, 5, 6, 8, 10, 15, 20]:
        print(f"{N:3d}  with {fp['with']['ultimate_ruin'][N]:.3f}   "
              f"without {fp['without']['ultimate_ruin'][N]:.3f}")

    print("\nMECHANISMS (T=50): survival and expected final resources by receiver")
    print("N0  | survival: none  r-take  r-red both | E[N_T]: none  r-take r-red both "
          "| survival gain share: risk-taking / risk-reducing")
    mech_rows = []
    for N0 in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
        m = mechanisms(replace(BASE, initial_utility=float(N0)), N0)
        mech_rows.append(m)
        g = m["survival_gain"]
        print(f"{N0:2d}  |  " + "  ".join(f"{m[f'survival_{k}']:.3f}" for k in MECHANISMS) + " | "
              + " ".join(f"{m[f'expected_N_{k}']:6.1f}" for k in MECHANISMS)
              + f" | {m['survival_shapley_springboard'] / g:4.0%} / {m['survival_shapley_safety_net'] / g:4.0%}")
    with (OUT / "WCM_exact_mechanisms.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mech_rows[0]))
        w.writeheader(); w.writerows(mech_rows)

    keys = [k for k in rows[0] if k not in ("surv_w_t", "surv_o_t")]
    with (OUT / "WCM_exact_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nWritten: {OUT / 'WCM_exact_results.csv'}")


if __name__ == "__main__":
    main()
