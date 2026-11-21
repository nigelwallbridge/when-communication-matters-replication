"""
Selected manuscript v21 values, recomputed from the model.

    Wallbridge & Wallbridge
    "When Communication Matters: Quantifying Communicative Value"

Each line gives the value, where it appears in the paper, and the value as
printed there, so the text can be checked against the code. Run after
WCM_sweep.py if you also want the optional Monte Carlo validation; otherwise that block is skipped.

Run:
    python WCM_numbers.py
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from WCM_exact import (
    BASE, direct_share_by_horizon, finite_horizon, first_passage, mechanisms, policies,
)

A, B, C, D = BASE.A, BASE.B, BASE.C, BASE.D
q = BASE.true_probability_favourable
prior = BASE.prior_probability_favourable
LF = BASE.subjective_p_favourable_signal_given_favourable
LU = BASE.subjective_p_favourable_signal_given_unfavourable
OUT = Path(__file__).resolve().parent / "outputs"


def f(N, p=prior):
    """U_E - U_C at belief p (Equation S4)."""
    return p * np.log((N + A) / (N + C)) + (1 - p) * np.log((N + B) / (N + D))


def p_star(N):
    """Belief threshold (Equation S7)."""
    return np.log((N + D) / (N + B)) / (np.log((N + A) / (N + C)) + np.log((N + D) / (N + B)))


def VO(N, R):
    """Objective communicative value (Equations S20-S22), receiver reads 0.55/0.45."""
    if f(N) < 0:
        return q * R * np.log((N + A) / (N + C)) + (1 - q) * (1 - R) * np.log((N + B) / (N + D))
    return q * (1 - R) * np.log((N + C) / (N + A)) + (1 - q) * R * np.log((N + D) / (N + B))


def H(x):
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def mutual_information(R):
    """Bits of information the source's report carries about the environment."""
    return H(q * R + (1 - q) * (1 - R)) - H(R)


def line(value, where, printed):
    print(f"  {value:<28} {where:<46} paper: {printed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--monte-carlo', action='store_true')
    args = parser.parse_args()
    print("SECTION 2: SINGLE DECISION")
    line(f"{q * A + (1 - q) * B:.2f}", "Table 1 caption, Expansive dN", "2")
    line(f"{q * C + (1 - q) * D:.2f}", "Table 1 caption, Cautious dN", "0.2")
    cross = brentq(f, 4.01, 50)
    line(f"{cross:.2f}", "Fig. 1, Sec. 2(d), S1: prior crossover", "6.44")
    g = np.linspace(cross + 1e-6, 200, 400001)
    line(f"{g[np.argmax(f(g))]:.1f}", "Fig. 1, S1: maximum of strategic stakes", "12.2")
    post_f = LF * prior / (LF * prior + LU * (1 - prior))
    post_u = (1 - LF) * prior / ((1 - LF) * prior + (1 - LU) * (1 - prior))
    line(f"{post_f:.3f} / {post_u:.3f}", "Sec. 2(c), S2: posteriors after F / U", "0.647 / 0.551")
    lo = brentq(lambda N: p_star(N) - post_f, 4.001, 50)
    hi = brentq(lambda N: p_star(N) - post_u, 4.001, 50)
    line(f"({lo:.2f}, {hi:.2f})", "Sec. 2(d), Fig. 2, S2: decisive window", "(5.18, 9.51)")
    line(f"{p_star(6.0):.3f}", "Sec. 3(vii): belief needed at N = 6", "(reasoning: 0.60 < threshold < 0.647)")
    s0, sf, su = policies(6, BASE)
    line(f"{s0} -> {sf}", "Sec. 3(vii): favourable report at N = 6", "Cautious -> Expansive")

    print("\nSECTION 5(a): INFORMATION VS VALUE")
    v = VO(cross - 1e-9, 0.55)
    line(f"{mutual_information(0.55):.4f} bits", "accuracy 0.55 source", "about 0.007 bits")
    line(f"{v:.3f}", "value at the crossover (log units)", "about 0.05")
    line(f"{cross * (math.exp(v) - 1):.2f}", "certainty equivalent at crossover", "about 0.3 units of resource")
    line(f"{mutual_information(0.75):.3f} bits", "accuracy 0.75 source", "0.18 bits")
    line(f"{mutual_information(0.75) / mutual_information(0.55):.0f}x", "ratio of the two", "roughly 25 times")
    line(f"{max(q, 1 - q):.0%}", "receiver's prior-only accuracy", "60%")

    print("\nSECTION 4: REPEATED DECISIONS (exact)")
    decisive = [N for N in range(1, 30) if len(set(policies(N, BASE))) > 1]
    line(str(decisive), "states where communication can act", "N = 6, 7, 8 and 9")
    N0s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    mech = [mechanisms(replace(BASE, initial_utility=float(n)), n) for n in N0s]
    gains = [m["survival_gain"] for m in mech]
    line(f"{min(gains):.3f} to {max(gains):.3f}", "survival gain over initial states", "up to 0.17 (positive at every state)")
    alone = all(m["survival_springboard"] > m["survival_none"] and m["survival_safety_net"] > m["survival_none"]
                and m["survival_both"] > max(m["survival_springboard"], m["survival_safety_net"]) for m in mech)
    line(str(alone), "each channel alone helps; both best", "Fig. 3a dashed lines")
    fp = first_passage(BASE)
    line(f"{fp['without']['ultimate_ruin'][5]:.2f} -> {fp['with']['ultimate_ruin'][5]:.2f}",
         "P(ever ruined) from N0 = 5", "0.41 -> 0.27")
    r5 = finite_horizon(replace(BASE, initial_utility=5.0), 5)
    line(f"{r5['survival_without']:.3f} -> {r5['survival_with']:.3f}", "survival at T = 50 from N0 = 5", "(Fig. 3a)")
    shares = direct_share_by_horizon(replace(BASE, initial_utility=5.0), 5, (5, 10, 20, 50, 100))
    line(" / ".join(f"{shares[T]:.0%}" for T in shares), "direct share, T = 5/10/20/50/100", "63% / 46% / 30% / 15% / 8%")

    print("\nSUPPLEMENTARY NOTE S4: RUIN DEFINED AS N <= 4")
    rob = [finite_horizon(replace(BASE, initial_utility=float(n)), n, ruin_level=4) for n in range(5, 21)]
    line(f"{min(r['survival_gain'] for r in rob):.2f} to {max(r['survival_gain'] for r in rob):.2f}",
         "survival gain, N0 = 5-20", "0.03-0.31")
    ds = [r["direct"] / r["expected_N_gain"] for r in rob]
    line(f"{min(ds):.0%} to {max(ds):.0%}", "direct share at T = 50", "8-13%")

    if not args.monte_carlo:
        return
    print("\nOPTIONAL MONTE CARLO VALIDATION")
    sweep = OUT / "WCM_sweep_results.csv"
    if not sweep.exists():
        print("  (skipped: run WCM_sweep.py first)")
        return
    rows = list(csv.DictReader(sweep.open()))
    def sim(experiment, value):
        r = next(r for r in rows if r["experiment"] == experiment and float(r["sweep_value"]) == value)
        return float(r["survival_gain"]), float(r["se_survival_gain"])
    for label, exp_name, val, params, n0, printed in (
            ("N0 = 5, R = 0.70", "initial_utility", 5.0, BASE, 5, "0.141 vs 0.145 +/- 0.005"),
            ("R = 0.30 (N0 = 5)", "source_accuracy", 0.30, replace(BASE, source_accuracy=0.30), 5, "-0.205 vs -0.205 +/- 0.006"),
            ("N0 = 1, R = 0.70", "initial_utility", 1.0, replace(BASE, initial_utility=1.0), 1, "0.040 vs 0.038 +/- 0.003")):
        exact = finite_horizon(params, n0)["survival_gain"]
        m, se = sim(exp_name, val)
        line(f"{exact:.3f} vs {m:.3f} +/- {se:.3f}", f"survival gain, {label}: exact vs MC", printed)


if __name__ == "__main__":
    main()
