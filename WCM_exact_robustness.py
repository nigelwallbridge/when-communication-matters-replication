"""
Robustness of the Section 4 results to the definition of ruin.

Main text: ruin = resources reach 0. In that model all ruin happens at N = 1,
where both strategies have expected log-utility -inf and the tie-break picks
Cautious. Here ruin is instead defined as falling to N <= 4, the admissible
boundary L of Equation 2.1, so that ruin arises from Expansive losses
(e.g. 7 - 4 = 3) rather than from Cautious losses at N = 1. The receiver's
decision rule is unchanged: it does not know about the threshold.

Run:
    python WCM_exact_robustness.py
"""

from dataclasses import replace

import numpy as np

from WCM_exact import BASE, finite_horizon, first_passage, mechanisms, one_step_ruin

RUIN_LEVELS = (0, 4)
N0S = range(5, 21)


def main() -> None:
    for L in RUIN_LEVELS:
        rw, ro = one_step_ruin(BASE, True, 30, L), one_step_ruin(BASE, False, 30, L)
        diff = [N for N in range(1, 31) if not np.isclose(rw[N], ro[N])]
        print(f"\n=== ruin when N <= {L} ===")
        print("states where communication changes one-step ruin probability:",
              diff or "none", *(f"  N={N}: {ro[N]:.2f} -> {rw[N]:.2f}" for N in diff))
        print("N0   S_without  S_with  dS      dlog-growth  E[N]gain  %indirect")
        for N0 in N0S:
            r = finite_horizon(replace(BASE, initial_utility=float(N0)), N0, ruin_level=L)
            share = 100 * r["indirect"] / r["expected_N_gain"]
            print(f"{N0:3d}  {r['survival_without']:.3f}      {r['survival_with']:.3f}   "
                  f"{r['survival_gain']:+.3f}  {r['log_growth_gain']:+.4f}      "
                  f"{r['expected_N_gain']:+7.2f}   {share:5.1f}%")
        fp = first_passage(BASE, ruin_level=L)
        print("leave window upward: P(reach >=10 before ruin), with/without:",
              "  ".join(f"N={n}: {fp['with']['escape_prob'][n]:.3f}/{fp['without']['escape_prob'][n]:.3f}"
                        for n in (5, 6, 7)))
        print("leave window downward: P(reach <=5 before >=10), with/without:",
              "  ".join(f"N={n}: {fp['with']['fall_prob'][n]:.3f}/{fp['without']['fall_prob'][n]:.3f}"
                        for n in (6, 7, 8, 9)))
        print("survival-gain share risk-taking / risk-reducing:",
              "  ".join(f"N0={n}: {m['survival_shapley_springboard'] / m['survival_gain']:.0%}/"
                        f"{m['survival_shapley_safety_net'] / m['survival_gain']:.0%}"
                        for n in N0S
                        for m in [mechanisms(replace(BASE, initial_utility=float(n)), n, ruin_level=L)]))


if __name__ == "__main__":
    main()
