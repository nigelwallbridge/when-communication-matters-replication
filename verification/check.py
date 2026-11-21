"""Check regenerated tables, independently propagate Figure 3, and test boundary inputs."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from WCM_exact import BASE, OUT, finite_horizon, first_passage, mechanisms, survival_curve

REF = Path(__file__).resolve().parent / "reference"


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def compare_csv(name):
    expected, actual = rows(REF / name), rows(OUT / name)
    assert len(actual) == len(expected), name
    for a, b in zip(actual, expected, strict=True):
        assert a.keys() == b.keys(), name
        for key in a:
            try:
                value = float(b[key])
            except ValueError:
                assert a[key] == b[key], (name, key)
            else:
                assert math.isclose(float(a[key]), value, abs_tol=1e-10, rel_tol=1e-11), (name, key)


def policy(n, p):
    def utility(up, down):
        if n + min(up, down) <= 0:
            return -math.inf
        return p * math.log(n + up) + (1 - p) * math.log(n + down)
    return utility(6, -4) > utility(-1, 2)


def transitions(n, channel, threshold):
    if n <= threshold:
        return [(0, 1.0)]
    default = policy(n, .60)
    use_report = channel == "both" or channel == ("safety_net" if default else "springboard")
    branches = []
    for good, favourable_report, prob in [(True, True, .42), (True, False, .18),
                                          (False, True, .12), (False, False, .28)]:
        belief = (.33 / .51 if favourable_report else .27 / .49) if use_report else .60
        step = (6 if good else -4) if policy(n, belief) else (-1 if good else 2)
        branches.append((n + step if n + step > threshold else 0, prob))
    return branches


def propagate(n0, horizon, channel, threshold=0):
    pmf = {n0 if n0 > threshold else 0: 1.0}
    survival = [1 - pmf.get(0, 0)]
    for _ in range(horizon):
        following = defaultdict(float)
        for state, mass in pmf.items():
            for nxt, prob in transitions(state, channel, threshold):
                following[nxt] += mass * prob
        pmf = following
        survival.append(1 - pmf.get(0, 0))
    return survival, sum(n * mass for n, mass in pmf.items())


def check_figure():
    channels = ("none", "springboard", "safety_net", "both")
    points = rows(OUT / "figure3_by_initial_state.csv")
    curves = rows(OUT / "figure3_survival_over_time.csv")
    assert [int(r["N0"]) for r in points] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    assert [int(r["decision"]) for r in curves] == list(range(101))
    for channel in channels:
        for row in points:
            survival, resources = propagate(int(row["N0"]), 50, channel)
            assert math.isclose(float(row[f"survival_{channel}"]), survival[-1], abs_tol=1e-11)
            assert math.isclose(float(row[f"expected_N_{channel}"]), resources, abs_tol=1e-11)
        np.testing.assert_allclose([float(r[channel]) for r in curves],
                                   propagate(5, 100, channel)[0], atol=1e-11, rtol=0)
    for ext in ("pdf", "png"):
        assert (OUT / f"figure3.{ext}").stat().st_size > 0


def check_robustness_report():
    reference = (REF / "robustness_ruin.txt").read_text(encoding="utf-8")
    actual = (OUT / "robustness_ruin.txt").read_text(encoding="utf-8")
    reference_cases = reference.split("=== ruin when ")
    actual_cases = actual.split("=== ruin when ")
    assert len(reference_cases) == len(actual_cases) == 3
    for expected_case, actual_case in zip(reference_cases[1:], actual_cases[1:], strict=True):
        # The candidate adds the previously unsampled integer initial states.
        expected_lines = expected_case.splitlines()
        actual_lines = actual_case.splitlines()
        expected_rows = [line for line in expected_lines if line.strip().split(" ")[0].isdigit()]
        assert all(line in actual_lines for line in expected_rows), "Alternate-ruin report changed"


def check_ruin_and_boundaries():
    actual = first_passage(BASE)
    for ch, key in (("none", "without"), ("both", "with")):
        upper = 240
        system = np.eye(upper - 1)
        hit_ruin = np.zeros(upper - 1)
        for n in range(1, upper):
            for nxt, prob in transitions(n, ch, 0):
                if nxt == 0:
                    hit_ruin[n - 1] += prob
                elif nxt < upper:
                    system[n - 1, nxt - 1] -= prob
        independent = np.linalg.solve(system, hit_ruin)[4]
        assert math.isclose(actual[key]["ultimate_ruin"][5], independent, abs_tol=1e-11, rel_tol=0)
    for threshold in (0, 4):
        for n in range(threshold + 1, 21):
            result = finite_horizon(BASE, n, 50, threshold)
            for channel, key in (("none", "without"), ("both", "with")):
                curve, resources = propagate(n, 50, channel, threshold)
                np.testing.assert_allclose(result["surv_w_t" if key == "with" else "surv_o_t"], curve, atol=1e-11, rtol=0)
                assert math.isclose(result[f"expected_N_{key}"], resources, abs_tol=1e-11)
        ruined = finite_horizon(BASE, threshold, 3, threshold)
        assert ruined["surv_w_t"] == [0.0] * 4
        assert ruined["expected_N_with"] == 0 and math.isnan(ruined["survivor_mean_with"])
        assert mechanisms(BASE, threshold, 0, threshold)["survival_both"] == 0
        assert survival_curve(BASE, threshold, 0, ruin_level=threshold).tolist() == [0.0]
        alive = finite_horizon(BASE, threshold + 1, 0, threshold)
        assert alive["surv_w_t"] == [1.0]
        assert alive["expected_N_with"] == threshold + 1 and math.isnan(alive["log_growth_with"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monte-carlo", action="store_true")
    args = parser.parse_args()
    compare_csv("WCM_exact_results.csv")
    compare_csv("WCM_exact_mechanisms.csv")
    if args.monte_carlo:
        compare_csv("WCM_sweep_results.csv")
    reference = (REF / "paper_numbers.txt").read_text(encoding="utf-8").strip()
    generated = (OUT / "paper_numbers.txt").read_text(encoding="utf-8").strip()
    if not args.monte_carlo:
        reference = reference.split("OPTIONAL MONTE CARLO VALIDATION")[0].strip()
    assert generated == reference, "The displayed values differ from the v21 audit reference"
    check_figure()
    check_robustness_report()
    check_ruin_and_boundaries()
    print("Verified v21 numerical references, Figure 3 data, ruin probabilities, and boundary inputs.")


if __name__ == "__main__":
    main()
