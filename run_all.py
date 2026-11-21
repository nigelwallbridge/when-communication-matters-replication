"""Run the v21 analyses using the current Python interpreter."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monte-carlo", action="store_true", help="Also run the seeded validation sweep.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    steps = [
        ("WCM_exact.py", "exact_results.txt"),
        ("WCM_exact_robustness.py", "robustness_ruin.txt"),
        ("WCM_exact_figures.py", "figures_log.txt"),
    ]
    if args.monte_carlo:
        steps.append(("WCM_sweep.py", "sweep_log.txt"))
    steps.append(("WCM_numbers.py", "paper_numbers.txt"))

    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    env["PYTHONIOENCODING"] = "utf-8"
    for script, log in steps:
        print(f"{script} -> outputs/{log}", flush=True)
        options = ["--monte-carlo"] if args.monte_carlo and script == "WCM_numbers.py" else []
        with (output / log).open("wb") as stream:
            subprocess.run([sys.executable, script, *options], cwd=root, stdout=stream, env=env, check=True)
    check = ["--monte-carlo"] if args.monte_carlo else []
    subprocess.run([sys.executable, "-m", "verification.check", *check], cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
