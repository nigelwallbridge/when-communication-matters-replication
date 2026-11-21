# When Communication Matters: Quantifying Communicative Value

Nigel Wallbridge and Sarenne Wallbridge

This replication package accompanies manuscript version 21. It compares receivers using a fixed prior with receivers that update that prior after a source report. The repeated-decision analysis measures how communication changes survival and expected resources through immediate actions and the resource states those actions lead to.

## Reproduce the analysis

Use Python 3.12. The same runner works with Windows, Linux and macOS Python installations. Run these commands from the directory containing this README.

Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_all.py
.\.venv\Scripts\python.exe run_all.py --monte-carlo
```

Linux or macOS

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_all.py
.venv/bin/python run_all.py --monte-carlo
```

The last command includes the optional Monte Carlo validation and repeats the exact pipeline. Either run writes logs and generated files to `outputs/` and checks the results against the v21 audit references. The Monte Carlo sweep uses 10,000 paired runs per condition, 24 conditions, 50 decisions per run and master seed 20260730. Its results are an independent simulation check and do not enter Figure 3 or the S4 calculations.

## Results and source files

Use the virtual environment's Python interpreter for each command below.

| Manuscript result | Source and output |
| --- | --- |
| Table 1. Payoffs and expected resource changes | The payoffs +6/−4 and −1/+2 are inputs in `WCM_exact.py` and used by `WCM_model.py`. Run `WCM_numbers.py` to recompute the expected changes, 2 and 0.2. There is no table renderer. |
| Figure 1. Strategy utilities and crossover | `WCM_numbers.py` checks the crossover and maximum stakes. The final artwork has no generator in this package. The utility formulas are in manuscript §2, S1 and `WCM_model.py`. |
| Figure 2. Stakes and objective communicative value | Run `python WCM_Figure2_revised.py` to generate the two-panel figure as `outputs/WCM_Figure2_revised.pdf`, `.svg` and `.png` (600 dpi). The manuscript gives the formulas in S1 and S3. `WCM_numbers.py` checks the decisive-window bounds and the value at the prior crossover among other single-decision quantities. |
| Figure 3. Repeated decisions | `WCM_exact_figures.py` writes `figure3.pdf`, `figure3.png`, `figure3_by_initial_state.csv` (panels a and c), and `figure3_survival_over_time.csv` (panel b). The four series are baseline without communication, risk-taking only, risk reduction only and both channels. |
| §4 and S4. Exact survival, eventual ruin and channel comparisons | `WCM_exact.py` writes `WCM_exact_results.csv` and `WCM_exact_mechanisms.csv`. The runner captures its first-passage and eventual-ruin outputs in `exact_results.txt`. |
| Equation S23. Direct and indirect resource gains | `WCM_exact.py` computes both components by summing the one-step drifts over each policy's state distributions. The direct term uses the communicating occupancy distribution and the drift difference. The indirect term uses the occupancy difference and baseline drift. `WCM_numbers.py` reports the direct percentages at 5, 10, 20, 50 and 100 decisions from initial resources 5. The indirect share is the complement. |
| S4. Alternative ruin threshold | `WCM_exact_robustness.py` repeats the calculation for ruin at resources ≤4 and compares it with the baseline. The receiver's decision rule is held fixed. The runner captures both sets of results in `robustness_ruin.txt`. |
| Optional Monte Carlo check | `WCM_sweep.py` uses `WCM_monte_carlo.py` and writes `WCM_sweep_results.csv`. Run `WCM_numbers.py --monte-carlo` after the sweep to print the comparisons. |

Figure 3 and the finite-horizon tables propagate the Markov chain deterministically, subject to floating-point precision. The verification directory includes reference data, regression checks and an independently constructed baseline transition calculation.

## Assumptions and supported inputs

Environmental states are independent across decisions and favourable with probability 0.60 at baseline. Conditional on each true state, the source reports it correctly with probability 0.70, with independent report errors across decisions. The receiver updates the same prior of 0.60 each time, using subjective report likelihoods 0.55 and 0.45. The action maximises expected log utility, with ties going to Cautious. State-dependent communication can change the action at resources 6 through 9. Ruined trajectories have zero resources and make no further decisions.

The exact routines support nonnegative integer initial resources and horizons with the manuscript payoffs and thresholds 0 and 4. The finite-horizon ceiling is above the maximum reachable resource level given the largest step of +6. Horizon zero returns the initial distribution. An initially ruined receiver is absorbed at time zero. Survivor means and conditional log growth are undefined when no receivers survive, and per-decision growth is undefined at horizon zero. Undefined results are represented by NaN. The calibrated and fixed-interpretation accuracy sweeps use the settings in `WCM_exact.py`.

The eventual-ruin calculation solves a first-passage system with ruin at zero and treats reaching resources ≥1480 as escape. At baseline, both policies take the Expansive action at resources ≥10. The chance of returning from the cutoff to resources ≤9 is bounded above by 0.9 raised to the power 1471 (less than 5×10⁻⁶⁸) because the expected one-step multiplier of 0.9 raised to the resource change is 0.92853 away from the window. This bounds the cutoff error separately from floating-point error. Reaching resources 10 is a separate first-passage measure and still permits later returns to the window.

## Citation

See [CITATION.cff](CITATION.cff) for the software citation metadata.

## Licence

The replication package includes the [MIT License](LICENSE). Public release is pending approval by both authors.
