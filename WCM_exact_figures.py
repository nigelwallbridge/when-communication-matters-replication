"""Generate manuscript v21 Figure 3 and the plotted data."""
import csv
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from WCM_exact import BASE, OUT, mechanisms, survival_curve

WITH, WITHOUT = "#2a78d6", "#eb6834"
MECH_SPRING, MECH_NET = "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e2dc"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False,
    "legend.fontsize": 7.5, "axes.titlesize": 9.5, "axes.titlelocation": "left",
})


SERIES = [("none", "No communication", WITHOUT),
          ("springboard", "Risk-taking only", MECH_SPRING),
          ("safety_net", "Risk-reducing only", MECH_NET),
          ("both", "With communication", WITH)]


def _line(ax, x, y, mech, label, colour):
    secondary = mech in ("springboard", "safety_net")
    ax.plot(x, y, color=colour, marker="o" if len(x) < 30 else None,
            ms=3 if secondary else 4, lw=1.3 if secondary else 2,
            ls=(0, (3, 2)) if secondary else "-", alpha=0.8 if secondary else 1,
            zorder=2 if secondary else 3, label=label)


def figure3():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"wspace": 0.30})
    N0s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    res = [mechanisms(replace(BASE, initial_utility=float(n)), n) for n in N0s]

    for mech, label, colour in SERIES:
        _line(ax1, N0s, [r[f"survival_{mech}"] for r in res], mech, label, colour)
    ax1.set_ylim(0, 1); ax1.set_xticks([1, 5, 10, 15, 20])
    ax1.set_xlabel("Initial state $N_0$"); ax1.set_ylabel("Probability")
    ax1.set_title("a   Survival after 50 decisions")
    ax1.legend(loc="lower right")

    T, N0 = 100, 5
    t = np.arange(T + 1)
    curves = {mech: survival_curve(BASE, N0, T, mech) for mech, _, _ in SERIES}
    for mech, label, colour in SERIES:
        _line(ax2, t, curves[mech], mech, label, colour)
    ax2.set_ylim(0, 1); ax2.set_xlim(0, T)
    ax2.set_xlabel("Decision"); ax2.set_ylabel("Probability")
    ax2.set_title(f"b   Survival over time ($N_0$ = {N0})")

    for mech, label, colour in SERIES:
        _line(ax3, N0s, [r[f"expected_N_{mech}"] for r in res], mech, label, colour)
    ax3.set_xticks([1, 5, 10, 15, 20])
    ax3.set_xlabel("Initial state $N_0$"); ax3.set_ylabel("Resources (ruined receivers count as 0)")
    ax3.set_title("c   Expected final resources")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figure3.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    keys = [mech for mech, _, _ in SERIES]
    with (OUT / "figure3_by_initial_state.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["N0"] + [f"{metric}_{key}" for metric in ("survival", "expected_N") for key in keys]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(res)
    with (OUT / "figure3_survival_over_time.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["decision", *keys])
        writer.writerows((int(day), *(curves[key][day] for key in keys)) for day in t)


if __name__ == "__main__":
    figure3()
