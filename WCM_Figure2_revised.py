"""
WCM_Figure2.py

Figure 2: Strategic stakes and objective expected communicative value.

The two panels hold the receiver's interpretation of the source fixed,
but vary the source's objective accuracy.

Manuscript parameters:
    A = 6
    B = -4
    C = -1
    D = 2
    prior p(E_f) = 0.60

Receiver interpretation:
    P_R(a_F | E_f) = 0.55
    P_R(a_F | E_u) = 0.45

Objective source accuracies:
    Panel (a): R = 0.55
    Panel (b): R = 0.75

The green curve is:
    |f(N,p)| = |U_E(N,p) - U_C(N,p)|

It represents the strategic stakes under the receiver's prior belief.

The pink curve is objective expected communicative value. Whether the
receiver changes strategy is determined by its interpretation of the
source. The objective value of that change is determined using the
source's actual accuracy.

Because receiver interpretation is identical in the two panels, the
decisive interval is identical. Changing objective source accuracy
changes the magnitude and shape of communicative value within it.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ------------------------------------------------------------
# Model parameters
# ------------------------------------------------------------

A = 6.0
B = -4.0
C = -1.0
D = 2.0

# Receiver's prior probability of favourable conditions.
# For the present analysis this is also the objective probability
# of favourable environmental conditions.
p_prior = 0.60
q_objective = 0.60

# Receiver's subjective interpretation of the source.
P_R_aF_given_Ef = 0.55
P_R_aF_given_Eu = 0.45

# Objective source accuracies shown in the two panels.
source_accuracies = [0.55, 0.75]


# ------------------------------------------------------------
# Expected utility difference
# ------------------------------------------------------------

def f(N, p):
    """
    Expected log-utility difference:

        f(N,p) = U_E(N,p) - U_C(N,p)

    Positive values favour Expansive.
    Negative values favour Cautious.
    """
    return (
        p * np.log((N + A) / (N + C))
        + (1.0 - p) * np.log((N + B) / (N + D))
    )


# ------------------------------------------------------------
# Bayesian posteriors used by the receiver
# ------------------------------------------------------------

P_R_aU_given_Ef = 1.0 - P_R_aF_given_Ef
P_R_aU_given_Eu = 1.0 - P_R_aF_given_Eu

p_plus = (
    P_R_aF_given_Ef * p_prior
    / (
        P_R_aF_given_Ef * p_prior
        + P_R_aF_given_Eu * (1.0 - p_prior)
    )
)

p_minus = (
    P_R_aU_given_Ef * p_prior
    / (
        P_R_aU_given_Ef * p_prior
        + P_R_aU_given_Eu * (1.0 - p_prior)
    )
)


# ------------------------------------------------------------
# Simple numerical root finder
# ------------------------------------------------------------

def find_root(func, lower, upper, tol=1e-12, max_iter=200):
    """
    Bisection root finder, avoiding a scipy dependency.
    """
    f_lower = func(lower)
    f_upper = func(upper)

    if f_lower * f_upper > 0:
        raise ValueError("Root is not bracketed.")

    for _ in range(max_iter):
        mid = 0.5 * (lower + upper)
        f_mid = func(mid)

        if abs(f_mid) < tol:
            return mid

        if f_lower * f_mid <= 0:
            upper = mid
            f_upper = f_mid
        else:
            lower = mid
            f_lower = f_mid

    return 0.5 * (lower + upper)


# ------------------------------------------------------------
# Receiver-state boundaries
# ------------------------------------------------------------

# Lower edge:
# a favourable observation places receiver exactly at threshold.
N_lower = find_root(
    lambda N: f(N, p_plus),
    4.000001,
    20.0
)

# Strategy crossover under prior belief.
N_crossover = find_root(
    lambda N: f(N, p_prior),
    4.000001,
    20.0
)

# Upper edge:
# an unfavourable observation places receiver exactly at threshold.
N_upper = find_root(
    lambda N: f(N, p_minus),
    4.000001,
    20.0
)


# ------------------------------------------------------------
# Plotting range and strategic stakes
# ------------------------------------------------------------

N = np.linspace(4.05, 16.0, 2500)

strategic_stakes = np.abs(f(N, p_prior))


# ------------------------------------------------------------
# Objective expected communicative value
# ------------------------------------------------------------

def objective_communicative_value(N, source_accuracy):
    """
    Objective expected value attributable to communication.

    Receiver behaviour is governed by the fixed subjective source
    interpretation above.

    Objective valuation uses the actual source accuracy.

    Below the prior crossover:
        receiver initially chooses Cautious;
        a favourable observation may induce Cautious -> Expansive.

    Above the prior crossover:
        receiver initially chooses Expansive;
        an unfavourable observation may induce Expansive -> Cautious.

    Outside the decisive interval communication does not change
    strategy, so directly attributable communicative value is zero.
    """

    R = source_accuracy
    q = q_objective

    V = np.zeros_like(N)

    # --------------------------------------------------------
    # Cautious -> Expansive after favourable observation
    # --------------------------------------------------------

    mask_C_to_E = (
        (N > N_lower)
        & (N < N_crossover)
    )

    N_left = N[mask_C_to_E]

    V[mask_C_to_E] = (
        q * R
        * np.log((N_left + A) / (N_left + C))
        +
        (1.0 - q) * (1.0 - R)
        * np.log((N_left + B) / (N_left + D))
    )

    # --------------------------------------------------------
    # Expansive -> Cautious after unfavourable observation
    # --------------------------------------------------------

    mask_E_to_C = (
        (N >= N_crossover)
        & (N < N_upper)
    )

    N_right = N[mask_E_to_C]

    V[mask_E_to_C] = (
        q * (1.0 - R)
        * np.log((N_right + C) / (N_right + A))
        +
        (1.0 - q) * R
        * np.log((N_right + D) / (N_right + B))
    )

    return V


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

# Use a locally installed font. Vector exports carry only the drawn glyphs.
for family in ("Arial", "Liberation Sans", "DejaVu Sans"):
    try:
        font_manager.findfont(family, fallback_to_default=False)
    except ValueError:
        continue
    break

plt.rcParams.update({
    "font.family": family,
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "mathtext.fontset": "custom",
    "mathtext.rm": family,
    "mathtext.it": f"{family}:italic",
    "mathtext.fallback": None,
    "pdf.fonttype": 3,
    "svg.fonttype": "path",
})

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.0, 5.65),
    sharex=True,
    sharey=True
)


for panel, (ax, R) in enumerate(
    zip(axes, source_accuracies)
):

    V = objective_communicative_value(N, R)

    # --------------------------------------------------------
    # Light shading of decisive interval
    # --------------------------------------------------------

    ax.axvspan(
        N_lower,
        N_upper,
        color="mediumvioletred",
        alpha=0.06,
        zorder=0
    )

    # --------------------------------------------------------
    # Green: strategic stakes
    # --------------------------------------------------------

    ax.plot(
        N,
        strategic_stakes,
        color="seagreen",
        linewidth=2.5,
        label="Absolute difference in expected utility between strategies"
    )

    # --------------------------------------------------------
    # Pink: objective expected communicative value
    # --------------------------------------------------------

    ax.plot(
        N,
        V,
        color="mediumvioletred",
        linewidth=2.5,
        label="Objective expected communicative value"
    )

    # --------------------------------------------------------
    # Decisive-window boundaries
    # --------------------------------------------------------

    for x in (N_lower, N_upper):
        ax.axvline(
            x,
            color="mediumvioletred",
            linestyle=(0, (5, 4)),
            linewidth=1.0,
            alpha=0.8
        )

    # --------------------------------------------------------
    # Prior strategy crossover
    # --------------------------------------------------------

    ax.axvline(
        N_crossover,
        color="black",
        linestyle="--",
        linewidth=1.1,
        alpha=0.8
    )

    ax.scatter(
        N_crossover,
        0,
        s=35,
        color="black",
        zorder=5
    )

    # --------------------------------------------------------
    # Panel label and source accuracy
    # --------------------------------------------------------

    panel_letter = chr(ord("a") + panel)

    ax.set_title(
        f"({panel_letter}) Source accuracy = {R:.2f}",
        fontsize=11,
        pad=8,
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlim(4, 16)
    ax.set_ylim(-0.01, 0.27)

    ax.set_xlabel(
        r"Receiver resources ($N$)",
        fontsize=11
    )

    ax.tick_params(
        axis="both",
        labelsize=9
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# Shared y-axis label
axes[0].set_ylabel(
    "Expected logarithmic utility",
    fontsize=11
)


# Keep the original axes dimensions (5.378 by 4.266 inches) and panel gap.
# The extra canvas height accommodates the headings and the shared legend.
fig.subplots_adjust(
    left=0.061296296296296314,
    right=0.9808333333333333,
    bottom=1.0244802517361114 / 5.65,
    top=5.29 / 5.65,
    wspace=0.0518065132615374,
)
handles, labels = axes[0].get_legend_handles_labels()
panel_center = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(panel_center, 0.025),
    ncols=1,
    frameon=False,
    fontsize=9,
)

# Write the three versions to the replication package output directory.
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)
output_stem = output_dir / Path(__file__).stem
for extension in (".pdf", ".svg", ".png"):
    fig.savefig(output_stem.with_suffix(extension), dpi=600)

plt.close(fig)


# ------------------------------------------------------------
# Numerical checks
# ------------------------------------------------------------

print(
    "Posterior after favourable observation:   "
    f"p+ = {p_plus:.6f}"
)

print(
    "Posterior after unfavourable observation: "
    f"p- = {p_minus:.6f}"
)

print()

print(
    f"Lower decisive boundary: N = {N_lower:.6f}"
)

print(
    f"Prior crossover:          N = {N_crossover:.6f}"
)

print(
    f"Upper decisive boundary: N = {N_upper:.6f}"
)

print()

for R in source_accuracies:
    V = objective_communicative_value(N, R)

    i_max = np.argmax(V)

    print(
        f"Source accuracy {R:.2f}: "
        f"maximum objective communicative value "
        f"= {V[i_max]:.6f} at N = {N[i_max]:.6f}"
    )
