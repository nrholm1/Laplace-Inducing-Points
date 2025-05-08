#%%
# fixed_map_plot.py
import os
# avoids MKL/BLAS multithreading issues
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 0) --- Tell Matplotlib to use LaTeX for all text, load your exact font setup, embed TrueType fonts ---
mpl.rcParams.update({
    "text.usetex": True,                     # hand off all text to LaTeX
    "font.family": "serif",                  # use a serif face
    "font.serif": ["Computer Modern Roman"], # explicitly point to the CM font
    "text.latex.preamble":
        r"\usepackage[T1]{fontenc}"+         # proper font encoding
        r"\usepackage{lmodern}" +            # if your doc uses Latin Modern
        r"\usepackage{amsmath}"             # for \text{…} inside math
        # add any other packages you use in your .tex here
    ,
    "pdf.fonttype": 42,                      # embed Type-42 (TrueType) fonts in PDF
    "ps.fonttype": 42,
    "font.size": 12
})

# 1) build an asymmetric, slightly wiggly posterior density  ----------------
def posterior_density(x):
    comp1 = 0.6 * np.exp(-0.5 * ((x - 0.0) / 1.0) ** 2)
    comp2 = 0.25 * np.exp(-0.5 * ((x - 1.8) / 0.7) ** 2)
    ripples = 0.05 * (np.sin(1 + 3 * x) + 1.2)
    return comp1 + comp2 + np.clip(ripples, 0, None) + 0.03

x = np.linspace(-4, 5, 800)
p = posterior_density(x)

# 2) find MAP & Laplace approx --------------------------------------------
idx_max = np.argmax(p)
x0 = x[idx_max]
p0 = p[idx_max]
h = x[1] - x[0]
logp = np.log(p + 1e-12)
second_deriv = (logp[idx_max + 1] - 2*logp[idx_max] + logp[idx_max - 1]) / h**2
sigma = np.sqrt(-1.0 / second_deriv)
laplace = p0 * np.exp(-0.5 * ((x - x0)/sigma)**2)

# 3) plot ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(x, p,      linewidth=2,
        label=r"True posterior $p(\theta\mid\mathcal{D})$",
        color="green")
ax.plot(x, laplace, linestyle="--", linewidth=3,
        label=r"Laplace Approximation $q(\theta)$",
        color="orange")

ax.set_xlabel(r"Parameter $\theta$")
ax.set_ylabel("Density")

ax.legend(frameon=False, loc="upper right")
ax.set_xlim(-4, 5)
ax.set_ylim(-0.01, ax.get_ylim()[1]*1.1)

# vertical mode line + label
ax.axvline(x0, linestyle=":", color="#00a5ee")
ax.text(
    x0, ax.get_ylim()[1],
    r"Posterior mode $\theta_{\mathbf{MAP}}$",
    ha="center", va="top",
    color="#00a5ee",
    bbox=dict(
        facecolor="white",
        alpha=1.0,
        edgecolor="none",
        boxstyle="round,pad=0.2"
    )
)

# clean up axes
ax.set_xticks([]); ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("la_example.pdf", dpi=300, bbox_inches="tight")
