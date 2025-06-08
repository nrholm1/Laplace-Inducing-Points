#%%
# fixed_map_plot.py
import os
# avoids MKL/BLAS multithreading issues
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

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

# 1) Define un-normalized “nice” posterior shape
def unnorm_nice_posterior(x):
    # three-component Gaussian mixture
    comp = (
        0.85 * norm.pdf(x, loc=1.0, scale=0.2)
      + 0.025 * norm.pdf(x, loc=0.5, scale=0.05)
      + 0.125 * norm.pdf(x, loc=1.5, scale=0.1)
    )
    return comp

# 2) Normalize on a finite grid via trapezoidal rule
x = np.linspace(-1, 3, 1000)
u = unnorm_nice_posterior(x)
Z = 1*np.trapezoid(u, x)
p = u / Z

# 3) Laplace around the mode
idx = np.argmax(p)
x0 = x[idx]
p0 = p[idx]
h = x[1] - x[0]
logp = np.log(p + 1e-16)
second_deriv = (logp[idx+1] - 2*logp[idx] + logp[idx-1]) / h**2
sigma = np.sqrt(-1.0 / second_deriv)
# laplace = p0 * np.exp(-0.5 * ((x - x0)/sigma)**2)
laplace = norm.pdf(x, loc=x0, scale=sigma)

# 3) plot ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(x, laplace, linestyle="--", linewidth=3,
        label=r"Laplace Approximation $q(\theta\mid\mathcal{D})$",
        color="orange")
ax.plot(x, p,      linewidth=2,
        label=r"True posterior $p(\theta\mid\mathcal{D})$",
        color="green")

ax.set_xlabel(r"Parameter $\theta$")
ax.set_ylabel("Density")

ax.legend(frameon=False, loc="upper right")
ax.set_xlim(0.2, 2.1)
ax.set_ylim(-0.01, ax.get_ylim()[1]*1.04)

# vertical mode line + label
ax.vlines(x0, ymin=0, ymax=p0, linewidth=2, linestyle=":", color="#00a5ee")
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

ax.scatter(x0, p0, marker='*', s=200, zorder=10, color='#00a5ee')

# clean up axes
ax.set_xticks([]); ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("fig/la_example.pdf", dpi=300, bbox_inches="tight")
