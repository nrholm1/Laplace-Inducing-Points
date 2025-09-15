#%%
# kl_gaussian_demo.py
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"      # optional, avoids BLAS thread storms

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0) Matplotlib style – identical to your example file
# ----------------------------------------------------------------------
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": (
        r"\usepackage[T1]{fontenc}"
        r"\usepackage{lmodern}"
        r"\usepackage{amsmath}"
    ),
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 14
})

# ----------------------------------------------------------------------
# 1) Two Gaussian PDFs ---------------------------------------------------
def gaussian_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )

mu_p, sigma_p =  0.0, 1.0
mu_q, sigma_q =  1.0, 1.0

x = np.linspace(-4, 4, 1200)
p = gaussian_pdf(x, mu_p, sigma_p)
q = gaussian_pdf(x, mu_q, sigma_q)

# ----------------------------------------------------------------------
# 2) KL integrands & numerical values -----------------------------------
eps = 1e-16                  # numerical safety-net
kl_pq_int = p * np.log((p + eps) / (q + eps))   #  p‖q
kl_qp_int = q * np.log((q + eps) / (p + eps))   #  q‖p
kl_pq = np.trapz(kl_pq_int, x)
kl_qp = np.trapz(kl_qp_int, x)

# ----------------------------------------------------------------------
# 3) Figure --------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))

# p_color = "#2b4f9c"
# q_color = "#9b3156"
# p_color = "green"
p_color = "#00a5ee"
q_color = "orange"
p_linestyle = None#":"
q_linestyle = "--"

# 3a) PDFs ---------------------------------------------------------------
ax = axs[0]
ax.plot(x, p, label=r"$p(x)$", color=p_color, lw=2.5, linestyle=p_linestyle)
ax.plot(x, q, label=r"$q(x)$", color=q_color, lw=2.5, linestyle=q_linestyle)
ax.set_xlabel(r"$x$")
ax.set_ylabel("Density")
ax.set_title(r"Gaussian PDFs")
ax.legend(frameon=False)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0, 1.08*max(p.max(), q.max()))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(direction="out")
ax.set_xticks([0]); ax.set_yticks([])

# 3b) KL(P||Q) -----------------------------------------------------------
ax = axs[1]
ax.plot(x, kl_pq_int, color=p_color, lw=2, linestyle=p_linestyle)
ax.fill_between(x, 0, kl_pq_int, where=kl_pq_int>=0, color=p_color, alpha=0.25)
ax.fill_between(x, 0, kl_pq_int, where=kl_pq_int<=0, color=p_color, alpha=0.25)
ax.axhline(0, ls=":", lw=1, color="black")
ax.axvline(0, ls=":", lw=1, color="black")
ax.set_ylabel("KL")
ax.set_xlabel(r"$x$")
ax.set_title(
    # r"$p(x)\log\!\frac{p(x)}{q(x)}$"+"\n"
    # + 
    r"$\text{KL}[P~\!\|\!~Q]=%.1f$" % kl_pq
    # r"$\text{KL}[P~\!\|\!~Q]$"
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(direction="out")
ax.set_xticks([0]); ax.set_yticks([0])

# 3c) KL(Q||P) -----------------------------------------------------------
ax = axs[2]
ax.plot(x, kl_qp_int, color=q_color, lw=2, linestyle=q_linestyle)
ax.fill_between(x, 0, kl_qp_int, where=kl_qp_int>=0, color=q_color, alpha=0.25)
ax.fill_between(x, 0, kl_qp_int, where=kl_qp_int<=0, color=q_color, alpha=0.25)
ax.axhline(0, ls=":", lw=1, color="black")
ax.axvline(0, ls=":", lw=1, color="black")
ax.set_title(
    # r"$q(x)\log\!\frac{q(x)}{p(x)}$"+"\n"
    # + 
    r"$\text{KL}[Q~\!\|\!~P]=%.2f$" % kl_qp
    # r"$\text{KL}[Q~\!\|\!~P]$"
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# ax.spines["left"].set_visible(False)
# ax.spines["bottom"].set_visible(False)
ax.tick_params(direction="out")
ax.set_xticks([0]); ax.set_yticks([0])
ax.set_xlabel(r"$x$")

fig.tight_layout()
plt.savefig("fig/kl_example.pdf")
plt.show()
#%%