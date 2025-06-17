import matplotlib as plt


from src.toydata import load_toydata
from src.utils import load_array_checkpoint, load_checkpoint
from src.nplot import make_comparison_figure


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




for m in [8, 16, 32]:
    zinducing = load_array_checkpoint(
        ckpt_dir=args.ckpt_induc,
        name=induc_ckpt_name,
        step=epochs_inducing
    )

(xtrain,ytrain),*_ = load_toydata("xor")
make_comparison_figure(map_model_state, xtrain, zinducing, alpha, matrix_free=False, num_mc_samples=args.num_mc_samples_lla)
plt.savefig(f"fig/xor-evolution-{m}.pdf", dpi=300, bbox_inches="tight")