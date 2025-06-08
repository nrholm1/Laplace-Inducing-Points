import pdb
import jax, jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from evaluate import eval_dataset


def grid_search_alpha(state,
                      Z0,
                      val_loader,
                      full_set_size,
                      model_type,
                      num_mc_samples=30,
                      scalable=True,
                      log10_min=-3,
                      log10_max=2,
                      n_coarse=7,
                      refine=True,
                      rng_key=0,
                      verbose=True):
    # ---------- build the coarse log grid ----------
    alphas = jnp.logspace(log10_min, log10_max, n_coarse)
    rng    = jax.random.PRNGKey(rng_key)

    def val_nll(alpha, rng):
        """Compute NLL on the *validation* loader at this alpha."""
        nll, _ = eval_dataset(state,
                              val_loader,
                              Z0,
                              alpha=float(alpha),
                              full_set_size=full_set_size,
                              model_type=model_type,
                              num_mc_samples=num_mc_samples,
                              scalable=scalable)
        return nll


    # ---------- coarse sweep ----------
    nlls = []
    for a in alphas:
        nlls.append(val_nll(a, rng))
        if verbose:
            print(f"alpha={a:9.3e}  NLL={nlls[-1]:.4f}")
    nlls = jnp.array(nlls)
    best_idx = int(jnp.argmin(nlls))

    # ---------- local refinement ----------
    if refine:
        # choose the best alpha and its nearest neighbour on the *log* axis
        if best_idx == 0:
            a_left,  a_right = alphas[0], alphas[1]
        elif best_idx == len(alphas) - 1:
            a_left,  a_right = alphas[-2], alphas[-1]
        else:
            a_left  = alphas[best_idx - 1]
            a_right = alphas[best_idx + 1]

        # bisect [a_left, a_right] on the log scale
        log_left, log_right = jnp.log10(jnp.array([a_left, a_right]))
        mid = 10 ** ((log_left + log_right) / 2)
        quarter_left  = 10 ** ((3*log_left + log_right) / 4)
        quarter_right = 10 ** ((log_left + 3*log_right) / 4)
        refine_alphas = jnp.array([quarter_left, mid, quarter_right])

        if verbose:
            print("\n-- refinement pass --")
        refine_nlls = []
        for a in refine_alphas:
            refine_nlls.append(val_nll(a, rng))
            if verbose:
                print(f"alpha={a:9.3e}  NLL={refine_nlls[-1]:.4f}")

        # concat and choose the overall best
        all_alphas = jnp.concatenate([alphas, refine_alphas])
        all_nlls   = jnp.concatenate([nlls,   jnp.array(refine_nlls)])
        best_idx   = int(jnp.argmin(all_nlls))

        alpha_best = float(all_alphas[best_idx])
        best_nll   = float(all_nlls[best_idx])
    else:
        alpha_best = float(alphas[best_idx])
        best_nll   = float(nlls[best_idx])

    if verbose:
        print(f"\n>>> selected  alpha* = {alpha_best:9.3e}  "
              f"(val NLL = {best_nll:.4f})")

    return alpha_best