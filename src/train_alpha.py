# train_alpha.py
from __future__ import annotations
from typing import Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp

from src.train_map import log_joint
from src.utils import flatten_nn_params
from src.scalemodels import TrainState
from src.utils import IPConfig

from matfree import (
    decomp,
    funm,
    stochtrace,
)

from src.ggn2 import compute_W_vps


def _prepare_D(unravel_fn, flat_params, model_type: str) -> int:
    D = sum(x.size for x in jax.tree_util.tree_leaves(unravel_fn(flat_params)))
    if model_type == "regressor":
        D -= 1
    return D


def _make_Sz_vp(
    *,
    map_state_template: TrainState,
    Z: jnp.ndarray,
    alpha: float,
    flat_params,
    unravel_fn,
    full_set_size: Optional[int],
) -> callable:
    """
    Matrix-free curvature on Z: v ↦ alpha*v + beta*GGN_Z(v),
    with beta = N / M and GGN_Z(v) = Wz(WTz(v)).
    (Kept for parity with your earlier refactor; not directly used by SLQ below.)
    """
    M = int(Z.shape[0])
    N = M if full_set_size is None else int(full_set_size)
    beta = N / float(M)

    Wz, WTz = compute_W_vps(
        Z,
        flat_params,
        map_state_template.apply_fn,
        unravel_fn,
        mode="memeff",
    )

    def ggn_vp(v):
        return Wz(WTz(v))

    def Sz_vp(v):
        return alpha * v + beta * ggn_vp(v)

    return Sz_vp


def _make_optimize_alpha_step(
    *,
    map_state_template: TrainState,  # structure/dtypes
    unravel_fn,
    flat_params,
    D: int,
    cfg: IPConfig,
    full_set_size: Optional[int],
):
    """
    Matches the SLQ construction in train_inducing:
      - Build Wz, WTz on the inducing set Z
      - Run SLQ on  I + (beta/alpha) * (W^T W)  in the "small" space
      - Add D * log(alpha)
    """

    def logdet_term(alpha, Z, key):
        # Dimensions / scaling
        M = int(Z.shape[0])
        N = M if full_set_size is None else int(full_set_size)
        beta = N / float(M)

        # Build matrix-free W and W^T on inducing set
        Wz, WTz = compute_W_vps(
            Z,
            flat_params,
            map_state_template.apply_fn,
            unravel_fn,
            mode="memeff",
        )

        # Determine inner shape (the "small" space where W^T maps)
        x0 = jnp.zeros((D,), dtype=getattr(flat_params, "dtype", jnp.float32))
        inner_shape = WTz(x0).shape
        d_z = int(np.prod(inner_shape))

        # SLQ target on the small space: u ↦ u + (beta/alpha) * (W^T W u)
        def small_slq_target(u_flat):
            u = u_flat.reshape(inner_shape)
            v_flat = WTz(Wz(u)).reshape(-1)
            return u_flat + (beta / alpha) * v_flat

        # Lanczos/SLQ config
        slq_num_matvecs = M if (cfg.slq_num_matvecs is None) else min(int(cfg.slq_num_matvecs), M)
        v0 = jnp.zeros((d_z,), dtype=x0.dtype)
        tri = decomp.tridiag_sym(slq_num_matvecs)
        problem = funm.integrand_funm_product_logdet(tri)
        # Use the configured number of SLQ samples (probes)
        num_probes = int(cfg.slq_samples)
        sampler = stochtrace.sampler_rademacher(v0, num=num_probes)
        estimator = stochtrace.estimator(problem, sampler=sampler)

        # Estimate logdet(I + (beta/alpha) W^T W) and add D * log(alpha)
        res = estimator(small_slq_target, key)
        return D * jnp.log(alpha) + res

    def loss_and_aux(params, map_state, Z, batch, key):
        log_alpha = params["log_alpha"]
        alpha = jnp.exp(log_alpha)

        neg_log_post, new_bs = log_joint(
            map_state.params,
            map_state.batch_stats,
            map_state,
            batch,
            alpha,
            cfg.model_type,
        )

        prior_normalizer = -0.5 * D * log_alpha
        logdet = logdet_term(alpha, Z, key)

        loss = neg_log_post + prior_normalizer + 0.5 * logdet
        return loss, new_bs

    @jax.jit
    def step(log_alpha_state, map_state, Z, batch, key):
        (loss, new_bs), grad_alpha = jax.value_and_grad(
            loss_and_aux, argnums=0, has_aux=True
        )(log_alpha_state.params, map_state, Z, batch, key)

        new_log_alpha_state = log_alpha_state.apply_gradients(grads=grad_alpha)
        new_map_state = map_state.replace(batch_stats=new_bs)
        return new_log_alpha_state, new_map_state, loss

    return step


def train_alpha(
    map_state: TrainState,
    log_alpha_state: TrainState,
    Z,
    get_batch_fn: Callable[[], tuple],
    *,
    ip_cfg: IPConfig,
    rng=None,
    full_set_size: Optional[int] = None,
):
    """
    Interleaved alpha optimization for `num_steps` batches.
    Returns: (log_alpha_state, map_state)
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    flat_params, unravel_fn = flatten_nn_params(map_state.params)
    D = _prepare_D(unravel_fn, flat_params, model_type=ip_cfg.model_type)

    step = _make_optimize_alpha_step(
        map_state_template=map_state,
        unravel_fn=unravel_fn,
        flat_params=flat_params,
        D=D,
        cfg=ip_cfg,
        full_set_size=full_set_size,
    )

    for _ in range(ip_cfg.alpha_steps_per_call):
        batch = get_batch_fn()
        rng, subkey = jax.random.split(rng)
        log_alpha_state, map_state, _ = step(
            log_alpha_state, map_state, Z, batch, subkey
        )

    return log_alpha_state, map_state
