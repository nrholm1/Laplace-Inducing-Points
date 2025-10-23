from __future__ import annotations
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from src.train_map import log_joint
from src.lla import compute_curvature_approx
from src.utils import flatten_nn_params
from src.slq import estimate_logdet_slq
from src.scalemodels import TrainState
from src.utils import IPConfig


def _prepare_D(unravel_fn, flat_params, model_type: str) -> int:
    D = sum(x.size for x in jax.tree_util.tree_leaves(unravel_fn(flat_params)))
    if model_type == "regressor":
        D -= 1
    return D


def _make_optimize_alpha_step(
    *,
    map_state_template: TrainState,  # structure/dtypes
    unravel_fn,
    flat_params,
    D: int,
    cfg: IPConfig,
    full_set_size: Optional[int],
):
    """Build a single jitted alpha-optimization step capturing all static bits."""

    def logdet_term(alpha, Z, key):
        # curvature operator on inducing points
        Sz_vp = compute_curvature_approx(
            map_state_template,
            Z,
            alpha=alpha,
            model_type=cfg.model_type,
            full_set_size=full_set_size,
            flat_params=flat_params,
            unravel_fn=unravel_fn,
        )

        # SLQ expects a flat->flat mv function
        def target(u_flat):
            v = Sz_vp(u_flat.reshape(-1))
            return v.reshape(-1)

        M = int(Z.shape[0])
        slq_mv = M if cfg.slq_num_matvecs is None else min(int(cfg.slq_num_matvecs), M)
        return estimate_logdet_slq(
            target,
            D=D,
            M=M,
            key=key,
            slq_samples=int(cfg.slq_samples),
            slq_num_matvecs=slq_mv,
        )

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
        log_alpha_state, map_state, _ = step(log_alpha_state, map_state, Z, batch, subkey)

    return log_alpha_state, map_state
