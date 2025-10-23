# src/train_inducing.py
from __future__ import annotations

from dataclasses import dataclass
import pdb
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from matfree import stochtrace as matfree_stochtrace

from src.scalemodels import TrainState
from src.train_alpha import train_alpha
from src.lla import compute_curvature_approx_dense, compute_curvature_approx
from src.ggn import compute_W_vps
from src.utils import flatten_nn_params, IPConfig
from src.slq import estimate_logdet_slq
from src.sampling2 import get_conditional_theta_sampler



def _mask_Z(Z, mask_bool):
    Zf = Z.astype(float)
    return jnp.where(
        mask_bool[(...,) + (None,) * (Z.ndim - 1)],
        Zf,
        jax.lax.stop_gradient(Zf),
    )

def _mask_from_indices(M, batch_idx, dtype=jnp.bool_):
    mask = jnp.zeros((M,), dtype=dtype)
    return mask.at[batch_idx].set(True)


@dataclass
class Shapes:
    D: int  # flattened parameter dimension


def _objective_mf(
    Z,
    X,
    state,
    alpha: float,
    *,
    ip_cfg: IPConfig,
    shapes: Shapes,
    flat_params,
    unravel_fn,
    key,
    full_set_size: int,
):
    """
    Scalable objective using stochastic trace + SLQ.
    """
    
    key_trace,key_slq,key_mask = jax.random.split(key, 3)
    
    if full_set_size is None:
        raise ValueError("full_set_size must be provided for matrix-free objective.")
    N = full_set_size
    M = Z.shape[0]
    beta = N / M

    # Minibatch subset of inducing points
    # ip_batch_size = max(1, int(M * ip_cfg.ip_batch_frac))
    # B = jax.random.permutation(key_mask, M)[:ip_batch_size]
    # batch_mask = _mask_from_indices(M, B)
    # Z_eff = _mask_Z(Z, batch_mask)
    Z_eff = Z
    # pdb.set_trace()

    # Curvature on data
    ggn_real = compute_curvature_approx(
        state,
        X,
        alpha=alpha,
        model_type=ip_cfg.model_type,
        flat_params=flat_params,
        unravel_fn=unravel_fn,
        full_set_size=N,
    )

    # W(V)Ps from inducing points
    Wz, WzT = compute_W_vps(
        state,
        Z_eff,
        model_type=ip_cfg.model_type,
        flat_params=flat_params,
        unravel_fn=unravel_fn,
        full_set_size=None,
    )

    # Dimensions for SLQ space
    D = shapes.D
    x0 = jnp.zeros((D,), dtype=getattr(flat_params, "dtype", jnp.float32))
    inner_shape = WzT(x0).shape
    d_z = int(np.prod(inner_shape))

    # Stochastic trace
    theta_sampler = get_conditional_theta_sampler(
        Z_eff, alpha, beta, state, atol=1e-4, btol=1e-4, ctol=1e-5
    )
    integrand = matfree_stochtrace.integrand_trace()
    sampler = lambda __key: theta_sampler(__key, num_samples=ip_cfg.st_samples)
    estimate = matfree_stochtrace.estimator(integrand, sampler)
    trace_term = estimate(lambda v: ggn_real(v), key_trace)

    # SLQ on I + (beta/alpha) W^T W
    slq_num_matvecs = M if ip_cfg.slq_num_matvecs is None else min(ip_cfg.slq_num_matvecs, M)

    def small_slq_target(u_flat):
        u = u_flat.reshape(inner_shape)
        v_flat = WzT(Wz(u)).reshape(-1)
        return u_flat + (beta / alpha) * v_flat

    res = estimate_logdet_slq(
        small_slq_target,
        D=d_z,
        M=M,
        key=key_slq,
        slq_samples=ip_cfg.slq_samples,
        slq_num_matvecs=slq_num_matvecs,
    )
    logdet_term = D * jnp.log(alpha) + res
    return trace_term + logdet_term


def _objective_dense(
    Z,
    X,
    state,
    alpha: float,
    *,
    ip_cfg: IPConfig,
    shapes: Shapes,
    flat_params,
    unravel_fn,
    full_set_size: int,
):
    """
    Dense objective: KL[q(theta|Z) || p(theta|data)]
    """
    S, *_ = compute_curvature_approx_dense(
        state,
        X,
        alpha=alpha,
        model_type=ip_cfg.model_type,
        full_set_size=full_set_size,
        flat_params=flat_params,
        unravel_fn=unravel_fn,
    )
    S_z, *_ = compute_curvature_approx_dense(
        state,
        Z,
        alpha=alpha,
        model_type=ip_cfg.model_type,
        full_set_size=full_set_size,
        flat_params=flat_params,
        unravel_fn=unravel_fn,
    )

    solve_Sz_S = jnp.linalg.solve(S_z, S)
    trace_term = jnp.trace(solve_Sz_S)

    sS, logdet_S = jnp.linalg.slogdet(S)
    sSz, logdet_Sz = jnp.linalg.slogdet(S_z)
    logdet_term = -sS*logdet_S + sSz*logdet_Sz

    return logdet_term + trace_term


def _prepare_shapes(unravel_fn, flat_params, model_type: str) -> Shapes:
    D = sum(x.size for x in jax.tree_util.tree_leaves(unravel_fn(flat_params)))
    if model_type == "regressor":
        D -= 1
    return Shapes(D=D)


def _make_optimize_step(
    *,
    ip_cfg: IPConfig,
    shapes: Shapes,
    unravel_fn,
    flat_params,
    zupdate: Callable,
    full_set_size: int,
):
    def loss_fn(Z, X, state, alpha, key):
        if ip_cfg.scalable:
            return _objective_mf(
                Z,
                X,
                state,
                alpha,
                ip_cfg=ip_cfg,
                shapes=shapes,
                flat_params=flat_params,
                unravel_fn=unravel_fn,
                key=key,
                full_set_size=full_set_size,
            )
        else:
            return _objective_dense(
                Z,
                X,
                state,
                alpha,
                ip_cfg=ip_cfg,
                shapes=shapes,
                flat_params=flat_params,
                unravel_fn=unravel_fn,
                full_set_size=full_set_size,
            )

    @jax.jit
    def step(Z, X, map_model_state, alpha, opt_state, key):
        loss, grads = jax.value_and_grad(loss_fn)(Z, X, map_model_state, alpha, key)
        updates, new_opt_state = zupdate(grads, opt_state, Z)
        new_Z = optax.apply_updates(Z, updates)
        return new_Z, new_opt_state, loss

    return step


def train_inducing_points(
    *,
    map_state: TrainState,
    Z_init: jnp.ndarray,
    optimizer: optax.GradientTransformation,
    data_loader,
    rng: jax.random.KeyArray,
    alpha: float,
    full_set_size: int,
    ip_cfg: IPConfig,
    num_steps: int,
) -> Tuple[jnp.ndarray, float, TrainState]:
    """
    Trains inducing points Z using either scalable (matrix-free) or dense objective,
    with optional interleaved alpha tuning.
    """
    Z = Z_init
    opt_state = optimizer.init(Z)

    # Prepare parameter flattening & shapes once
    flat_params_map, unravel_fn_map = flatten_nn_params(map_state.params)
    shapes = _prepare_shapes(unravel_fn_map, flat_params_map, model_type=ip_cfg.model_type)

    # Build jitted step with closures
    step = _make_optimize_step(
        ip_cfg=ip_cfg,
        shapes=shapes,
        unravel_fn=unravel_fn_map,
        flat_params=flat_params_map,
        zupdate=optimizer.update,
        full_set_size=full_set_size,
    )

    # Alpha state for interleaved tuning
    alpha_tx = optax.adam(learning_rate=1e-2)
    log_alpha_state = TrainState.create(
        apply_fn=lambda p: p,
        params={"log_alpha": jnp.log(alpha)},
        tx=alpha_tx,
    )

    # Dataloader cycling
    _iter = iter(data_loader)
    def next_X():
        nonlocal _iter
        try:
            batch = next(_iter)
        except StopIteration:
            _iter = iter(data_loader)
            batch = next(_iter)
        return batch[0]

    pbar = tqdm(range(num_steps))
    for t in pbar:
        X_batch = next_X()
        step_key = jax.random.fold_in(rng, t)
        Z, opt_state, loss = step(Z, X_batch, map_state, alpha, opt_state, step_key)

        # Interleaved alpha updates
        if (t % ip_cfg.alpha_steps_every == 0) and (t > ip_cfg.alpha_steps_burnin):
            if ip_cfg.alpha_steps_per_call > 5:
                pbar.set_description_str(f"Training ⍺ for {ip_cfg.alpha_steps_per_call} steps. Last ⍺={alpha:.3e} | Last Loss: {float(loss):.3f}", refresh=True)
            _, alpha_rng = jax.random.split(step_key)
            log_alpha_state, map_state = train_alpha(
                map_state=map_state,
                log_alpha_state=log_alpha_state,
                Z=Z,
                get_batch_fn=lambda: next_X(),
                ip_cfg=ip_cfg,
                rng=alpha_rng,
                full_set_size=full_set_size,
            )
            alpha = jnp.exp(log_alpha_state.params["log_alpha"]).item()

        pbar.set_description_str(f"⍺: {alpha:.3e} | Loss: {float(loss):.3f}", refresh=True)

    return Z, float(alpha), map_state
