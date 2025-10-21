import pdb
import jax
import jax.numpy as jnp
import jax.flatten_util

from matfree import decomp
from matfree.funm import funm_lanczos_sym#, dense_funm_sym_eigh, funm_arnoldi

from src.matfree_monkeypatch import dense_funm_sym_eigh

from src.ggn import compute_W_vps, build_WTW
from src.utils import flatten_nn_params


def inv_matsqrt_vp(
    state,
    Z,
    D,
    alpha,
    model_type,
    full_set_size=None,
    key=None,
    num_proj_steps=1,
    *,
    flat_params,
    unravel_fn,
):
    """
    Returns a function v ↦ (alpha*I + W W^T)^{-1/2} projected to parameter space,
    using Woodbury structure with a dense W^T W and a single Cholesky per call.
    """
    # use unscaled W, WT (full_set_size=None by design here)
    W, WT = compute_W_vps(
        state, Z, model_type,
        full_set_size=None,
        flat_params=flat_params, unravel_fn=unravel_fn,
    )

    dummy       = WT(jnp.zeros(D, dtype=jnp.float32))
    inner_shape = dummy.shape
    d           = dummy.size

    # dense W^T W once, stable dtype and sizable blocks
    WTW = build_WTW(W, WT, inner_shape, d, dtype=jnp.float32, block=min(64, int(Z.shape[0])))

    def solve_WTW(u):
        # y = jax.scipy.linalg.solve_triangular(L_wtw,   u, lower=True)
        # x = jax.scipy.linalg.solve_triangular(L_wtw.T, y, lower=False)
        x = jnp.linalg.solve(WTW, u)
        return x

    # null-projection term
    def nullproj_vp(v):
        u = WT(v).reshape(d)
        x = solve_WTW(u)
        return v - W(x.reshape(inner_shape))

    def nullproj_vp_approx(v, k, steps):
        def outer_body(_, state):
            v, k = state
            k, sub = jax.random.split(k)
            # TODO: finish if you re-enable alternating projections
            return (v, k)
        v, _ = jax.lax.fori_loop(0, steps, outer_body, (v, k))
        return v

    nullproj_term = lambda v: (1.0 / jnp.sqrt(alpha)) * nullproj_vp(v)

    # beta from external full_set_size (not used inside W,WT)
    M = Z.shape[0]
    N = full_set_size or M
    beta = N / M

    # A(u) = alpha * u + beta * (WTW @ u), operator on R^d
    invsqrt_fun   = dense_funm_sym_eigh(lambda x: 1.0 / jnp.sqrt(x))
    decomp_method = decomp.tridiag_sym(M)
    invmatsqrt    = funm_lanczos_sym(invsqrt_fun, decomp_method)

    def A_mv(u):
        return alpha * u + beta * (WTW @ u)

    def invmatsqrt_apply(u_flat):
        return invmatsqrt(A_mv, u_flat)

    def outer_fun(v):
        u = WT(v).reshape(d)                 # R^d
        y = invmatsqrt_apply(u)              # (alpha I + beta WTW)^{-1/2} u
        x = solve_WTW(y)                     # (WTW)^{-1} y
        return W(x.reshape(inner_shape))     # lift back

    @jax.jit
    def vp(v):
        return outer_fun(v) + nullproj_term(v)

    return vp


def sample(
    state,
    Z,
    D,
    alpha,
    key,
    model_type,
    num_samples=1,
    full_set_size=None,
    num_proj_steps=10,
    *,
    flat_params,
    unravel_fn,
):
    sample_key, _ = jax.random.split(key, 2)
    Eps = jax.random.normal(sample_key, shape=(num_samples, D), dtype=jnp.float32)

    inv_ms = inv_matsqrt_vp(
        state, Z, D, alpha, model_type,
        full_set_size=full_set_size, 
        key=None, # todo None on purpose. If set, alternating projections will be used (currently does not work).
        num_proj_steps=num_proj_steps,
        flat_params=flat_params, unravel_fn=unravel_fn,
    )

    return jax.lax.map(inv_ms, Eps)

