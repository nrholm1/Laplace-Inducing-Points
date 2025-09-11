import pdb
import jax
import jax.numpy as jnp
from functools import partial

from src.utils import flatten_nn_params


def compute_W_vps(state, Z, model_type, full_set_size=None, blockwise=False):
    flat_params, unravel_fn = flatten_nn_params(state.params)
    M = Z.shape[0]
    N = full_set_size or M
    scale = jnp.sqrt(N / M)
    f32 = jnp.float32

    def _apply(pflat, zi):
        p_unr = unravel_fn(pflat)
        if model_type == "regressor":
            return state.apply_fn(p_unr, zi, return_logvar=False)
        else:
            variables = {"params": p_unr, "batch_stats": state.batch_stats}
            return state.apply_fn(variables, zi, train=False, mutable=False)

    def _sqrt_H_T(f_out, u):
        if model_type == "regressor":
            c = jnp.exp(-state.params["logvar"]["logvar"]).astype(f32)
            return jnp.sqrt(c) * u
        else:
            p = jax.nn.softmax(f_out)
            s = jnp.sqrt(p)
            return s * u - (jnp.dot(s, u)) * p

    def _sqrt_H(f_out, u):
        if model_type == "regressor":
            c = jnp.exp(-state.params["logvar"]["logvar"]).astype(f32)
            return jnp.sqrt(c) * u
        else:
            p = jax.nn.softmax(f_out)
            s = jnp.sqrt(p)
            return s * u - (jnp.dot(p, u)) * s

    def _WT_i(i, v):
        zi = jax.lax.dynamic_index_in_dim(Z, i, keepdims=False)
        def f(p): return _apply(p, zi).squeeze()
        f0, jv = jax.jvp(f, (flat_params,), (v,))
        return _sqrt_H(f0, jv)

    def _W_i(i, u_i):
        zi = jax.lax.dynamic_index_in_dim(Z, i, keepdims=False)
        def f(p): return _apply(p, zi).squeeze()
        f0, lin = jax.linearize(f, flat_params)
        h = _sqrt_H_T(f0, u_i)
        jt = jax.linear_transpose(lin, flat_params)
        w_i, = jt(h)
        return w_i

    if blockwise:
        return (
            lambda i, U_i: scale * _W_i(i, U_i),
            lambda i, v:   scale * _WT_i(i, v),
        )

    def WTfun(v):
        idx = jnp.arange(M, dtype=jnp.int32)
        per_i = jax.vmap(lambda i: _WT_i(i, v))(idx)   # (M, K)
        return scale * per_i

    def Wfun(U):
        idx = jnp.arange(M, dtype=jnp.int32)
        per_i = jax.vmap(_W_i, in_axes=(0, 0))(idx, U) # (M, D)
        return scale * per_i.sum(axis=0)

    return Wfun, WTfun



def compute_ggn_vp(state, Z, model_type, full_set_size=None):
    flat_params, unravel_fn = flatten_nn_params(state.params)
    M = Z.shape[0]
    N = full_set_size or M
    scale = (N / M)
    if model_type == "regressor":
        scale = scale * jnp.exp(-state.params["logvar"]["logvar"])

    def _apply(pflat, zi):
        p_unr = unravel_fn(pflat)
        if model_type == "regressor":
            return state.apply_fn(p_unr, zi, return_logvar=False)
        else:
            variables = {"params": p_unr, "batch_stats": state.batch_stats}
            return state.apply_fn(variables, zi, train=False, mutable=False)

    def _H_action(f_out, u):
        if model_type == "classifier":
            p = jax.nn.softmax(f_out)
            H = jnp.diag(p) - jnp.outer(p, p)
            return H @ u
        else:
            return u

    def ggn_vp(v):
        def body(i, acc):
            zi = jax.lax.dynamic_index_in_dim(Z, i, keepdims=False)
            def f(p): return _apply(p, zi).squeeze()
            f0, lin = jax.linearize(f, flat_params)
            jv = lin(v)
            hv = _H_action(f0, jv)
            jt = jax.linear_transpose(lin, flat_params)
            jt_h, = jt(hv)
            return acc + jt_h
        total = jax.lax.fori_loop(0, M, body, jnp.zeros_like(flat_params))
        return scale * total

    return ggn_vp


def compute_ggn_dense(state, Z, model_type, full_set_size=None):
    """
    Computes the GGN, instantiating everything along the way.
    @params
        Z: data points, i.e. potentially inducing points.
        w: global recalibration parameter (learned).
        model_type: "regressor"|"classifier"
        full_set_size: (if using inducing points or minibatching) size of full data set.
    """
    flat_params, unravel_fn = flatten_nn_params(state.params)

    def model_fun(flatp, xi):
        p_unr = unravel_fn(flatp)
        if model_type == "regressor": return state.apply_fn(p_unr, xi, return_logvar=False)
        else: return state.apply_fn({'params': p_unr}, xi, train=False)

    M = Z.shape[0]
    # Initialize GGN as a zero matrix
    GGN = jnp.zeros((flat_params.shape[0], flat_params.shape[0]))

    def body_fun(i, acc):
        zi = jax.lax.dynamic_index_in_dim(Z, i, keepdims=False)
        J = jax.jacobian(lambda p: model_fun(p, zi))(flat_params)
        if model_type == "classifier":
            # ! per datum hessian for classification
            fxi = model_fun(flat_params, zi)
            probs = jax.nn.softmax(fxi)
            H_loss = jnp.diag(probs) - jnp.outer(probs, probs)
            ggn_i = J.T @ H_loss @ J
        else:
            ggn_i = J.T @ J
        return acc + ggn_i

    GGN = jax.lax.fori_loop(0, M, body_fun, GGN)
    
    if model_type == "regressor":
        # ! hessian for regression - equivalent simply to a scalar coefficient
        varinv = jnp.exp( - state.params['logvar']['logvar']) 
        GGN *= varinv
    
    # recalibration term
    N = full_set_size or M
    GGN *= N / M

    return GGN, flat_params, unravel_fn
    
    


def build_WTW(W, WT, inner_shape, d, *, dtype=jnp.bfloat16, block=64):
    """
    Return WᵀW ∈ R^{dxd} with ≤ (block · #params) peak memory.
    """
    @partial(jax.remat, static_argnums=1)          # k is static
    def col_block(start, k):
        rows = start + jnp.arange(k, dtype=jnp.int32)        # shape (k,)
        E    = jax.nn.one_hot(rows, d, dtype=dtype)\
                  .reshape((k,) + inner_shape)               # (k, M, C)
        cols = jax.vmap(lambda e: WT(W(e)).reshape(-1))(E)   # (k, d)
        return cols.astype(dtype)                            # (k, d)

    WTW = jnp.zeros((d, d), dtype=dtype)

    n_full, tail = divmod(d, block)

    def body(b, acc):
        start = b * block
        cols  = col_block(start, block)      # (block, d)
        return jax.lax.dynamic_update_slice(acc, cols.T, (0, start))

    WTW = jax.lax.fori_loop(0, n_full, body, WTW)

    # Tail slice, checkpointed
    if tail:
        start  = n_full * block
        cols_t = col_block(start, tail).T    # (d, tail)
        WTW    = jax.lax.dynamic_update_slice(WTW, cols_t, (0, start))

    return jnp.triu(WTW) + jnp.triu(WTW, 1).T
        




def ensure_symmetry(M, jitter=1e-8):
    return 0.5 * (M + M.T) + jitter * jnp.eye(M.shape[0]) # ! enforce symmetry of a theoretically symmetric matrix for numerical stability
