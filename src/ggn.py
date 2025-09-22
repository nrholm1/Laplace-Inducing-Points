import pdb
import jax
import jax.numpy as jnp
from functools import partial

from src.utils import flatten_nn_params


def compute_W_vps(state, Z, model_type, *, flat_params, unravel_fn, full_set_size=None, blockwise=False):
    M = Z.shape[0]
    N = full_set_size or M
    scale = jnp.sqrt(N / M)

    def _apply(pflat, zi):
        p_unr = unravel_fn(pflat)
        if model_type == "regressor":
            return state.apply_fn(p_unr, zi, return_logvar=False)
        else:
            variables = {"params": p_unr, "batch_stats": state.batch_stats}
            return state.apply_fn(variables, zi, train=False, mutable=False)

    def _sqrt_H_T(f_out, u):
        if model_type == "regressor":
            c = jnp.exp(-state.params["logvar"]["logvar"])
            return jnp.sqrt(c) * u
        else:
            p = jax.nn.softmax(f_out); s = jnp.sqrt(p)
            return s * u - (jnp.dot(s, u)) * p

    def _sqrt_H(f_out, u):
        if model_type == "regressor":
            c = jnp.exp(-state.params["logvar"]["logvar"])
            return jnp.sqrt(c) * u
        else:
            p = jax.nn.softmax(f_out); s = jnp.sqrt(p)
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
        per_i = jax.lax.map(lambda i: _WT_i(i, v), idx)
        return scale * per_i

    def Wfun(U):
        idx = jnp.arange(M, dtype=jnp.int32)
        def body_map(pair):
            i, u_i = pair
            return _W_i(i, u_i)
        per_i = jax.lax.map(body_map, (idx, U))
        return scale * per_i.sum(axis=0)

    return Wfun, WTfun



def compute_ggn_vp(state, Z, model_type, *, flat_params, unravel_fn, full_set_size=None):
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


def compute_ggn_dense(state, Z, model_type, *, flat_params, unravel_fn, full_set_size=None):
    def model_fun(pflat, xi):
        p_unr = unravel_fn(pflat)
        if model_type == "regressor":
            return state.apply_fn(p_unr, xi, return_logvar=False)
        else:
            return state.apply_fn({'params': p_unr}, xi, train=False)

    M = Z.shape[0]
    D = flat_params.shape[0]
    GGN = jnp.zeros((D, D), dtype=flat_params.dtype)

    def body_fun(i, acc):
        zi = jax.lax.dynamic_index_in_dim(Z, i, keepdims=False)
        J = jax.jacobian(lambda p: model_fun(p, zi))(flat_params)
        if model_type == "classifier":
            fxi = model_fun(flat_params, zi)
            probs = jax.nn.softmax(fxi)
            H_loss = jnp.diag(probs) - jnp.outer(probs, probs)
            ggn_i = J.T @ H_loss @ J
        else:
            ggn_i = J.T @ J
        return acc + ggn_i

    GGN = jax.lax.fori_loop(0, M, body_fun, GGN)

    if model_type == "regressor":
        varinv = jnp.exp(-state.params['logvar']['logvar'])
        GGN *= varinv

    N = full_set_size or M
    GGN *= N / M
    return GGN, flat_params, unravel_fn
    
    


def build_WTW(W, WT, inner_shape, d, *, dtype=jnp.bfloat16, block=256):
    compute_dtype = dtype
    acc_dtype = jnp.float32 if compute_dtype in (jnp.bfloat16, jnp.float16) else compute_dtype

    W_b  = jax.vmap(W,  in_axes=0, out_axes=0)
    WT_b = jax.vmap(WT, in_axes=0, out_axes=0)

    @partial(jax.remat, static_argnums=(1,))
    def col_block(start, k):
        rows = start + jnp.arange(k, dtype=jnp.int32)
        E = jax.nn.one_hot(rows, d, dtype=compute_dtype)\
                .reshape((k,) + inner_shape)
        WE   = W_b(E)                   
        WTWE = WT_b(WE).reshape(k, d)   
        return WTWE.T.astype(acc_dtype)        

    WTW = jnp.zeros((d, d), dtype=acc_dtype)
    n_full, tail = divmod(d, block)

    def body(carry, b):
        start = b * block
        colsT = col_block(start, block)
        carry = jax.lax.dynamic_update_slice(carry, colsT, (0, start))
        return carry, None

    # main blocks
    WTW, _ = jax.lax.scan(body, WTW, jnp.arange(n_full, dtype=jnp.int32))

    # tail block
    if tail:
        start  = n_full * block
        cols_t = col_block(start, tail)
        WTW    = jax.lax.dynamic_update_slice(WTW, cols_t, (0, start))

    return 0.5 * (WTW + WTW.T)

        




def ensure_symmetry(M, jitter=1e-8):
    return 0.5 * (M + M.T) + jitter * jnp.eye(M.shape[0]) # ! enforce symmetry of a theoretically symmetric matrix for numerical stability
