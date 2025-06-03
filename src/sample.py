import pdb
import jax
import jax.numpy as jnp
import jax.flatten_util

from matfree import decomp
from matfree.funm import funm_lanczos_sym#, dense_funm_sym_eigh, funm_arnoldi

from src.matfree_monkeypatch import dense_funm_sym_eigh

from src.ggn import compute_W_vps, build_WTW
from src.utils import flatten_nn_params


    
def inv_matsqrt_dense(state, Z, D, alpha, model_type, full_set_size=None):
    """Note: Only for debugging use."""
    # materialize W and WT
    flat_params, _ = flatten_nn_params(state.params)
    D = flat_params.shape[0]
    M = Z.shape[0]
    N = full_set_size or M
    beta = N / M                    # todo remember to use beta!
    
    Wfun, WTfun = compute_W_vps(state, Z, model_type, full_set_size=None)
    I_D = jnp.eye(D)
    W = jax.vmap(WTfun, out_axes=1)(I_D).reshape(D,-1) # (D, d)
    WT = W.T                                           # (d, D)
    
    composite = WT@W
    inv_composite = jnp.linalg.solve(
        composite,
        jnp.eye(composite.shape[0])
    )
    
    nullproj = I_D - W@inv_composite@WT
    term1 = 1/jnp.sqrt(alpha) * nullproj
    
    
    I_d = jnp.eye(W.shape[1])
    evals, evecs = jnp.linalg.eigh(alpha*I_d + beta*composite)
    inv_sqrt_term = (evecs * (1.0 / jnp.sqrt(jnp.clip(evals, 0, jnp.inf)))) @ evecs.T
    
    # invsqrt_fun = dense_funm_sym_eigh(lambda x: 1.0/jnp.sqrt(x))
    # tridiag = decomp.tridiag_sym(min(D, M))  # compare with cg solver iters (todo) # todo maybe set num_matvecs according to sample size - i.e. number of inducing points
    # invmatsqrt_fun = funm_lanczos_sym(invsqrt_fun, tridiag)
    # inv_sqrt_term = jax.vmap(lambda v: invmatsqrt_fun(lambda u: (alpha*I_d + beta*composite)@u, v))(I_d)
    
    term2 = W@inv_composite@inv_sqrt_term@WT

    return term1 + term2
    # return term2


def inv_matsqrt_vp(state, Z, D, alpha, model_type, full_set_size=None, key=None, num_proj_steps=1):
    """
    Computes 1/sqrt(A) for matrix A, a low rank pertubation to the scaled identity, i.e.
    
    A = alpha*I + WW^T, W_m = sqrt(beta) * J_m^T H_m
    
    From thm (1.2) in https://nhigham.com/wp-content/uploads/2023/02/fhl23.pdf
    """
    # ! explicitly set full_set_size to None! Analytically derived beta to only be used inside the matfree invsqrt function
    Wfun, WTfun = compute_W_vps(state, Z, model_type, full_set_size=None)
    Wfun_b, WTfun_b = compute_W_vps(state, Z, model_type, full_set_size=None, blockwise=True) # todo use for alternating projection

    def composite_vp(v):
        return WTfun(Wfun(v))

    def composite_inv_vp(v):
        x,info = jax.scipy.sparse.linalg.cg(A=composite_vp, b=v)
        return x
    
    dummy = WTfun(jnp.zeros(D))
    inner_shape = dummy.shape
    d           = dummy.size
    WTW = build_WTW(Wfun, WTfun, inner_shape, d, dtype=float, block=2)
    def nullproj_vp(v): # verify: mult with GGN should output 0
        u = WTfun(v)
        uflat,unravel_fn = jax.flatten_util.ravel_pytree(u)
        x = jax.scipy.linalg.solve(
            WTW,
            uflat
        )
        return v - Wfun(unravel_fn(x))
    
    def nullproj_approx(v, key, steps):
        def outer_body(step, state):
            v, key = state
            # generate a new permutation for each outer iteration
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, N)
            
            # todo make batched, dense.
            def inner_body(i, v):
                b = perm[i]
                # return nullproj_fun_b(b, v)
            
            v = jax.lax.fori_loop(0, N, inner_body, v)
            return (v, key)
        v, _ = jax.lax.fori_loop(0, steps, outer_body, (v, key))
        return v
    
    if key is not None:
        nullproj_term = lambda v: 1/jnp.sqrt(alpha) * nullproj_approx(v, key=key, steps=num_proj_steps) # Alternating projection stuff!
    else:
        nullproj_term = lambda v: 1/jnp.sqrt(alpha) * nullproj_vp(v) # direct projection!
    
    M = Z.shape[0]
    N = full_set_size or M
    beta = N / M
    
    invsqrt_fun = dense_funm_sym_eigh(lambda x: 1.0/jnp.sqrt(x)) # ! using monkeypatched clipped version of dense_funm_sym_eigh!!
    decomp_method = decomp.tridiag_sym(2*M)
    invmatsqrt = funm_lanczos_sym(invsqrt_fun, decomp_method)

    def invmatsqrt_term(V):
        """Note: matfree expects 1D input, so we wrap the operation in flatten/unflatten."""
        Vflat, unravel_fn = jax.flatten_util.ravel_pytree(V)
        def inner_fun_flat(Uflat):
            # Umat = unravel_fn(Uflat)
            # WTUmat = composite_vp(Umat)
            WTUmat = WTW@Uflat
            WTUmat_flat,_ = jax.flatten_util.ravel_pytree(WTUmat)
            return alpha*Uflat + beta*WTUmat_flat
        result_flat = invmatsqrt(inner_fun_flat, Vflat)
        result = unravel_fn(result_flat)
        return result
    
    def outer_fun(v):
        u = invmatsqrt_term(
            WTfun(v)
        )
        uflat,unravel_fn = jax.flatten_util.ravel_pytree(u)
        x = jax.scipy.linalg.solve(
            WTW,
            uflat
        )
        return Wfun(unravel_fn(x))
    
    @jax.jit
    def vp(v):
        return outer_fun(v) + nullproj_term(v)
    
    return vp


def sample(state, Z, D, alpha, key, model_type, num_samples=1, full_set_size=None, num_proj_steps=10):
    sample_key, altproj_key = jax.random.split(key, 2)
    altproj_key = None # TODO handle! # currently the alternating projections just give NaN values
    Eps = jax.random.normal(sample_key, shape=(num_samples, D))
    inv_matsqrt_fun = inv_matsqrt_vp(state, Z, D, alpha, model_type, full_set_size=full_set_size, key=altproj_key, num_proj_steps=num_proj_steps)
    # flat_params, unravel_fn = flatten_nn_params(state.params['params']) # todo could potentially make it s.t. we reuse MAP by passing flat params to inv_sqrtm
    # samples = jax.vmap(inv_matsqrt_fun, in_axes=(0,))(Eps) #+ flat_params
    samples = jax.lax.map(inv_matsqrt_fun, Eps)
    return samples


def sample_dense(state, Z, D, alpha, key, model_type, num_samples=1, full_set_size=None):
    inv_matsqrt_ggn = inv_matsqrt_dense(state, Z, D, alpha, model_type, full_set_size=full_set_size)
    inv_matsqrt_fun = lambda v: inv_matsqrt_ggn @ v
    Eps = jax.random.normal(key, shape=(num_samples, D))
    flat_params, unravel_fn = flatten_nn_params(state.params['params']) # todo could potentially make it s.t. we reuse MAP by passing flat params to inv_sqrtm
    samples = jax.vmap(inv_matsqrt_fun, in_axes=(0,))(Eps) + flat_params
    return samples


def sample_both(state, Z, D, alpha, key, model_type, num_samples=1, full_set_size=None):
    Eps = jax.random.normal(key, shape=(num_samples, D))
    # flat_params, unravel_fn = flatten_nn_params(state.params['params']) # todo could potentially make it s.t. we reuse MAP by passing flat params to inv_sqrtm
    
    inv_matsqrt_fun = inv_matsqrt_vp(state, Z, D, alpha, model_type, full_set_size=full_set_size)
    samples = jax.vmap(inv_matsqrt_fun, in_axes=(0,))(Eps)
    
    inv_matsqrt_ggn = inv_matsqrt_dense(state, Z, D, alpha, model_type, full_set_size=full_set_size)
    inv_matsqrt_fun_dense = lambda v: inv_matsqrt_ggn @ v
    dense_samples = jax.vmap(inv_matsqrt_fun_dense, in_axes=(0,))(Eps)
    return samples, dense_samples