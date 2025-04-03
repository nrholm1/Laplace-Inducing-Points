import jax
import jax.numpy as jnp
from matfree import decomp
from matfree.funm import funm_lanczos_sym, dense_funm_sym_eigh, funm_arnoldi

from src.ggn import compute_W_vps
from src.utils import flatten_nn_params


def altkerproj(Wfun, WTfun, nsteps=10):
    """
    Minibatched alternating projection algorithm to approximate the projection into ker(W^T.W).
    
    Returns a vp.
    
    The approximation is I - proj(W^T.W) ≈ (prod_b [I_d - Proj(W_b^T@W_b)])^t, for t → inf,
    where we provide a Wfun that computes concatenated W_b
    """
    


def inv_matsqrt_vp(state, Z, D, alpha, model_type, full_set_size=None):
    """
    Computes 1/sqrt(A) for matrix A, a low rank pertubation to the scaled identity, i.e.
    
    A = alpha*I + WW^T, W_m = sqrt(beta) * J_m^T H_m
    
    From thm (1.2) in https://nhigham.com/wp-content/uploads/2023/02/fhl23.pdf
    """
    Wfun, WTfun = compute_W_vps(state, Z, model_type, full_set_size=full_set_size)
    Wfun_b, WTfun_b = compute_W_vps(state, Z, model_type, full_set_size=full_set_size, blockwise=True)

    def composite_vp(v):
        return WTfun(Wfun(v))

    def composite_inv_vp(v):
        x,info = jax.scipy.sparse.linalg.cg(A=composite_vp, b=v)
        return x
    
    def nullproj_vp(v): # verify: mult with GGN should output 0
        return v - Wfun(composite_inv_vp(WTfun(v)))
    
    nullproj_term = lambda v: 1/jnp.sqrt(alpha) * nullproj_vp(v) # todo Alternating projection stuff!
    
    M = Z.shape[0]
    N = full_set_size or M
    beta = N / M                    # todo remember to use beta!
    dummy_primal = jnp.ones(D,)
    Wshape = WTfun(dummy_primal).shape
    
    invsqrt_fun = dense_funm_sym_eigh(lambda x: 1.0/jnp.sqrt(x))
    tridiag = decomp.tridiag_sym(min(16,M))  # todo maybe set num_matvecs according to sample size - i.e. number of inducing points
    invmatsqrt = funm_lanczos_sym(invsqrt_fun, tridiag)

    def invmatsqrt_term(V):
        """
        v has shape (p,).  We do:
        1) V = WTfun(v) -> shape (M,K)
        2) flatten to shape (M*K,)
        3) call 'invmatsqrt' in that flattened space
        4) reshape back to (M,K).
        """
        Vflat = V.reshape(-1)           # shape (M*K,)

        def inner_fun_flat(Uflat):
            # 1) unflatten U to shape (M,K)
            Umat = Uflat.reshape(*Wshape)
            # 2) apply operator in matrix form
            #    This is alpha*U + beta*(WTfun(Wfun(U))),
            WTUmat = WTfun(Wfun(Umat))  # shape (M,K)
            WTUmat_flat = WTUmat.reshape(-1)  # shape (M*K,)
            # 3) also flatten the "alpha * Umat" part:
            return alpha * Uflat + beta * WTUmat_flat

        # Now do the Lanczos-based call in the 1D space of length M*K
        result_flat = invmatsqrt(inner_fun_flat, Vflat)  # result is shape (M*K,)

        # Optionally reshape the result back to (M, K) if needed:
        result = result_flat.reshape(*Wshape)
        return result
    # u = WTfun(v)    => (200, 2)
    # w = innerfun(u) => (200, 2)
    
    def outer_fun(v):
        return Wfun(
            composite_inv_vp(
                invmatsqrt_term(WTfun(v))
            )
        )
    
    # todo @jax.jit
    def vp(v):
        return outer_fun(v) + nullproj_term(v)
    
    return vp



def sample(state, Z, D, alpha, key, model_type, num_samples=1):
    inv_matsqrt_fun = inv_matsqrt_vp(state, Z, D, alpha, model_type)
    Eps = jax.random.normal(key, shape=(num_samples, D))
    flat_params, unravel_fn = flatten_nn_params(state.params['params']) # todo could potentially make it s.t. we reuse MAP by passing flat params to inv_sqrtm
    return jax.vmap(inv_matsqrt_fun, in_axes=(0,))(Eps) + flat_params