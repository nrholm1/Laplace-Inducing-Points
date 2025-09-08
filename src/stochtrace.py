from functools import partial
import pdb
import jax
import jax.numpy as jnp


def stochastic_trace_estimator_mvp(Xfun, D, seed, num_samples=1_000, dtype=jnp.float32):
    """
    Uses Girard-Hutchinson estimator with linear operator oracles.
    """
    
    def sample_eps(seed, num_samples): 
        return jax.random.rademacher(key=seed, shape=(num_samples, D), dtype=dtype)
        # return jax.random.normal(key=seed, shape=(num_samples, X.shape[0]))
    def single_estimate(Xfun, eps):
        return jnp.dot(eps, Xfun(eps))
    Eps = sample_eps(seed=seed, num_samples=num_samples)
    
    return jax.vmap(single_estimate, in_axes=(None, 0))(Xfun,Eps).mean()



def apply_X(Xfun, M):                  # M  (k, n)  rows = probes
    return jax.vmap(Xfun, in_axes=0, out_axes=1)(M)


# @partial(jax.jit, static_argnames=("Xfun", "s1", "s2"))
def hutchpp(Xfun, sampler, *, s1, s2):
    eps = sampler(...)          # (2k, n)   ← rows = probes
    # k   = eps.shape[0] // 2
    S, G = jnp.split(eps, (s1,), axis=0)   # (k, n), (k, n)

    # -- low-rank QR part --------------------------------------------------
    Y   = apply_X(Xfun, S)                  # (n, k)
    Q, _ = jnp.linalg.qr(Y, mode='reduced') # (n, k), orthonormal columns

    XQ     = jax.remat(apply_X, static_argnums=0)(Xfun, Q.T)  # (n, k)
    low_rank = jnp.trace(XQ.T @ Q)            # tr(Qᵀ X Q)

    # -- residual Hutchinson part  ----------------------------------------
    G_perp = G - (G @ Q) @ Q.T             # projector
    XGp    = jax.remat(apply_X, static_argnums=0)(Xfun, G_perp)
    resid  = jnp.trace(G_perp @ XGp) / s2

    return low_rank + resid
