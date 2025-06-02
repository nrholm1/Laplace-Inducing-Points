from functools import partial
import pdb
import jax
import jax.numpy as jnp


def stochastic_trace_estimator_dense(X, seed, num_samples=1_000):
    """
    Uses Girard-Hutchinson estimator on fully instantiated matrices.
    """
    def sample_eps(X, seed, num_samples): 
        return jax.random.rademacher(key=seed, shape=(num_samples, X.shape[0]), dtype=X.dtype)
        # return jax.random.normal(key=seed, shape=(num_samples, X.shape[0]))
    def single_estimate(X, eps):
        y = jnp.matmul(X, eps)
        return jnp.dot(eps, y)
    Eps = sample_eps(X, seed=seed, num_samples=num_samples)
    
    return jax.vmap(single_estimate, in_axes=(None, 0))(X,Eps).mean()


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


def hutchpp_dense(X, seed, num_samples=10):
    """
    Uses Hutch++ with fully instantiated matrices.
    """
    # ? Sample isotropic random vectors, either from N(0,I) or with Rademacher dist. (unif{-1,+1} indices)
    # eps = jax.random.rademacher(key=seed, shape=(num_samples * 2, X.shape[0]))
    eps = jax.random.normal(key=seed, shape=(num_samples * 2, X.shape[0])) 
    S,G = jnp.split(eps, 2, axis=0)
    
    Q,R = jnp.linalg.qr(X @ S.T)
    orthproj = (jnp.eye(Q.shape[0]) - Q@Q.T) # symmetric
    
    return jnp.trace(Q.T@X@Q) + (1/num_samples) * jnp.trace(G@orthproj@X@orthproj@G.T)


def hutchpp_mvp(Xfun, D, seed, num_samples=10):
    """
    Uses Hutch++ with linear operator oracle function.
    - `Xfun`: oracle computing v -> X@v, where X: square matrix
    - `D`: int, dim(X)
    """
    
    # ? Sample isotropic random vectors, either from N(0,I) or Rademacher dist. (i.e. unif{-1,+1} indices)
    # eps = jax.random.rademacher(key=seed, shape=(num_samples * 2, D))
    eps = jax.random.normal(key=seed, shape=(num_samples * 2, D))
    S,G = jnp.split(eps, 2, axis=0)
    
    Q,R = jnp.linalg.qr(Xfun(S.T))
    orthproj = (jnp.eye(Q.shape[0]) - Q@Q.T) # symmetric
    
    @jax.jit
    def quad_term(M):
        """
        Compute M^T X M
        as M.T@X@M = M.T@(X@M)
        """
        # Y = jax.vmap(Xfun, in_axes=1, out_axes=1)(M)
        Y = Xfun(M)
        return M.T @ Y
    
    estimates = jnp.trace(quad_term(Q)) + (1/num_samples) * jnp.trace(quad_term(orthproj@G.T))
    
    return estimates


def hutchpp(Xfun, sampler):
    """
    Uses Hutch++ with linear operator oracle function.
    - `Xfun`: oracle computing v -> X@v, where X: square matrix
    """
    
    eps = sampler(...)
    num_samples = eps.shape[0]
    S,G = jnp.split(eps, 2, axis=0)
    
    Q,_ = jnp.linalg.qr(
        jax.vmap(Xfun,in_axes=0,out_axes=1)(S),
        mode='reduced'
    )
    orthproj = (jnp.eye(Q.shape[0]) - Q@Q.T) # symmetric
    
    # @jax.jit
    def quad_term(M):
        """
        Compute M^T X M
        as M.T@X@M = M.T@(X@M)
        """
        Y = jax.vmap(Xfun, in_axes=1, out_axes=1)(M)
        # Y = Xfun(M)
        return M.T @ Y
    
    # pdb.set_trace()
    estimates = jnp.trace(quad_term(Q)) + (1/num_samples) * jnp.trace(quad_term(orthproj@G.T))
    
    return estimates

def apply_X(Xfun, M):                  # M  (k, n)  rows = probes
    return jax.vmap(Xfun, in_axes=0, out_axes=1)(M)


# @partial(jax.jit, static_argnames=("Xfun", "s1", "s2"))
def hutchpp_v2(Xfun, sampler):
    eps = sampler(...)          # (2k, n)   ← rows = probes
    k   = eps.shape[0] // 2
    S, G = jnp.split(eps, (k,), axis=0)   # (k, n), (k, n)

    # -- low-rank QR part --------------------------------------------------
    Y   = apply_X(Xfun, S)                  # (n, k)
    Q, _ = jnp.linalg.qr(Y, mode='reduced') # (n, k), orthonormal columns

    XQ     = jax.remat(apply_X, static_argnums=0)(Xfun, Q.T)  # (n, k)
    low_rank = jnp.trace(XQ.T @ Q)            # tr(Qᵀ X Q)

    # -- residual Hutchinson part  ----------------------------------------
    G_perp = G - (G @ Q) @ Q.T             # projector
    XGp    = jax.remat(apply_X, static_argnums=0)(Xfun, G_perp)
    resid  = jnp.trace(G_perp @ XGp) / k

    return low_rank + resid


def hutchpp_inv_mvp(Xfun, D, seed, num_samples=10):
    """
    Uses Conjugate Gradient on Hutch++ with linear operator oracle function to approximate the trace of the inverse.
    - `Xfun`: oracle computing v -> X@v, where X: square matrix
    - `D`: int, dim(X)
    """
    @jax.jit
    def Xinvfun(v):
        x,info = jax.scipy.sparse.linalg.cg(A=Xfun, b=v)
        return x
    return hutchpp_mvp(Xinvfun, D, seed, num_samples=num_samples)


def na_hutchpp_dense(X, seed, num_samples=10):
    """
    Uses NA-Hutch++ with fully instantiated matrices.
    """
    c1,c2,c3 = .25,.5,.25 # good values, given in Hutch++ paper.
    # ? Sample isotropic random vectors, either from N(0,I) or with Rademacher dist. (unif{-1,+1} indices)
    eps = jax.random.rademacher(key=seed, shape=(num_samples * 4, X.shape[0]))
    # eps = jax.random.normal(key=seed, shape=(num_samples * 4, X.shape[0])) 
    S,R,G = jnp.split(eps, [num_samples, num_samples*3], axis=0) # split into [1/4, 2/4, 1/4]
    W = X @ S.T
    Z = X @ R.T
    
    return jnp.trace(jnp.linalg.pinv(S@Z) @ (W.T@Z)) + (1/(c3*4*num_samples)) * (jnp.trace(G@X@G.T) - jnp.trace(G@Z@jnp.linalg.pinv(S@Z)@W.T@G.T))


def na_hutchpp_mvp(Xfun, D, seed, num_samples=10, dtype=jnp.float32):
    """
    Uses NA-Hutch++ with linear operator oracle function.
    - `Xfun`: oracle computing v -> X@v, where X: square matrix
    - `D`: int, dim(X)
    """
    c1,c2,c3 = .25,.5,.25 # good values, given in Hutch++ paper.
    # ? Sample isotropic random vectors, either from N(0,I) or with Rademacher dist. (unif{-1,+1} indices)
    eps = jax.random.rademacher(key=seed, shape=(num_samples * 4, D), dtype=dtype)
    # eps = jax.random.normal(key=seed, shape=(num_samples * 4, D)) 
    S,R,G = jnp.split(eps, [num_samples, num_samples*3], axis=0) # split into [1/4, 2/4, 1/4]
    W = Xfun(S.T)
    Z = Xfun(R.T)
    
    return jnp.trace(jnp.linalg.pinv(S@Z) @ (W.T@Z)) + (1/(c3*4*num_samples)) * (jnp.trace(G@Xfun(G.T)) - jnp.trace(G@Z@jnp.linalg.pinv(S@Z)@W.T@G.T))


def na_hutchpp_inv_mvp(Xfun, D, seed, num_samples=10):
    """
    Uses Conjugate Gradient on NA-Hutch++ with linear operator oracle function to approximate the trace of the inverse.
    - `Xfun`: oracle computing v -> X@v, where X: square matrix
    - `D`: int, dim(X)
    """
    @jax.jit
    def Xinvfun(v):
        v = v.astype(jnp.float64)
        x,info = jax.scipy.sparse.linalg.cg(A=Xfun, b=v)
        return x
    return na_hutchpp_mvp(Xinvfun, D, seed, num_samples=num_samples)


# todo could also implement XTrace? Seems to not be a better choice for our case, so defer...