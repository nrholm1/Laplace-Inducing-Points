import jax
import jax.numpy as jnp


def stochastic_trace_estimator_full(X, seed, num_samples=1_000):
    """
    Uses Girard-Hutchinson estimator on fully instantiated matrices.
    # ! quite unstable! Bug or just slow MSE convergence?
    """
    def sample_eps(X, seed, num_samples): 
        return jax.random.rademacher(key=seed, shape=(num_samples, X.shape[0]))
        # return jax.random.normal(key=seed, shape=(num_samples, X.shape[0]))
    def single_estimate(X, eps):
        y = jnp.matmul(X, eps)
        return jnp.dot(eps, y)
    eps = sample_eps(X, seed=seed, num_samples=num_samples)
    
    return jax.vmap(single_estimate, in_axes=(None,0))(X,eps)


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
    
    def quad_term(M):
        """
        Compute M^T X M
        as M.T@X@M = M.T@(X@M)
        """
        Y = jax.vmap(Xfun, in_axes=1, out_axes=1)(M)
        return M.T @ Y
    
    estimates = jnp.trace(quad_term(Q)) + (1/num_samples) * jnp.trace(quad_term(orthproj@G.T))
    
    return estimates.mean()


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


def na_hutchpp_mvp(Xfun, D, seed, num_samples=10):
    """
    Uses NA-Hutch++ with linear operator oracle function.
    - `Xfun`: oracle computing v -> X@v, where X: square matrix
    - `D`: int, dim(X)
    """
    c1,c2,c3 = .25,.5,.25 # good values, given in Hutch++ paper.
    # ? Sample isotropic random vectors, either from N(0,I) or with Rademacher dist. (unif{-1,+1} indices)
    eps = jax.random.rademacher(key=seed, shape=(num_samples * 4, D))
    # eps = jax.random.normal(key=seed, shape=(num_samples * 4, D)) 
    S,R,G = jnp.split(eps, [num_samples, num_samples*3], axis=0) # split into [1/4, 2/4, 1/4]
    W = Xfun(S.T)
    Z = Xfun(R.T)
    
    return jnp.trace(jnp.linalg.pinv(S@Z) @ (W.T@Z)) + (1/(c3*4*num_samples)) * (jnp.trace(G@Xfun(G.T)) - jnp.trace(G@Z@jnp.linalg.pinv(S@Z)@W.T@G.T))


# todo could also implement XTrace? Seems to not be a better choice for our case, so defer...