import pdb
import jax
import jax.numpy as jnp
from jax import lax
import tensorflow_probability.substrates.jax.distributions as tfd

from src.utils import flatten_nn_params


def stochastic_trace_estimator_full(X, seed, num_samples=1_000, symmetrize=True):
    """
    Uses Girard-Hutchinson estimator on fully instantiated matrices.
    # ! quite unstable! Bug?
    """
    X = ensure_symmetry(X) if symmetrize else X
    def sample_eps(X, seed, num_samples): 
        return jax.random.rademacher(key=seed, shape=(num_samples, X.shape[0]))
        # return jax.random.normal(key=seed, shape=(num_samples, X.shape[0]))
    def single_estimate(X, eps):
        y = jnp.matmul(X, eps)
        return jnp.dot(eps, y)
    eps = sample_eps(X, seed=seed, num_samples=num_samples)
    estimates = jax.vmap(single_estimate, in_axes=(None,0))(X,eps)
    return estimates.mean()


def stochastic_trace_estimator_full_2(X, seed, num_samples=10):
    """
    Uses Hutch++ on fully instantiated matrices.
    """
    num_samples = num_samples if num_samples % 2 == 0 else num_samples + 1 # ensure evenness.
    def sample_eps(): 
        """Sample isotropic random vectors, either from N(0,I) or with Rademacher dist. (unif{-1,+1} indices)"""
        # return jax.random.rademacher(key=seed, shape=(num_samples, X.shape[0]))
        return jax.random.normal(key=seed, shape=(num_samples * 2, X.shape[0]))
    S,G = jnp.split(sample_eps(), 2, axis=0)
    Q,R = jnp.linalg.qr(X @ S.T)
    Q_term = (jnp.eye(Q.shape[0]) - Q@Q.T)
    estimates = jnp.trace(Q.T@X@Q) + (1/num_samples) * jnp.trace(G@Q_term@X@Q_term@G.T)
    
    return estimates.mean()


def stochastic_trace_estimator_jvp(Xfun, D, seed, num_samples=10):
    """
    Uses Hutch++ with jvp.
    Xfun: oracle computing v |-> X@v, where X: square matrix
    D: int, dim(X)
    """
    num_samples = num_samples if num_samples % 2 == 0 else num_samples + 1 # ensure evenness.
    def sample_eps(): 
        """Sample isotropic random vectors, either from N(0,I) or Rademacher dist. (i.e. unif{-1,+1} indices)"""
        # return jax.random.rademacher(key=seed, shape=(num_samples, D))
        return jax.random.normal(key=seed, shape=(num_samples * 2, D))
    S,G = jnp.split(sample_eps(), 2, axis=0)
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



def compute_ggn_vp(state, Z, w, V, model_type, full_set_size=None):
    """
    Computes GGN vector product, i.e. GGN @ v.
    @params
        Z: data points, i.e. potentially inducing points.
        w: global recalibration parameter (learned).
        V: vectors to multiply on GGN.
        model_type: "regressor"|"classifier"
        full_set_size: (if using inducing points or minibatching) size of full data set.
    """
    flat_params, unravel_fn = flatten_nn_params(state.params)
    
    def model_fun(flatp, xi):
        p_unr = unravel_fn(flatp)
        if model_type == "regressor": return state.apply_fn(p_unr, xi, return_logvar=False)
        else: return state.apply_fn(p_unr, xi)
    
    return (None, None)


def compute_full_ggn(state, Z, w, model_type, full_set_size=None):
    """
    Computes the full GGN, instantiating everything along the way.
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
        else: return state.apply_fn(p_unr, xi)

    m = Z.shape[0]
    # Initialize GGN as a zero matrix
    GGN = jnp.zeros((flat_params.shape[0], flat_params.shape[0]))

    def body_fun(i, acc):
        xi = jax.lax.dynamic_index_in_dim(Z, i, keepdims=False)
        J = jax.jacobian(lambda p: model_fun(p, xi))(flat_params)
        if model_type == "classifier":
            # ! per datum hessian for classification
            fxi = model_fun(flat_params, xi)
            probs = jax.nn.softmax(fxi)
            H_loss = jnp.diag(probs) - jnp.outer(probs, probs)
            ggn_i = J.T @ H_loss @ J
        else:
            ggn_i = J.T @ J
        return acc + ggn_i

    GGN = jax.lax.fori_loop(0, m, body_fun, GGN)
    
    if model_type == "regressor":
        # ! hessian for regression - equivalent simply to a scalar coefficient
        varinv = jnp.exp( - state.params['params']['logvar']) 
        GGN *= varinv
    
    # ! still use recalibration term?
    N = Z.shape[0]
    M = full_set_size or N
    GGN *= M / N
    GGN *= w

    return GGN, flat_params, unravel_fn
    
    
def ensure_symmetry(M, jitter=1e-8):
    return 0.5 * (M + M.T) + jitter * jnp.eye(M.shape[0]) # ! enforce symmetry of a theoretically symmetric matrix for numerical stability
