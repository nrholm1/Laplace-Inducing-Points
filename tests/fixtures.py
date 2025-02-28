import pytest
import jax
import jax.numpy as jnp

@pytest.fixture
def regression_1d_data():
    """
    Returns a small 1D regression dataset (X, y).
    For instance: y ~ 2x + noise
    """
    X = jnp.array([[-1.0], [0.0], [1.0], [2.0]])
    y = 2.0 * X + 0.1 * jax.random.normal(jax.random.PRNGKey(42), shape=X.shape)
    return X, y


@pytest.fixture
def small_model_state(regression_1d_data):
    """
    Create a very small 'model' state for a 1-layer linear model.
    
    Only parameters "W" and "b" are learnable.
    The apply_fn returns a fixed logvar of 0, so that the likelihood
    is equivalent to a mean squared error loss.
    
    With fixed variance, the Gauss-Newton approximation should match
    the full Hessian (computed via jax.hessian) for this linear model.
    """
    def apply_fn(params, x):
        W = params["W"]  # scalar
        b = params["b"]  # scalar
        mu = W * x + b
        # logvar = 0.0  # Fixed logvar: always 0
        logvar = params["logvar"]
        return mu, logvar * jnp.ones_like(mu)

    key = jax.random.PRNGKey(0)
    W_init = jax.random.normal(key, ()) * 0.1
    b_init = jax.random.normal(key, ()) * 0.1
    logvar_init = jax.random.uniform(key, ()) * 0.1

    params = {
        "W": W_init,
        "b": b_init,
        "logvar": logvar_init,
    }

    class State:
        def __init__(self, params, apply_fn):
            self.params = params
            self.apply_fn = apply_fn

    return State(params, apply_fn)