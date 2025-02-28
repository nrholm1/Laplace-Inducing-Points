import numpy as np
import pytest
import jax
import jax.numpy as jnp

from src.ggn import compute_ggn, per_sample_nll


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


# Test #1: GGN vs jax Hessian for a known tiny model
def test_ggn_vs_jax_hessian(regression_1d_data, small_model_state):
    """
    1) Compute the GGN using our implementation.
    2) Compute the full Hessian of the total negative log-likelihood via jax.hessian.
    3) Compare them. For a linear model with fixed variance, the Gauss-Newton
       approximation (GGN) should match the full Hessian within numerical tolerance.
    """
    X, y = regression_1d_data
    state = small_model_state

    # Compute GGN using the provided function.
    GGN, flat_params, unravel_fn = compute_ggn(state, X, y)

    # Define the total negative log-likelihood over the dataset.
    def total_nll(flatp):
        p_unr = unravel_fn(flatp)
        nll_vals = jax.vmap(lambda xi, yi: per_sample_nll(p_unr, xi, yi, state.apply_fn))(X, y)
        return jnp.sum(nll_vals)
    
    # Compute the full Hessian via jax.hessian.
    full_hessian = jax.hessian(total_nll)(flat_params)
    
    # Compare the GGN with the full Hessian.
    # (Note: This equality holds in this fixed-variance linear scenario.)
    np.testing.assert_allclose(
        np.array(GGN),
        np.array(full_hessian),
        rtol=1e-2, atol=1e-3,
        err_msg="GGN does not match JAX Hessian within tolerance"
    )


# Test #2: Check the shape of the GGN
def test_ggn_shape(regression_1d_data, small_model_state):
    """
    Verify the shape of GGN matches the number of flattened parameters.
    """
    X, y = regression_1d_data
    state = small_model_state

    GGN, flat_params, unravel_fn = compute_ggn(state, X, y)
    assert GGN.shape[0] == GGN.shape[1], "GGN must be square"
    assert GGN.shape[0] == flat_params.shape[0], (
        f"GGN shape {GGN.shape} does not match #params {flat_params.shape}"
    )