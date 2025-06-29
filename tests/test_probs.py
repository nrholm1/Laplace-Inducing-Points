import numpy as np
import jax
import jax.numpy as jnp

from src.train_map import nl_posterior_fun_regression, nl_likelihood_fun_regression, nl_prior_fun
from fixtures import regression_1d_data, small_model_state


# 1. Testing the negative log likelihood (nl_likelihood_fun)
def test_nl_likelihood_fun(regression_1d_data, small_model_state):
    """
    Use a small synthetic model with known parameters to compare
    nl_likelihood_fun's output to a manual computation.
    """
    X, y = regression_1d_data
    state = small_model_state

    # Set known parameters
    state.params['params']["W"] = 2.0
    state.params['params']["b"] = 0.0
    state.params['logvar']["logvar"] = 2.3

    nll_fun_val = nl_likelihood_fun_regression(state.apply_fn, state.params, regression_1d_data)

    # Manually compute the nll
    manual_nll = 0.0
    sigma2 = jnp.exp(2.3)  # variance corresponding to logvar=2.3
    for xi, yi in zip(X, y):
        mu = 2.0 * xi + 0.0
        manual_nll += 0.5 * (jnp.log(2 * jnp.pi) + 2.3 + ((yi - mu)**2) / sigma2)

    np.testing.assert_allclose(nll_fun_val, manual_nll, rtol=1e-4, atol=1e-6)

# 2. Testing the negative log prior (nl_prior_fun)
def test_nl_prior_fun(small_model_state):
    """
    Assume a standard normal prior for each parameter.
    Then the negative log prior for a parameter theta is:
      0.5 * theta^2 + 0.5 * log(2*pi)
    and nl_prior_fun should sum this over all parameters.
    """
    state = small_model_state

    # Set known parameters, e.g., W = 2, b = 0, logvar = 0.
    state.params['params']["W"] = 2.0
    state.params['params']["b"] = 0.0
    state.params['logvar']["logvar"] = 0.0
    
    stdev = 2.3

    nl_prior_val = nl_prior_fun(state.params, stdev=stdev) # todo what should the stdev be?

    manual_nl_prior = 0.5 / (stdev**2) * (2.0**2 + 0.0**2 + 0.0**2) # + 3 * 0.5 * jnp.log(2 * jnp.pi) # ! ignore const term
    np.testing.assert_allclose(nl_prior_val, manual_nl_prior, rtol=1e-4, atol=1e-6)

# 3. Testing the full posterior (nl_posterior_fun)
def test_nl_posterior_fun(regression_1d_data, small_model_state):
    """
    Check that the posterior function equals the sum of the likelihood and prior.
    """
    state = small_model_state

    # Set known parameters.
    state.params['params']["W"] = 2.0
    state.params['params']["b"] = 0.0
    state.params['logvar']["logvar"] = 0.0
    
    stdev = 2.3

    nll_val = nl_likelihood_fun_regression(state.apply_fn, state.params, regression_1d_data)
    prior_val = nl_prior_fun(state.params, stdev=stdev) # todo what should stdev be?
    posterior_val = nl_posterior_fun_regression(state, state.params, regression_1d_data, prior_precision=1.0/stdev**2)
    
    np.testing.assert_allclose(posterior_val, nll_val + prior_val, rtol=1e-4, atol=1e-6)


# 4. Verifying the learned variance term's effect.
def test_learned_variance_effect(regression_1d_data, small_model_state):
    """
    Verify that changing the logvar parameter has an expected effect on the likelihood.
    For a given fixed model (W, b), the negative log likelihood is
       0.5*(log(2*pi) + logvar + (y - (Wx+b))^2 / exp(logvar)).
    Changing logvar should change this value in a predictable manner.
    """
    state = small_model_state

    # Fix W and b.
    state.params['params']["W"] = 2.0
    state.params['params']["b"] = 0.0

    # Compute nll for two different logvar values.
    state.params['logvar']["logvar"] = -1.0  # variance = exp(-1) ~ 0.37
    nll_low = nl_likelihood_fun_regression(state.apply_fn, state.params, regression_1d_data)
    state.params['logvar']["logvar"] = 1.0   # variance = exp(1) ~ 2.72
    nll_high = nl_likelihood_fun_regression(state.apply_fn, state.params, regression_1d_data)

    # They should be different. (You could further manually compute the expected change,
    # but here we assert that the outputs differ.)
    assert not np.allclose(nll_low, nll_high)