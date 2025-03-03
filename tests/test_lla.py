import pytest

import numpy as np
import jax
import jax.numpy as jnp

from src.lla import compute_curvature_approx, posterior_lla, predict_lla
from fixtures import small_model_state, regression_1d_data



def test_posterior_lla(small_model_state, regression_1d_data):
    prior_std = 1.0
    X, y = regression_1d_data
    post_dist = posterior_lla(small_model_state, prior_std, X, y)
    
    _, flat_params_map, _ = compute_curvature_approx(
        small_model_state, regression_1d_data, prior_std, return_Hinv=False
    )
    np.testing.assert_allclose(post_dist.mean(), flat_params_map, rtol=1e-4, atol=1e-6)
    
    # Check that the covariance of the posterior is positive definite.
    eigvals = jnp.linalg.eigvals(post_dist.covariance())
    assert jnp.all(eigvals > 0)


def test_predict_lla(small_model_state, regression_1d_data):
    prior_std = 1.0
    X, y = regression_1d_data
    # Define some new input points.
    xnew = jnp.array([[-0.5], [0.5]])
    pred_dist = predict_lla(small_model_state, xnew, X, y, prior_std=prior_std)
    
    # Compute the predictive mean using the MAP parameters.
    cov, flat_params_map, unravel_fn = compute_curvature_approx(
        small_model_state, regression_1d_data, prior_std, return_Hinv=False
    )
    
    def flat_apply_fn(flat_p, inputs):
        p = unravel_fn(flat_p)
        mu, _ = small_model_state.apply_fn(p, inputs)
        return mu
    pred_mean = flat_apply_fn(flat_params_map, xnew).squeeze(axis=-1)
    np.testing.assert_allclose(pred_dist.mean(), pred_mean, rtol=1e-4, atol=1e-6)
    
    # Check that the predictive covariance is positive definite.
    cov_pred = pred_dist.covariance()
    eigvals = jnp.linalg.eigvals(cov_pred)
    assert jnp.all(eigvals > 0)