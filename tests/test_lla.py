import pytest

import numpy as np
import jax
import jax.numpy as jnp

from src.lla import compute_curvature_approx_dense, posterior_lla_dense, predict_lla_dense, predict_lla_fun, materialize_covariance
from tests.fixtures import small_model_state, regression_1d_data, toyregressor_state, sine_data, classifier_state, classification_2d_data
from src.utils import flatten_nn_params


def test_posterior_lla(small_model_state, regression_1d_data):
    prior_std = 1.0
    X, y = regression_1d_data

    post_dist = posterior_lla_dense(small_model_state, X, prior_std=prior_std, model_type="regressor")
    
    _, flat_params_map, _ = compute_curvature_approx_dense(
        small_model_state, X, prior_std=prior_std, model_type="regressor", return_Hinv=False
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
    pred_dist = predict_lla_dense(small_model_state, xnew, X, model_type="regressor", prior_std=prior_std)
    
    # Compute the predictive mean using the MAP parameters.
    cov, flat_params_map, unravel_fn = compute_curvature_approx_dense(
        small_model_state, X, prior_std=prior_std, model_type="regressor", return_Hinv=False
    )
    
    def flat_apply_fn(flat_p, inputs):
        p = unravel_fn(flat_p)
        return small_model_state.apply_fn(p, inputs, return_logvar=False)
    pred_mean = flat_apply_fn(flat_params_map, xnew).squeeze(axis=-1)
    np.testing.assert_allclose(pred_dist.mean(), pred_mean, rtol=1e-4, atol=1e-6)
    
    # Check that the predictive covariance is positive definite.
    cov_pred = pred_dist.covariance()
    eigvals = jnp.linalg.eigvals(cov_pred)
    assert jnp.all(eigvals > 0)
    
    
def test_predict_lla_jvp(toyregressor_state, sine_data):
    jax.config.update("jax_enable_x64", True) # 64 bit floats
    
    train_loader, test_loader = sine_data
    X,y = next(iter(test_loader))
    N = X.shape[0]
    
    # convert stuff to f64
    X = X.astype(jnp.float64)
    y = y.astype(jnp.float64)
    state = toyregressor_state
    state = state.replace(
        params=jax.tree_util.tree_map(lambda param: param.astype(jnp.float64), state.params)
    )
    
    flat_params, _ = flatten_nn_params(state.params)
    D = flat_params.shape[0]
    
    xnew = jnp.array([[-.5], [.5]])
    
    pred_dist = predict_lla_dense(state, xnew, X, model_type="regressor", prior_std=1.0)
    f_mean_dense = pred_dist.mean()
    f_cov_dense = pred_dist.covariance()
    
    f_mean, f_cov_vp = predict_lla_fun(state, xnew, X, model_type="regressor", prior_std=1.0)
    I = jnp.eye(xnew.shape[0])
    
    f_cov_mf = jnp.diag( materialize_covariance(f_cov_vp, *f_mean.shape, mode='diag').squeeze() )
    assert jnp.all( jnp.isclose(f_cov_dense, f_cov_mf, atol=1e-8))


def test_predict_lla_jvp_classifier(classifier_state, classification_2d_data):
    jax.config.update("jax_enable_x64", True) # 64 bit floats
    
    X,y = classification_2d_data
    N = X.shape[0]
    
    # convert stuff to f64
    X = X.astype(jnp.float64)
    y = y.astype(jnp.float64)
    state = classifier_state
    state = state.replace(
        params=jax.tree_util.tree_map(lambda param: param.astype(jnp.float64), state.params)
    )
    
    flat_params, _ = flatten_nn_params(state.params)
    D = flat_params.shape[0]
    
    xnew = jnp.array([[-.5, .5], [1.0, -1.5], [2.0, 2.0]])
    
    f_mean, f_cov_vp = predict_lla_fun(state, xnew, X, model_type="classifier", prior_std=1.0)
    
    diag = materialize_covariance(f_cov_vp, *f_mean.shape, mode='diag')
    full = materialize_covariance(f_cov_vp, *f_mean.shape, mode='full')
    
    assert jnp.all( jnp.linalg.eigvals(full) > 0. ), "Covariance should be PSD!"
    assert jnp.all( jnp.isclose(full, full.T, rtol=1e-4, atol=1e-12)), "Covariance should be symmetric up to a numerical error!"
    