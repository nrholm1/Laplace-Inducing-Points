import numpy as np
import jax
import jax.numpy as jnp
import jax.flatten_util

from src.ggn import compute_ggn_dense, compute_ggn_vp
from src.utils import flatten_nn_params, is_pd
from fixtures import regression_1d_data, small_model_state, classifier_state, classification_2d_data

def per_sample_regression_nll(params, xi, yi, apply_fn):
    """# ! Closed form NLL for regression - returns a scalar."""
    mu, logvar = apply_fn(params, xi)
    return 0.5 * (
        jnp.log(2.0 * jnp.pi * jnp.exp(logvar)) +
        (yi - mu) ** 2 / jnp.exp(logvar)
    ).squeeze()


# Test #1: GGN vs jax Hessian for a known tiny model
def test_full_ggn_vs_jax_hessian(regression_1d_data, small_model_state):
    """
    1) Compute the GGN using our implementation.
    2) Compute the full Hessian of the total negative log-likelihood via jax.hessian.
    3) Compare them. For a linear model with fixed variance, the Gauss-Newton
       approximation (GGN) should match the full Hessian within numerical tolerance.
    """
    X, y = regression_1d_data
    state = small_model_state
    w = jnp.array(1.) # jnp.ones((X.shape[0],))

    # Compute GGN using the provided function.
    GGN, *_ = compute_ggn_dense(state, X, w, model_type="regressor")

    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(state.params)

    # Define the total negative log-likelihood over the dataset.
    def total_nll(flatp):
        p_unr = unravel_fn(flatp)
        nll_vals = jax.vmap(lambda xi, yi: per_sample_regression_nll(p_unr, xi, yi, state.apply_fn))(X, y)
        return jnp.sum(nll_vals)
    
    # Compute the full Hessian via jax.hessian.
    full_hessian = jax.hessian(total_nll)(flat_params)
    full_hessian = full_hessian[:GGN.shape[0], :GGN.shape[1]] # ! extract model param Hessian block matrix
    
    # Compare the GGN with the full Hessian.
    # (Note: This equality holds in this fixed-variance linear scenario.)
    np.testing.assert_allclose(
        np.array(GGN),
        np.array(full_hessian),
        rtol=1e-2, atol=1e-3,
        err_msg="GGN does not match JAX Hessian within tolerance"
    )

# Test #2: Check the shape of the GGN
def test_full_ggn_shape(regression_1d_data, small_model_state):
    """
    Verify the shape of GGN matches the number of flattened parameters.
    """
    X, y = regression_1d_data
    state = small_model_state
    w = jnp.array(1.) # jnp.ones((X.shape[0],))

    # Call the updated compute_ggn with x, w, and y.
    GGN, flat_params, unravel_fn = compute_ggn_dense(state, X, w, model_type="regressor")
    assert GGN.shape[0] == GGN.shape[1], "GGN must be square"
    assert GGN.shape[0] == flat_params.shape[0], (
        f"GGN shape {GGN.shape} does not match #params {flat_params.shape}"
    )
    

# Test #3: GGN is positive definite
def test_full_ggn_pd(regression_1d_data, small_model_state):
    """
    1) Compute the GGN using our implementation.
    2) Check that it is PD
    """
    X, y = regression_1d_data
    state = small_model_state    
    w = jnp.array(1.) # jnp.ones((X.shape[0],))

    GGN, *_ = compute_ggn_dense(state, X, w, model_type="regressor")

    assert is_pd(GGN), "GGN is not positive definite!"
    

# Test #4: GGN vector product and identity matrix gives full GGN
def test_ggnvp_I_vs_full_ggn(regression_1d_data, small_model_state):
    """
    1) Compute GGN using GGNvp(I),
    2) Compute full GGN,
    3) Compare them.
    """
    X, y = regression_1d_data
    state = small_model_state    
    w = jnp.array(1.) # jnp.ones((X.shape[0],))
    I = jnp.identity(2)

    
    GGN_vp_fun = compute_ggn_vp(state, X, w, model_type="regressor")
    mf_GGN = jax.vmap(GGN_vp_fun, in_axes=1, out_axes=1)(I) # Matrix Free GGN
    full_GGN, *_ = compute_ggn_dense(state, X, w, model_type="regressor")
    
    assert jnp.all(jnp.isclose(mf_GGN, full_GGN, atol=1e-8)), "GGNs don't match!"
    

# Test #5: GGN vector product and identity matrix gives full GGN for 2D input space classifier
def test_ggnvp_I_vs_full_ggn_classifier(classification_2d_data, classifier_state):
    """
    Test for a classifier model:
    1) Compute GGN using the matrix-free oracle on the identity matrix,
    2) Compute the full dense GGN,
    3) Assert that they match up to a numerical tolerance.
    
    Args:
        classifier_data: A tuple (X, y) for the classifier.
        small_classifier_state: A state object with state.params and state.apply_fn
                                from a classifier (e.g., initialized via SimpleClassifier).
    """
    X, y = classification_2d_data
    state = classifier_state    
    w = jnp.array(1.)  # Global recalibration parameter

    # Obtain the dimension of the flattened parameter space.
    flat_params, _ = flatten_nn_params(state.params)
    d = flat_params.shape[0]
    I = jnp.eye(d)

    GGN_vp_fun = compute_ggn_vp(state, X, w, model_type="classifier") # Build the matrix–free GGN vector product oracle.    
    mf_GGN = jax.vmap(GGN_vp_fun, in_axes=1, out_axes=1)(I) # apply the oracle to each column of I.
    
    full_GGN, *_ = compute_ggn_dense(state, X, w, model_type="classifier")

    assert jnp.all(jnp.isclose(mf_GGN, full_GGN, atol=1e-6)), "GGNs don't match for classifier!"