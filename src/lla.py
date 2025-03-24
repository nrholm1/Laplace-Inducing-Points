import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from src.ggn import compute_ggn_dense, compute_ggn_vp


def compute_curvature_approx(map_state, x, model_type, prior_std, full_set_size=None):
    """
    Return linear operator oracle for computing mvp with PD negative approximate Hessian of the model parameters.
    
    > GGN = J.T @ H @ J + alpha·I,
    
    where alpha is prior precision.
    """
    ggn_vp = compute_ggn_vp(map_state, x, model_type=model_type, full_set_size=full_set_size)
    alpha = 1.0 / (prior_std**2)
    def curvature_vp(v):
        return ggn_vp(v) + alpha*v
    return curvature_vp


def compute_curvature_approx_dense(map_state, x, model_type, prior_std, full_set_size=None, return_Hinv=True):
    """
    Compute PD negative approximate Hessian of the model parameters.
    > GGN = J.T @ H @ J + alpha·I
    - Note: Instantiates dense GGN matrix.
    """
    GGN, flat_params_map, unravel_fn = compute_ggn_dense(map_state, x, model_type=model_type, full_set_size=full_set_size)
    prior_precision = 1.0 / (prior_std**2)
    GGN += prior_precision * jnp.eye(GGN.shape[0])
    if return_Hinv:
        return GGN, flat_params_map, unravel_fn
    else:
        return jnp.linalg.inv(GGN), flat_params_map, unravel_fn


def posterior_lla_dense(map_state, x, model_type, prior_std=1.0, full_set_size=None, return_unravel_fn=False):
    S_approx, flat_params_map, unravel_fn = compute_curvature_approx_dense(
        map_state, x, model_type=model_type, prior_std=prior_std, full_set_size=full_set_size, return_Hinv=False
    )
    posterior_dist = tfp.distributions.MultivariateNormalFullCovariance(
        loc=flat_params_map.astype(jnp.float64),
        covariance_matrix=S_approx
    )
    if return_unravel_fn:
        return posterior_dist, unravel_fn
    return posterior_dist


def predict_lla_dense(map_state, xnew, x, model_type, prior_std=1.0, full_set_size=None):
    S_approx, flat_params_map, unravel_fn = compute_curvature_approx_dense(
        map_state, x, model_type=model_type, prior_std=prior_std, full_set_size=full_set_size, return_Hinv=False
    )
    
    @jax.jit
    def flat_apply_fn(flat_p, inputs):
        p = unravel_fn(flat_p)
        if model_type=="regressor":
            mu_batched = map_state.apply_fn(p, inputs, return_logvar=False)
        else:
            mu_batched = map_state.apply_fn(p, inputs)
        return mu_batched
    
    @jax.jit
    def per_datum_jacobian(xi):
        return jax.jacobian(lambda fp: flat_apply_fn(fp, xi[None]).squeeze())(flat_params_map)
    
    Jnew = jax.vmap(per_datum_jacobian)(xnew)
    f_mean = flat_apply_fn(flat_params_map, xnew).squeeze() # ! works properly?
    
    @jax.jit
    def per_datum_cov(Ji):
        return Ji @ S_approx @ Ji.T
    f_cov = jax.vmap(per_datum_cov)(Jnew)
    if model_type == "regressor": f_cov = jnp.diag(f_cov)
    
    assert jnp.all(jnp.linalg.eigvals(f_cov) > 0), "Covariance matrix not PD!" # ! expensive?
    
    return tfp.distributions.MultivariateNormalFullCovariance(
        loc=f_mean,
        covariance_matrix=f_cov
    )