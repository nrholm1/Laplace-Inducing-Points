import pdb
import jax
import jax.numpy as jnp
import jax.flatten_util


def per_sample_nll(params, xi, yi, apply_fn):
    """# ! Closed form NLL for regression - returns a scalar."""
    mu, logvar = apply_fn(params, xi)
    return 0.5 * (
        jnp.log(2.0 * jnp.pi * jnp.exp(logvar)) +
        (yi - mu) ** 2 / jnp.exp(logvar)
    ).squeeze()


def second_derivative_outputs_nll(params, xi, yi, apply_fn):
    """
    Compute the full 2x2 Hessian of the per-sample NLL with respect to (mu, logvar).
    For Gaussian likelihood:
        L(mu, logvar) = 0.5 * [ log(2π) + logvar + (y - mu)^2 * exp(-logvar) ]
    """
    mu, logvar = apply_fn(params, xi)
    exp_neg_logvar = jnp.exp(-logvar)
    # Second derivative components:
    d2L_dmu2 = exp_neg_logvar
    d2L_dmu_dlogvar = (yi - mu) * exp_neg_logvar
    d2L_dlogvar2 = 0.5 * (yi - mu)**2 * exp_neg_logvar
    H = jnp.array([[d2L_dmu2,      d2L_dmu_dlogvar],
                   [d2L_dmu_dlogvar, d2L_dlogvar2]])
    return H


def compute_ggn(state, x, y, full_set_size=None):
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(state.params)

    def model_outputs(flatp, xi):
        p_unr = unravel_fn(flatp)
        return state.apply_fn(p_unr, xi)
    
    # todo more (memory-)efficient implementation of all of this? :D
    jac_tuple = jax.vmap(lambda xi: jax.jacobian(lambda p: model_outputs(p, xi))(flat_params))(x)
    J = jnp.stack(jac_tuple, axis=1).squeeze()
    H = jax.vmap(lambda xi, yi: second_derivative_outputs_nll(unravel_fn(flat_params), xi, yi, state.apply_fn))(x, y).squeeze()
    
    def per_sample_ggn(Ji, Hi):
        return Ji.T @ Hi @ Ji

    GGN = jax.vmap(per_sample_ggn)(J, H).sum(axis=0)
    
    # ! Recalibration term (if necessary) 
    # todo maybe make learnable?
    N = x.shape[0]
    M = full_set_size or N
    GGN *= N / M

    return GGN, flat_params, unravel_fn
    
    
def ensure_symmetry(X, jitter=1e-8):
    return 0.5 * (X + X.T) + jitter * jnp.eye(X.shape[0]) # ! enforce symmetry of a theoretically symmetric matrix for numerical stability
