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
    # d2L_dmu_dlogvar = (yi - mu) * exp_neg_logvar
    d2L_dlogvar2 = 0.5 * (yi - mu)**2 * exp_neg_logvar
    # H = jnp.array([[d2L_dmu2,      d2L_dmu_dlogvar],
    #                [d2L_dmu_dlogvar, d2L_dlogvar2]])
    H = jnp.array([[d2L_dmu2,      jnp.array([0.])],
                   [jnp.array([0.]), d2L_dlogvar2]]) # todo: major fix needed here! This is a hack
    # H = jnp.eye(2)
    return H


# todo more (memory-)efficient implementation of all of this? :D
def compute_ggn(state, x, w, y=None, full_set_size=None):
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(state.params)
    
    # ! constrain w all terms are nonnegative
    w_constrained = jax.nn.softmax(w)
    # w_constrained = jax.nn.softplus(w)

    def model_fun(flatp, xi):
        p_unr = unravel_fn(flatp)
        return state.apply_fn(p_unr, xi)

    m = x.shape[0]
    # Initialize GGN as a zero matrix
    GGN = jnp.zeros((flat_params.shape[0], flat_params.shape[0]))

    def body_fun(i, acc):
        xi = x[i]
        # Compute the Jacobian for the current inducing point
        jac_tuple = jax.jacobian(lambda p: model_fun(p, xi))(flat_params)
        J = jnp.stack(jac_tuple, axis=1).squeeze()
        if y is not None:
            yi = y[i]
            H = second_derivative_outputs_nll(unravel_fn(flat_params), xi, yi, state.apply_fn).squeeze()
            ggn_i = J.T @ H @ J
        else:
            ggn_i = J.T @ J
        # Multiply by the learnable weight for this inducing point and accumulate
        return acc + w_constrained[i] * ggn_i

    GGN = jax.lax.fori_loop(0, m, body_fun, GGN)

    return GGN, flat_params, unravel_fn
    
    
def ensure_symmetry(X, jitter=1e-8):
    return 0.5 * (X + X.T) + jitter * jnp.eye(X.shape[0]) # ! enforce symmetry of a theoretically symmetric matrix for numerical stability
