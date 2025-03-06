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


def second_derivative_outputs_nll(params, xi, apply_fn):
    _, logvar = apply_fn(params, xi)
    exp_neg_logvar = jnp.exp(-logvar)
    d2L_dmu2 = exp_neg_logvar
    return jnp.array([d2L_dmu2])#.squeeze(axis=0)


# todo more (memory-)efficient implementation of all of this? :D
def compute_full_ggn(state, x, w, full_set_size=None):
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(state.params)
    varinv = jnp.exp( - state.params['params']['logvar']) # todo make dependent on loss fun - this is closed form for regression
    
    # ! constrain w all terms are nonnegative
    # w_constrained = jax.nn.softmax(w)
    # w_constrained = jax.nn.softplus(w)
    w_constrained = jnp.ones_like(w) # ! no learned weights

    def model_fun(flatp, xi):
        p_unr = unravel_fn(flatp)
        return state.apply_fn(p_unr, xi)[0]

    m = x.shape[0]
    # Initialize GGN as a zero matrix
    GGN = jnp.zeros((flat_params.shape[0], flat_params.shape[0]))

    def body_fun(i, acc):
        xi = x[i]
        # Compute the Jacobian for the current inducing point
        J = jax.jacobian(lambda p: model_fun(p, xi))(flat_params)
        # H = second_derivative_outputs_nll(unravel_fn(flat_params), xi, state.apply_fn)
        ggn_i = J.T @ J
        # Multiply by the learnable weight for this inducing point and accumulate
        return acc + w_constrained[i] * ggn_i

    GGN = varinv * jax.lax.fori_loop(0, m, body_fun, GGN)
    # GGN = GGN.at[jnp.diag_indices(GGN.shape[0])].add(1e-9) # ! (inefficient?) make always PD
    
    # ! still use recalibration term?
    N = x.shape[0]
    M = full_set_size or N
    GGN *= M / N

    return GGN, flat_params, unravel_fn
    
    
def ensure_symmetry(X, jitter=1e-8):
    return 0.5 * (X + X.T) + jitter * jnp.eye(X.shape[0]) # ! enforce symmetry of a theoretically symmetric matrix for numerical stability
