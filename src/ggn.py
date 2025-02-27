import jax
import jax.numpy as jnp


def compute_ggn_old(state, x, full_set_size=None):
    """
    Computes a potentially scaled version of the GGN (for likelihood, i.e. w/o alpha*I).
    """
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(state.params)
    
    def per_datum_jacobian(xi):
        def scalar_output(flat_p):
            p = unravel_fn(flat_p)
            return state.apply_fn(p, xi[None]).squeeze()
        return jax.jacobian(scalar_output)(flat_params)

    # todo add hessian term for classification!
    Js = jax.vmap(per_datum_jacobian)(x)
    JtJ = jnp.einsum('ni,nj->ij', Js, Js)
    
    # rescaling weight if subsampled/inducing points
    N = x.shape[0]
    M = full_set_size or N
    GGN = N/M * JtJ
    
    return GGN, flat_params, unravel_fn


def per_sample_nll(params, x_i, y_i, apply_fn):
    """#! closed form for regression - returns scalar """
    mu, logvar = apply_fn(params, x_i[None])
    return 0.5 * (
        jnp.log(2.0*jnp.pi*jnp.exp(logvar)) +
        (y_i - mu)**2 / jnp.exp(logvar)
    ).squeeze()


def compute_ggn(state, x, y, full_set_size=None):
    """
    Computes a potentially scaled version of the GGN (for likelihood, i.e. w/o alpha*I).
    """
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(state.params)

    def per_sample_grad(flatp, xi, yi):
       def nll_wrt_flatp(p):
           p_unr = unravel_fn(p)
           return per_sample_nll(p_unr, xi, yi, state.apply_fn)
       return jax.grad(nll_wrt_flatp)(flatp)
   
    grads = jax.vmap(per_sample_grad, in_axes=(None, 0, 0))(flat_params, x, y)
    GGN = grads.T @ grads
    
    N = x.shape[0]
    M = full_set_size or N
    GGN *= N/M
    
    return GGN, flat_params, unravel_fn
    
    
def ensure_symmetry(X, jitter=1e-8):
    return 0.5 * (X + X.T) + jitter * jnp.eye(X.shape[0]) # ! enforce symmetry of a theoretically symmetric matrix for numerical stability
