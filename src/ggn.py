import pdb
import jax
import jax.numpy as jnp

from src.utils import flatten_nn_params



# todo more (memory-)efficient implementation of all of this? :D
def compute_full_ggn(state, x, w, model_type, loss_fn=None, full_set_size=None):
    flat_params, unravel_fn = flatten_nn_params(state.params)

    def model_fun(flatp, xi):
        p_unr = unravel_fn(flatp)
        if model_type == "regressor": return state.apply_fn(p_unr, xi, return_logvar=False)
        else: return state.apply_fn(p_unr, xi)
    

    m = x.shape[0]
    # Initialize GGN as a zero matrix
    GGN = jnp.zeros((flat_params.shape[0], flat_params.shape[0]))

    def body_fun(i, acc):
        xi = jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
        J = jax.jacobian(lambda p: model_fun(p, xi))(flat_params)
        if loss_fn is not None:
            fxi = model_fun(flat_params, xi)
            H_loss = jax.hessian(loss_fn)(fxi)
            ggn_i = J.T @ H_loss @ J
        else:
            if model_type == "classifier":
                # todo make dependent on loss fun - this is closed form for classification
                fxi = model_fun(flat_params, xi)
                probs = jax.nn.softmax(fxi)
                H_loss = jnp.diag(probs) - jnp.outer(probs, probs)
                ggn_i = J.T @ H_loss @ J
            else:
                ggn_i = J.T @ J
        return acc + ggn_i

    GGN = jax.lax.fori_loop(0, m, body_fun, GGN)
    
    if model_type == "regressor":
        # ! Hessian term!
        # todo make dependent on loss fun - this is closed form for regression
        varinv = jnp.exp( - state.params['params']['logvar']) 
        GGN *= varinv
    
    # ! still use recalibration term?
    N = x.shape[0]
    M = full_set_size or N
    GGN *= M / N
    GGN *= w

    return GGN, flat_params, unravel_fn
    
    
def ensure_symmetry(X, jitter=1e-8):
    return 0.5 * (X + X.T) + jitter * jnp.eye(X.shape[0]) # ! enforce symmetry of a theoretically symmetric matrix for numerical stability
