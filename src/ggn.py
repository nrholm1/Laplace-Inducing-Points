import pdb
import jax
import jax.numpy as jnp

from src.utils import flatten_nn_params


# def compute_ggn_vp(state, Z, w, V, model_type, full_set_size=None):
#     """
#     Computes GGN vector product, i.e. GGN @ v.
#     @params
#         Z: data points, i.e. potentially inducing points.
#         w: global recalibration parameter (learned).
#         V: vectors to multiply on GGN.
#         model_type: "regressor"|"classifier"
#         full_set_size: (if using inducing points or minibatching) size of full data set.
#     """
#     flat_params, unravel_fn = flatten_nn_params(state.params)
    
#     def model_fun(flatp, xi):
#         p_unr = unravel_fn(flatp)
#         if model_type == "regressor": return state.apply_fn(p_unr, xi, return_logvar=False)
#         else: return state.apply_fn(p_unr, xi)
        
#     m = Z.shape[0]
#     # Initialize GGN as a zero matrix
#     GGN = jnp.zeros((flat_params.shape[0], flat_params.shape[0]))

#     def body_fun(i, acc):
#         xi = jax.lax.dynamic_index_in_dim(Z, i, keepdims=False)
#         jvp = jax.jvp(lambda p: model_fun(p, xi))(flat_params)
#         if model_type == "classifier":
#             # ! per datum hessian for classification
#             fxi = model_fun(flat_params, xi)
#             probs = jax.nn.softmax(fxi)
#             H_loss = jnp.diag(probs) - jnp.outer(probs, probs)
#             ggn_i = jvp(jvp(H_loss).T)
#         else:
#             ggn_i = J.T @ J
#         return acc + ggn_i

#     GGN = jax.lax.fori_loop(0, m, body_fun, GGN)
    
#     if model_type == "regressor":
#         # ! hessian for regression - equivalent simply to a scalar coefficient
#         varinv = jnp.exp( - state.params['params']['logvar']) 
#         GGN *= varinv
    
#     # ! still use recalibration term?
#     N = Z.shape[0]
#     M = full_set_size or N
#     GGN *= M / N
#     GGN *= w
    
#     return GGN, flat_params, unravel_fn


def compute_ggn_dense(state, Z, w, model_type, full_set_size=None):
    """
    Computes the GGN, instantiating everything along the way.
    @params
        Z: data points, i.e. potentially inducing points.
        w: global recalibration parameter (learned).
        model_type: "regressor"|"classifier"
        full_set_size: (if using inducing points or minibatching) size of full data set.
    """
    flat_params, unravel_fn = flatten_nn_params(state.params)

    def model_fun(flatp, xi):
        p_unr = unravel_fn(flatp)
        if model_type == "regressor": return state.apply_fn(p_unr, xi, return_logvar=False)
        else: return state.apply_fn(p_unr, xi)

    m = Z.shape[0]
    # Initialize GGN as a zero matrix
    GGN = jnp.zeros((flat_params.shape[0], flat_params.shape[0]))

    def body_fun(i, acc):
        xi = jax.lax.dynamic_index_in_dim(Z, i, keepdims=False)
        J = jax.jacobian(lambda p: model_fun(p, xi))(flat_params)
        if model_type == "classifier":
            # ! per datum hessian for classification
            fxi = model_fun(flat_params, xi)
            probs = jax.nn.softmax(fxi)
            H_loss = jnp.diag(probs) - jnp.outer(probs, probs)
            ggn_i = J.T @ H_loss @ J
        else:
            ggn_i = J.T @ J
        return acc + ggn_i

    GGN = jax.lax.fori_loop(0, m, body_fun, GGN)
    
    if model_type == "regressor":
        # ! hessian for regression - equivalent simply to a scalar coefficient
        varinv = jnp.exp( - state.params['params']['logvar']) 
        GGN *= varinv
    
    # ! still use recalibration term?
    N = Z.shape[0]
    M = full_set_size or N
    GGN *= M / N
    GGN *= w

    return GGN, flat_params, unravel_fn
    
    
def ensure_symmetry(M, jitter=1e-8):
    return 0.5 * (M + M.T) + jitter * jnp.eye(M.shape[0]) # ! enforce symmetry of a theoretically symmetric matrix for numerical stability
