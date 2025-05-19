import pdb
import jax
import jax.numpy as jnp

from src.utils import flatten_nn_params


def compute_W_vps(state, Z, model_type, full_set_size=None, blockwise=False):
    flat_params, unravel_fn = flatten_nn_params(state.params['params'])
    M = Z.shape[0]
    N = full_set_size or M
    recal_term = jnp.sqrt( N/M ) # ! what is the correct recalibration?
    # recal_term = 1.
    
    def sqrt_Hi_apply_T(f_out, vec):
        if model_type == 'regressor':
            c = jnp.exp(-state.params['logvar']['logvar'])
            return jnp.sqrt(c) * vec
        elif model_type == 'classifier':
            # # Softmax cross-entropy Hessian = diag(p) - p p^T, PSD
            # -------- L · vec  --------
            p = jax.nn.softmax(f_out)           # (K,)
            s = jnp.sqrt(p)                     # (K,)
            tmp    = s * vec                    # diag(s) · v
            coeff  = jnp.dot(s, vec)            # sᵀ v
            return tmp - coeff * p              # tmp - (sᵀv) · p
    
    def sqrt_Hi_apply(f_out, vec):
        if model_type == 'regressor':
            c = jnp.exp(-state.params['logvar']['logvar'])
            return jnp.sqrt(c) * vec
        elif model_type == 'classifier':
            # -------- Lᵀ · vec --------
            p = jax.nn.softmax(f_out)
            s = jnp.sqrt(p)
            tmp    = s * vec                    # diag(s) · v
            coeff  = jnp.dot(p, vec)            # pᵀ v   (note the p!)
            return tmp - coeff * s              # tmp - (pᵀv) · s
            
    
    def model_fun(pflat, zi):
        p_unr = unravel_fn(pflat)
        if model_type == 'regressor':
            return state.apply_fn(p_unr, zi, return_logvar=False)
        else:
            return state.apply_fn(p_unr, zi)

    def WT_per_point(i,v):
        zi = jax.lax.dynamic_index_in_dim(Z, i, keepdims=False)
        # forward-mode JVP
        def fzi(flatp):
            return model_fun(flatp, zi).squeeze()
        _, jvp_out = jax.jvp(fzi, (flat_params,), (v,))
        f_val = fzi(flat_params)
        # apply sqrt(H_i)
        return sqrt_Hi_apply(f_val, jvp_out)
    
    def W_per_point(i, U_i):
        zi = jax.lax.dynamic_index_in_dim(Z, i, keepdims=False)

        def fzi(flatp):
            return model_fun(flatp, zi).squeeze()

        # evaluate model output at current params (for sqrt_Hi_apply)
        f_val = fzi(flat_params)
        # apply sqrt(H_i)
        h_sqrt_ui = sqrt_Hi_apply_T(f_val, U_i)
        # reverse-mode VJP => J_i^T( h_sqrt_ui )
        _, vjp_fn = jax.vjp(fzi, flat_params)
        return vjp_fn(h_sqrt_ui)[0]
    
    # ! recalibrate!
    rc_W_per_point, rc_WT_per_point = lambda *args,**kwargs: recal_term*W_per_point(*args,**kwargs), lambda *args,**kwargs: recal_term*WT_per_point(*args,**kwargs)
    
    if blockwise:
        return rc_W_per_point, rc_WT_per_point
    
    def WTfun(v):
        return jax.vmap(rc_WT_per_point, in_axes=(0,None))(jnp.arange(M), v)  # shape (M, K)

    def Wfun(U):
        # vmap over i to get an (M, d) array of per-example contributions
        per_example = jax.vmap(rc_W_per_point, in_axes=(0, 0))(jnp.arange(M), U)
        # Sum over the M dimension
        return per_example.sum(axis=0)

    return Wfun, WTfun



def compute_ggn_vp(state, Z, model_type, full_set_size=None):
    """
    Returns oracle for GGN vector product, i.e. (v |-> GGN @ v).
    @params
        Z: data points, i.e. potentially inducing points.
        w: global recalibration parameter (learned).
        V: vectors to multiply on GGN.
        model_type: "regressor"|"classifier"
        full_set_size: (if using inducing points or minibatching) size of full data set.
    """
    # flat_params, unravel_fn = jax.flatten_util.ravel_pytree(state.params['params'])
    flat_params, unravel_fn = flatten_nn_params(state.params['params'])
    M = Z.shape[0]
    N = full_set_size or M
    recal_term = N / M
    if model_type == "regressor": # handle closed form MSE hessian
        recal_term *= jnp.exp( - state.params['logvar']['logvar']) # ! just multiply the hessian scalar directly onto the recal_term
    
    def model_fun(flatp, zi):
        p_unr = unravel_fn(flatp)
        if model_type == "regressor": return state.apply_fn(p_unr, zi, return_logvar=False)
        else: return state.apply_fn(p_unr, zi)
        
    def H_action(fzi, u):
        if model_type == "classifier": # closed form softmax cross-entropy Hessian
            probs = jax.nn.softmax(fzi)
            H_loss = jnp.diag(probs) - jnp.outer(probs, probs)
            u = H_loss @ u
        elif model_type == "regressor": ... # closed form MSE - handled later on, since it reduces to a global scalar coefficient
        return u
    
    def ggn_vp(v):
        nonlocal recal_term
        total = jnp.zeros_like(flat_params)
        def body_fun(i, acc):
            zi = jax.lax.dynamic_index_in_dim(Z, i, keepdims=False)
            def fzi(flatp): return model_fun(flatp, zi).squeeze() # ! added squeeze here - maybe super bad?
            _, jvp_out = jax.jvp(fzi, (flat_params,), (v,)) # Compute the Jacobian–vector product: J_z @ v.
            f_val = fzi(flat_params)                        # Compute the model output at the current parameters.
            hv = H_action(f_val, jvp_out)                   # Apply the Hessian action: H_z @ (J_z @ v).
            _, vjp_fn = jax.vjp(fzi, flat_params)           # Compute the vector–Jacobian product: J_z^T @ (H_z @ (J_z @ v)).
            return acc + vjp_fn(hv)[0]
        return jax.lax.fori_loop(0, M, body_fun, total) * recal_term
            
    return ggn_vp


def compute_ggn_dense(state, Z, model_type, full_set_size=None):
    """
    Computes the GGN, instantiating everything along the way.
    @params
        Z: data points, i.e. potentially inducing points.
        w: global recalibration parameter (learned).
        model_type: "regressor"|"classifier"
        full_set_size: (if using inducing points or minibatching) size of full data set.
    """
    flat_params, unravel_fn = flatten_nn_params(state.params['params'])

    def model_fun(flatp, xi):
        p_unr = unravel_fn(flatp)
        if model_type == "regressor": return state.apply_fn(p_unr, xi, return_logvar=False)
        else: return state.apply_fn(p_unr, xi)

    M = Z.shape[0]
    # Initialize GGN as a zero matrix
    GGN = jnp.zeros((flat_params.shape[0], flat_params.shape[0]))

    def body_fun(i, acc):
        zi = jax.lax.dynamic_index_in_dim(Z, i, keepdims=False)
        J = jax.jacobian(lambda p: model_fun(p, zi))(flat_params)
        if model_type == "classifier":
            # ! per datum hessian for classification
            fxi = model_fun(flat_params, zi)
            probs = jax.nn.softmax(fxi)
            H_loss = jnp.diag(probs) - jnp.outer(probs, probs)
            ggn_i = J.T @ H_loss @ J
        else:
            ggn_i = J.T @ J
        return acc + ggn_i

    GGN = jax.lax.fori_loop(0, M, body_fun, GGN)
    
    if model_type == "regressor":
        # ! hessian for regression - equivalent simply to a scalar coefficient
        varinv = jnp.exp( - state.params['logvar']['logvar']) 
        GGN *= varinv
    
    # recalibration term
    N = full_set_size or M
    GGN *= N / M

    return GGN, flat_params, unravel_fn
    
    
def ensure_symmetry(M, jitter=1e-8):
    return 0.5 * (M + M.T) + jitter * jnp.eye(M.shape[0]) # ! enforce symmetry of a theoretically symmetric matrix for numerical stability
