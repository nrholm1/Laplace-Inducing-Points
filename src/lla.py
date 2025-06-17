import pdb
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from src.ggn import compute_ggn_dense, compute_ggn_vp
from src.utils import flatten_nn_params
from src.sample import sample


def compute_curvature_approx(map_state, Z, model_type, alpha, full_set_size=None):
    """
    Return linear operator oracle for computing mvp with PD negative approximate Hessian of the model parameters.
    
    > J.T @ H @ J + alpha·I,
    
    where alpha is prior precision.
    """
    ggn_vp = compute_ggn_vp(map_state, Z, model_type=model_type, full_set_size=full_set_size)
    # alpha = 1.0 / (prior_precision**2)
    def curvature_vp(v):
        return ggn_vp(v) + alpha*v
    return curvature_vp


def compute_curvature_approx_dense(map_state, x, model_type, alpha, full_set_size=None):
    """
    Compute PD negative approximate Hessian of the model parameters.
    > J.T @ H @ J + alpha·I
    - Note: Instantiates dense GGN matrix.
    """
    GGN, flat_params_map, unravel_fn = compute_ggn_dense(map_state, x, model_type=model_type, full_set_size=full_set_size)
    GGN += alpha * jnp.eye(GGN.shape[0])
    return GGN, flat_params_map, unravel_fn


def posterior_lla_dense(map_state, x, model_type, alpha, full_set_size=None, return_unravel_fn=False):
    S_inv, flat_params_map, unravel_fn = compute_curvature_approx_dense(
        map_state, x, model_type=model_type, alpha=alpha, full_set_size=full_set_size
    )
    S = jnp.linalg.solve(S_inv, jnp.eye(S_inv.shape[0])) # invert
    posterior_dist = tfp.distributions.MultivariateNormalFullCovariance(
        loc=flat_params_map.astype(jnp.float64),
        covariance_matrix=S
    )
    if return_unravel_fn:
        return posterior_dist, unravel_fn
    return posterior_dist


def predict_lla_dense(map_state, Xnew, Z, model_type, alpha, full_set_size=None):
    S_inv, flat_params_map, unravel_fn = compute_curvature_approx_dense(
        map_state, Z, model_type=model_type, alpha=alpha, full_set_size=full_set_size
    )
    S = jnp.linalg.solve(S_inv, jnp.eye(S_inv.shape[0])) # invert
    
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
    
    Jnew = jax.vmap(per_datum_jacobian)(Xnew)
    f_mean = flat_apply_fn(flat_params_map, Xnew).squeeze() # ! works properly?
    
    @jax.jit
    def per_datum_cov(Ji):
        return Ji @ S @ Ji.T
    f_cov = jax.vmap(per_datum_cov)(Jnew)
    if model_type == "regressor": f_cov = jnp.diag(f_cov)
    
    return tfp.distributions.MultivariateNormalFullCovariance(
        loc=f_mean,
        covariance_matrix=f_cov
    )

def predict_la_samples_dense(
    map_state,
    Xnew: jnp.ndarray,
    Z: jnp.ndarray,
    model_type: str,
    alpha: float,
    full_set_size: int = None,
    num_mc_samples: int = 100,
    key = None,
):
    # 1) get the dense Hessian‐approx inverse & MAP‐flat vector
    S_inv, flat_params_map, unravel_fn = compute_curvature_approx_dense(
        map_state, Z,
        model_type=model_type,
        alpha=alpha,
        full_set_size=full_set_size
    )
    # invert to get covariance
    S = jnp.linalg.inv(S_inv)

    # 2) set up PRNG
    if key is None:
        key = jax.random.PRNGKey(0)
    # draw all parameter samples at once
    flat_samples = jax.random.multivariate_normal(
        key,
        mean=flat_params_map,
        cov=S,
        shape=(num_mc_samples,)
    )  # → (num_mc_samples, D_flat)

    # 3) define one‐sample forward pass
    def apply_flat(flat_p):
        p = unravel_fn(flat_p)
        if model_type == "regressor":
            # your regressor apply_fn might require a return_logvar flag
            return map_state.apply_fn(p, Xnew, return_logvar=False).squeeze()
        else:
            # classification logits
            return map_state.apply_fn(p, Xnew)            # → (Nnew, n_classes)

    # 4) vectorise over the drawn parameter vectors
    f_samples = jax.vmap(apply_flat)(flat_samples)
    # shape = (num_mc_samples, Nnew) or (num_mc_samples, Nnew, n_classes)

    return f_samples
    


def predict_lla_scalable(map_state, Xnew, Z, model_type, alpha, key=None, full_set_size=None, num_samples=1):
    flat_params, unravel_fn = flatten_nn_params(map_state.params)
    D = flat_params.shape[0]
    key = key if key is not None else jax.random.PRNGKey(123) # todo handle
    w_samples = sample(map_state, Z, D, alpha=alpha, key=key, model_type=model_type, num_samples=num_samples, full_set_size=full_set_size)
    
    @jax.jit
    def model_fun(flat_p, x):
        p = unravel_fn(flat_p)
        if model_type=="regressor":
            mu_batched = map_state.apply_fn(p, x, return_logvar=False)
        else:
            # mu_batched = map_state.apply_fn(p, x, train=False, mutable=False)
            # vars_in = {"params": p['params'], "batch_stats": map_state.batch_stats}
            vars_in = {"params": p, "batch_stats": map_state.batch_stats}
            mu_batched = map_state.apply_fn(vars_in, x, train=False, mutable=False)
        return mu_batched
    fmu = model_fun(flat_params, Xnew)
    def fz(p):
        return model_fun(p, Xnew)
    dy_fun = lambda w_sample: jax.jvp(fz, (flat_params,), (w_sample,))[1]
    dys    = jax.lax.map(dy_fun, w_samples)
    # pdb.set_trace()
    return fmu[None] + dys



def materialize_covariance(f_cov_vp, N, out_dim, mode='diag'):
    """
    Build either the diagonal or the full (N*out_dim) x (N*out_dim) 
    predictive covariance matrix from the matrix-free operator f_cov_vp.

    Parameters
    ----------
    f_cov_vp : function(U) -> jnp.ndarray
        A function that applies (J Σ J^T) to U, 
        where U has shape (N, out_dim) or flattened (N*out_dim,).
    N : int
        Number of test points in the batch.
    out_dim : int
        Output dimension (e.g. # of classes or # of regression outputs).
    mode : {'diag', 'full'}
        - 'diag' => return only the diagonal (variances), shape (N, out_dim).
        - 'full' => return the full covariance matrix, shape (N*out_dim, N*out_dim).

    Returns
    -------
    cov : jnp.ndarray
        If mode='diag', shape = (N, out_dim), the diagonal entries (variances).
        If mode='full', shape = (N*out_dim, N*out_dim), the full covariance matrix.
    """
    K = N * out_dim

    if mode == 'diag':
        # We only want the diagonal => e_i^T (J Σ J^T) e_i for i in [0..M-1].
        diag_init = jnp.zeros(K, dtype=jnp.float64)

        def body_fun(i, diag):
            # 1) Create the i-th standard basis vector of length K
            e_i = jnp.zeros(K, dtype=jnp.float64).at[i].set(1.0)
            # 2) Apply the operator -> shape (N, out_dim)
            Ae_i = f_cov_vp(e_i).reshape(K)
            # 3) The diagonal element is Ae_i[i]
            diag = diag.at[i].set(Ae_i[i])
            return diag

        diag_cov = jax.lax.fori_loop(0, K, body_fun, diag_init)
        return diag_cov.reshape((N, out_dim))

    elif mode == 'full':
        # We want the entire KxK matrix: each column is (J Σ J^T) e_i
        cov_init = jnp.zeros((K, K), dtype=jnp.float64)

        def body_fun(i, cov):
            e_i = jnp.zeros(K, dtype=jnp.float64).at[i].set(1.0)
            col_i = f_cov_vp(e_i).reshape(K)   # shape (K,)
            # Place col_i in the i-th column
            cov = cov.at[:, i].set(col_i)
            return cov

        f_cov = jax.lax.fori_loop(0, K, body_fun, cov_init)
        return f_cov

    else:
        raise ValueError("mode must be 'diag' or 'full'")
