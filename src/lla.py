import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from src.ggn import compute_ggn_dense, compute_ggn_vp
from src.utils import flatten_nn_params


def compute_curvature_approx(map_state, Z, model_type, prior_std, full_set_size=None):
    """
    Return linear operator oracle for computing mvp with PD negative approximate Hessian of the model parameters.
    
    > GGN = J.T @ H @ J + alpha·I,
    
    where alpha is prior precision.
    """
    ggn_vp = compute_ggn_vp(map_state, Z, model_type=model_type, full_set_size=full_set_size)
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


def predict_lla_dense(map_state, Xnew, Z, model_type, prior_std=1.0, full_set_size=None):
    S_approx, flat_params_map, unravel_fn = compute_curvature_approx_dense(
        map_state, Z, model_type=model_type, prior_std=prior_std, full_set_size=full_set_size, return_Hinv=False
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
    
    Jnew = jax.vmap(per_datum_jacobian)(Xnew)
    f_mean = flat_apply_fn(flat_params_map, Xnew).squeeze() # ! works properly?
    
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
    


# TODO I think we might eventually for large GGNs use MC sampling
def predict_lla_fun(map_state, Xnew, Z, model_type, prior_std=1.0, full_set_size=None):
    # TODO big todo: how do we do inv S_vp??? CG?
    S_vp_inv = compute_curvature_approx(map_state, Z, model_type, prior_std, full_set_size)
    def S_vp(v):
        x,info = jax.scipy.sparse.linalg.cg(A=S_vp_inv, b=v)
        return x
    
    flat_params, unravel_fn = flatten_nn_params(map_state.params['params'])
    
    @jax.jit
    def model_fun(flat_p, x):
        p = unravel_fn(flat_p)
        if model_type=="regressor":
            mu_batched = map_state.apply_fn(p, x, return_logvar=False)
        else:
            mu_batched = map_state.apply_fn(p, x)
        return mu_batched
    
    f_mean = model_fun(flat_params, Xnew)
    
    N = Xnew.shape[0]
    out_dim = f_mean.shape[1] if f_mean.ndim > 1 else 1  # for multi-output
    
    def f_cov_vp(U):
        U = U.reshape(N, out_dim)
        def fz(p):
            return model_fun(p, Xnew)
        _, vjp_fn = jax.vjp(fz, flat_params)
        v_param = vjp_fn(U)[0]
        w_param = S_vp(v_param)
        _, jvp_out = jax.jvp(fz, (flat_params,), (w_param,))
        return jvp_out
    
    return f_mean, f_cov_vp



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
    M = N * out_dim

    if mode == 'diag':
        # We only want the diagonal => e_i^T (J Σ J^T) e_i for i in [0..M-1].
        diag_init = jnp.zeros(M, dtype=jnp.float64)

        def body_fun(i, diag):
            # 1) Create the i-th standard basis vector of length M
            e_i = jnp.zeros(M, dtype=jnp.float64).at[i].set(1.0)
            # 2) Apply the operator -> shape (N, out_dim)
            Ae_i = f_cov_vp(e_i).reshape(M)
            # 3) The diagonal element is Ae_i[i]
            diag = diag.at[i].set(Ae_i[i])
            return diag

        diag_cov = jax.lax.fori_loop(0, M, body_fun, diag_init)
        return diag_cov.reshape((N, out_dim))

    elif mode == 'full':
        # We want the entire MxM matrix: each column is (J Σ J^T) e_i
        cov_init = jnp.zeros((M, M), dtype=jnp.float64)

        def body_fun(i, cov):
            e_i = jnp.zeros(M, dtype=jnp.float64).at[i].set(1.0)
            col_i = f_cov_vp(e_i).reshape(M)   # shape (M,)
            # Place col_i in the i-th column
            cov = cov.at[:, i].set(col_i)
            return cov

        f_cov = jax.lax.fori_loop(0, M, body_fun, cov_init)
        return f_cov

    else:
        raise ValueError("mode must be 'diag' or 'full'")
