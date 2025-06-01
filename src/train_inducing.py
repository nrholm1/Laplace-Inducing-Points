from functools import partial
import functools
import pdb
# import pdb
import jax
import jax.numpy as jnp
import jax.flatten_util
from matplotlib import pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

from matfree import decomp, funm, stochtrace as matfree_stochtrace
from src.matfree_monkeypatch import integrand_funm_sym_logdet

from src.lla import posterior_lla_dense, compute_curvature_approx_dense, compute_curvature_approx
from src.ggn import compute_W_vps
from src.train_map import nl_likelihood_fun_regression
from src.utils import count_model_params
from src.toydata import plot_binary_classification_data
from src.data import make_iter
from src.nplot import scatterp, plot_mnist


def sample_params(mu, cov, rng, num_samples=1):
    eps = jax.random.normal(rng, shape=mu.shape)
    # Cholesky to map eps ~ N(0,I) -> theta ~ N(mu, cov)
    L = jnp.linalg.cholesky(cov)
    return mu + L @ eps


def var_loglik_fun(q, batch, apply_fn, unravel_fn, rng, num_mc_samples):
    """
    # ! using MC sample(s) of parameters
    """
    mu, cov = q.mean(), q.covariance()
    log_sum = 0.0
    for i in range(num_mc_samples):
        rng_i = jax.random.fold_in(rng, i)  # make a fresh key
        theta_sample = sample_params(mu, cov, rng_i)
        theta_sample = unravel_fn(theta_sample)
        log_p_data = -nl_likelihood_fun_regression(apply_fn, theta_sample, batch) # ! nll, therefore negate it...
        log_sum += log_p_data
    return log_sum / num_mc_samples


def var_kl_fun(q, alpha):
    mu, cov = q.mean(), q.covariance()
    D = cov.shape[0]
    tr_term = alpha * jnp.linalg.trace(cov)
    norm_term = alpha * jnp.linalg.norm(mu) ** 2
    logdetp_term = jnp.log(D / alpha + 1e-9)  # log(det( I * alpha^(-1) ))
    logdetq_term = jnp.log(jnp.linalg.det(cov) + 1e-9)  # todo fix: hack = add epsilon term
    kl_term = 0.5 * (tr_term - D + norm_term + logdetp_term - logdetq_term)
    return kl_term


def naive_objective(z, dataset, state, alpha, rng, num_mc_samples, model_type, full_set_size=None):
    q, unravel_fn = posterior_lla_dense(
        state,
        alpha=alpha,
        x=z,
        model_type=model_type,
        full_set_size=full_set_size,
        return_unravel_fn=True)
    loglik_term = var_loglik_fun(q, dataset, state.apply_fn, unravel_fn, rng, num_mc_samples=num_mc_samples)
    kl_term = var_kl_fun(q, alpha)
    reg_term = 0 # ! reg_coeff * (jnp.sum(jnp.square(x)) + jnp.sum(jnp.square(w)))
    return - (loglik_term - kl_term) + reg_term


def build_WTW(W, WT, inner_shape, d, *, dtype=jnp.bfloat16, block=64):
    """
    Return WᵀW ∈ R^{dxd} with ≤ (block · #params) peak memory.
    """
    @functools.partial(jax.remat, static_argnums=1)          # k is static
    def col_block(start, k):
        rows = start + jnp.arange(k, dtype=jnp.int32)        # shape (k,)
        E    = jax.nn.one_hot(rows, d, dtype=dtype)\
                  .reshape((k,) + inner_shape)               # (k, M, C)
        cols = jax.vmap(lambda e: WT(W(e)).reshape(-1))(E)   # (k, d)
        return cols.astype(dtype)                            # (k, d)

    WTW = jnp.zeros((d, d), dtype=dtype)

    n_full, tail = divmod(d, block)

    def body(b, acc):
        start = b * block
        cols  = col_block(start, block)      # (block, d)
        return jax.lax.dynamic_update_slice(acc, cols.T, (0, start))

    WTW = jax.lax.fori_loop(0, n_full, body, WTW)

    if tail:
        start  = n_full * block
        cols_t = col_block(start, tail).T    # (d, tail)
        WTW    = jax.lax.dynamic_update_slice(WTW, cols_t, (0, start))

    return jnp.triu(WTW) + jnp.triu(WTW, 1).T



def alternative_objective_scalable(Z, X, state, alpha, model_type, key, full_set_size=None,
                                   st_samples=256, slq_samples=2, slq_num_matvecs=None):
    """ MATRIX FREE
    =========================================
    Compute KL[ q(theta|Z) || p(theta|data) ]
    =========================================
    """
    N = full_set_size
    M = Z.shape[0]
    beta = N / M
    alpha_inv = 1.0 / alpha
    beta_inv = 1.0 / beta
    
    D = count_model_params(state.params)
    if model_type == 'regressor':
        D -= 1 # ! subtract logvar parameter!
    
    # compute matrix free linear operator oracles
    S_vp  = compute_curvature_approx(
        state, X, alpha=alpha, model_type=model_type, 
        full_set_size=N)
    Sz_vp = compute_curvature_approx(
        state, Z, alpha=alpha, model_type=model_type, 
        full_set_size=N)
    W, WT = compute_W_vps(
        state, Z, model_type=model_type, 
        full_set_size=None)
    
    
    # ! option 1: use conjugate gradient 
    def Sz_inv_vp_direct(v):
        x,info = jax.scipy.sparse.linalg.cg(A=Sz_vp, b=v)
        return x
    
    # ! option 2: woodbury matrix identity
    def Sz_inv_vp_woodbury_cg(v):
        def inner(u): 
            return u + alpha_inv*WT(W(u))
        u = WT(v)
        x,info = jax.scipy.sparse.linalg.cg(A=inner, b=u)
        return alpha_inv*v - alpha_inv**2*W(x)
    
    dummy = WT(jnp.zeros(D))
    inner_shape = dummy.shape
    d           = dummy.size
    I_d         = jnp.eye(d, dtype=float)
    # WTW = jax.vmap(lambda e: 
    #         WT(W(e.reshape(inner_shape)))
    #     )(I_d).reshape(d,d)
    WTW = build_WTW(W, WT, inner_shape, d, dtype=float, block=2) # ! build dense WTW in blocks to lower memory pressure
    def Sz_inv_vp_woodbury_dense(v):
        u = WT(v).reshape(d)
        x   = jax.scipy.linalg.solve(
            beta_inv*I_d + alpha_inv*WTW,
            u)
        return alpha_inv*v - alpha_inv**2*W(x.reshape(inner_shape))
    
    def composite_vp(v):
        # return S_vp(Sz_inv_vp_direct(v))
        return S_vp(Sz_inv_vp_woodbury_dense(v))
        # return S_vp(Sz_inv_vp_woodbury_cg(v))
    
    # ! use same random vectors for StochTrace and SLQ
    x0 = jnp.ones((D,), dtype=float)
    sampler = matfree_stochtrace.sampler_rademacher(x0, num=st_samples)
    probes = sampler(key)
    st_sampler =  lambda _: probes
    slq_sampler = lambda _: probes[:slq_samples]
    
    def stoch_trace(Xfun):
        integrand = matfree_stochtrace.integrand_trace()
        estimator = matfree_stochtrace.estimator(integrand, sampler=st_sampler)
        traces = estimator(Xfun, key)
        return traces
    trace_term = stoch_trace(composite_vp) 
    
    # ! use stochastic Lanczos quadrature
    slq_num_matvecs = slq_num_matvecs if slq_num_matvecs is not None else int(M*0.8)
    def slq_logdet(Xfun):
        # adapted directly from https://pnkraemer.github.io/matfree/Tutorials/1_compute_log_determinants_with_stochastic_lanczos_quadrature/
        # tridiag_sym = decomp.tridiag_sym(slq_num_matvecs)
        # problem = integrand_funm_sym_logdet(tridiag_sym)
        bidiag_sym = decomp.bidiag(slq_num_matvecs)
        problem = funm.integrand_funm_product_logdet(bidiag_sym)
        estimator = matfree_stochtrace.estimator(problem, sampler=slq_sampler)
        logdets = estimator(Xfun, key)
        return logdets
    # logdet_term = slq_logdet(Sz_vp)
    
    sqrt_alpha = jnp.sqrt(alpha)
    def bidiag_target(v):
        x = WT(v)
        xflat, unravel_fn = jax.flatten_util.ravel_pytree(x)
        return jnp.concatenate([sqrt_alpha*v, xflat])#, unravel_fn
    
    logdet_term = slq_logdet(bidiag_target)
    
    return logdet_term + trace_term


def alternative_objective_dense(Z, X, state, alpha, model_type, key, full_set_size=None):
    """
    =========================================
    Compute KL[ q(theta|Z) || p(theta|data) ]
    =========================================
    """
    S, *_ = compute_curvature_approx_dense(state, X, alpha=alpha, model_type=model_type, full_set_size=full_set_size)
    S_z,    *_ = compute_curvature_approx_dense(state, Z, alpha=alpha, model_type=model_type, full_set_size=full_set_size)
    S_z_inv = jnp.linalg.inv(S_z)
    
    trace_term = jnp.linalg.trace(S @ S_z_inv)
    
    _, S_logdet = 0., 0. # jnp.linalg.slogdet(S)
    _, S_z_inv_logdet = jnp.linalg.slogdet(S_z_inv)
    logdet_term = - S_logdet - S_z_inv_logdet
    
    return logdet_term + trace_term


variational_grad_dense = jax.value_and_grad(alternative_objective_dense)
variational_grad_scalable = jax.value_and_grad(alternative_objective_scalable)


@partial(jax.jit, static_argnames=('alpha', 'model_type', 'zoptimizer', 'num_mc_samples', 'full_set_size', 'scalable', 'st_samples', 'slq_samples', 'slq_num_matvecs'))
def optimize_step(Z, X, map_model_state, alpha, opt_state, rng, zoptimizer, num_mc_samples, model_type, full_set_size=None, scalable=True,
                  st_samples=256, slq_samples=2, slq_num_matvecs=None):
    if scalable:
        grad_fun = variational_grad_scalable  
        loss, grads = grad_fun(
            Z, 
            X, 
            map_model_state, 
            alpha, 
            key=rng,
            # num_mc_samples=num_mc_samples,
            model_type=model_type, 
            full_set_size=full_set_size,
            st_samples=st_samples, 
            slq_samples=slq_samples, 
            slq_num_matvecs=slq_num_matvecs
        )
    else: 
        grad_fun = variational_grad_dense
        loss, grads = grad_fun(
            Z, 
            X, 
            map_model_state, 
            alpha, 
            key=rng,
            # num_mc_samples=num_mc_samples,
            model_type=model_type, 
            full_set_size=full_set_size,
        )
    updates, new_opt_state = zoptimizer.update(grads, opt_state)  # ? ADAM, SGD, etc.
    # updates, new_opt_state = zoptimizer.update(grads, opt_state, Z) # ? ADAMW
    new_params = optax.apply_updates(Z, updates)
    return new_params, new_opt_state, loss


def train_inducing_points(map_model_state, zinit, zoptimizer, dataloader, model_type, rng, num_mc_samples, alpha, num_steps, full_set_size, scalable, plot_full_dataset_fn_debug=None,
                          st_samples=256, slq_samples=2, slq_num_matvecs=None):
    z = zinit
    opt_state = zoptimizer.init(z)
    # _iter = iter(dataloader)
    _iter = make_iter(dataloader)
    
    def get_next_sample(num_batches=1):
        nonlocal _iter 
        sample_batches = []
        for _ in range(num_batches):
            try:
                batch = next(_iter)
            except StopIteration:
                _iter = make_iter(dataloader)
                batch = next(_iter)
            sample_batches.append(batch)
        sample = list(zip(*sample_batches))
        sample = (jnp.concatenate(sample[0], axis=0), jnp.concatenate(sample[1], axis=0))
        return sample
    
    # ! bounding box for constraining inducing point range
    dataset_sample = get_next_sample(num_batches=32)[0]
    lb = dataset_sample.min(axis=0)
    ub = dataset_sample.max(axis=0)
    # z = z*dataset_sample.std() + dataset_sample.mean()
    del dataset_sample
    # z = jnp.clip(z, lb, ub)
    
    # ? for debugging
    # fig, ax = plt.subplots(figsize=(12, 8))
    # trajectory = [] 
    
    pbar = tqdm(range(num_steps))
    for step in pbar:
        dataset_sample = get_next_sample(num_batches=1)
        x_sample,y_sample = dataset_sample
        
        # ! Common Random Numbers - does it work???
        if step % 2 == 0:
            rng = jax.random.fold_in(rng, step) # ? TEST holding probes constant
        
        z, opt_state, loss = optimize_step(
            z, 
            x_sample, 
            map_model_state=map_model_state, 
            alpha=alpha, 
            opt_state=opt_state, 
            rng=rng,
            model_type=model_type,
            zoptimizer=zoptimizer, 
            num_mc_samples=num_mc_samples, 
            full_set_size=full_set_size,
            scalable=scalable,
            st_samples=st_samples, 
            slq_samples=slq_samples, 
            slq_num_matvecs=slq_num_matvecs
        )
        # pdb.set_trace()
        # ! Enforce constraints on z
        # z = jnp.clip(z,0,1)
        # z = jnp.clip(z, 
        #              lb*(1 - 0.5*jnp.sign(lb)),
        #              ub*(1 + 0.5*jnp.sign(ub))
        #     )
        
        pbar.set_description_str(f"Loss: {loss:.3f}", refresh=True)
        
        # todo for debug: every 2 steps, record & plot
        if step % 4 == 0:
            z_np = np.asarray(z)
            plot_mnist(z_np[:32].squeeze(), step)
            
            # trajectory.append(z_np)

            # traj = np.stack(trajectory)    # shape (n_points, 2)
            # ax.clear()
            # ax.plot(traj[:, :, 0], traj[:,:, 1], '-o', color="black", markersize=2, zorder=7)
            # # plot_full_dataset_fn_debug()
            # ax.set_xlim(lb[0] - 1.0, ub[0] + 1.0)
            # ax.set_ylim(lb[1] - 1.0, ub[1] + 1.0)
            # ax.set_xlabel('z[0]')
            # ax.set_ylabel('z[1]')
            # ax.set_title(f'Latent Trajectory after {step} steps')
            # scatterp(*z_np.T, color="yellow", zorder=8, marker="X", label="Inducing points")
            # plot_binary_classification_data(dataset_sample[0], dataset_sample[1].squeeze())
            
            # # force a draw
            # fig.canvas.draw()
            # fig.canvas.flush_events()
            # plt.savefig(f"fig/toy/test.png")
            
            # trajectory = trajectory[-3:]
        
    
    return z
