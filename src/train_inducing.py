from functools import partial
import functools
import pdb
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

from matfree import decomp, funm, stochtrace as matfree_stochtrace

from src.lla import posterior_lla_dense, compute_curvature_approx_dense, compute_curvature_approx
from src.stochtrace import hutchpp_mvp, na_hutchpp_mvp, stochastic_trace_estimator_mvp
from src.train_map import nl_likelihood_fun_regression
from src.utils import count_model_params
from src.toydata import plot_binary_classification_data
from src.nplot import scatterp


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
        alpha=alpha,# todo correct alpha used here?
        x=z,
        model_type=model_type,
        full_set_size=full_set_size,
        return_unravel_fn=True
    )

    loglik_term = var_loglik_fun(q, dataset, state.apply_fn, unravel_fn, rng, num_mc_samples=num_mc_samples)
    kl_term = var_kl_fun(q, alpha)
    reg_term = 0 # ! reg_coeff * (jnp.sum(jnp.square(x)) + jnp.sum(jnp.square(w)))

    return - (loglik_term - kl_term) + reg_term


def alternative_objective_scalable(z, x, state, alpha, model_type, key, full_set_size=None):
    M = z.shape[0]
    D = count_model_params(state.params)
    if model_type == 'regressor': # todo: handle this better?
        D -= 1 # ! subtract logvar parameter!
    
    # compute matrix free linear operator oracles
    print("Computing curvature")
    S_full_vp  = compute_curvature_approx(state, x, alpha=alpha, model_type=model_type, full_set_size=full_set_size)
    S_induc_vp = compute_curvature_approx(state, z, alpha=alpha, model_type=model_type, full_set_size=full_set_size) # todo something with the beta term?? Not correct, I believe.
    print("Finished computing curvature")
    
    # ! option 1: use conjugate gradient 
    # @jax.jit
    def S_induc_inv_vp(v):
        x,info = jax.scipy.sparse.linalg.cg(A=S_induc_vp, b=v)
        return x
    
    # @jax.jit
    def composite_vp(v):
        # computes S_Z^{-1} @ S_full
        return S_full_vp(S_induc_inv_vp(v))
    
    # ! option 2: investigate woodbury matrix identity to avoid inverse maybe?
    # todo ...maybe?
    
    # todo ENSURE REUSE CORRECT!
    # ! reuse randomly sampled vectors from stochtrace estimator for logdet estimator
    
    x0 = jnp.ones((D,), dtype=float)
    nsamples = 16
    # keys = jax.random.split(key, num=nsamples)
    keys = jax.random.split(key, num=1)
    sampler = matfree_stochtrace.sampler_normal(x0, num=nsamples)
    
    def stoch_trace(Xfun):
        integrand = matfree_stochtrace.integrand_trace()
        estimator = matfree_stochtrace.estimator(integrand, sampler=sampler)
        # estimator = functools.partial(estimator, Xfun)
        # traces = jax.lax.map(jax.checkpoint(estimator), keys) # ! note this forces recomputation => more comp. expensive!!
        # return traces.mean()
        traces = estimator(Xfun, keys[0])
        return traces
    trace_term = stoch_trace(composite_vp)
    
    # ! use stochastic Lanczos quadrature
    def stoch_lanczos_quadrature(Xfun):
        # adapted directly from https://pnkraemer.github.io/matfree/Tutorials/1_compute_log_determinants_with_stochastic_lanczos_quadrature/
        num_matvecs = min(M, min(D,32))                                         # todo NAN values for large models?
        tridiag_sym = decomp.tridiag_sym(num_matvecs)
        problem = funm.integrand_funm_sym_logdet(tridiag_sym)
        estimator = matfree_stochtrace.estimator(problem, sampler=sampler)
        # estimator = functools.partial(estimator, Xfun)
        # logdets = jax.lax.map(jax.checkpoint(estimator), keys) # ! note this forces recomputation => more comp. expensive!!
        # return logdets.mean()
        logdets = estimator(Xfun, keys[0]) ## todo WHY NAN values???
        return logdets
    logdet_term = stoch_lanczos_quadrature(S_induc_vp)
    pdb.set_trace()
    return trace_term + logdet_term # ? missing beta*D term? (No, because it is implicitly included in the logdet term...)


def alternative_objective(z, x, state, alpha, model_type, key, full_set_size=None):
    # prior_std = alpha**(-0.5) # σ = 1/sqrt(⍺) = ⍺^(-1/2)
    S_full, *_ = compute_curvature_approx_dense(state, x, alpha=alpha, model_type=model_type, full_set_size=full_set_size)
    S_induc,    *_ = compute_curvature_approx_dense(state, z, alpha=alpha, model_type=model_type, full_set_size=full_set_size)
    S_induc_inv = jnp.linalg.inv(S_induc)
    
    """
    =========================================
    Compute KL[ q(theta|Z) || p(theta|data) ]
    =========================================
    """
    trace_term = jnp.linalg.trace(S_full @ S_induc_inv)
    
    # log_det_term = jnp.log( 1 / (jnp.linalg.det(S_full_inv) * jnp.linalg.det(S_induc)) ) # ! problematic, super ill-conditioned?
    sign_full, logdet_full = jnp.linalg.slogdet(S_full)
    sign_induc, logdet_induc_inv = jnp.linalg.slogdet(S_induc_inv)
    # todo use signs to signal if determinants are nonpositive - does not play well with JIT
    logdet_term = - logdet_full - logdet_induc_inv
    
    D = 0 # todo const - does it matter for optimization?
    return 0.5 * (trace_term - D + logdet_term)

# variational_grad = jax.value_and_grad(naive_objective)
# variational_grad = jax.value_and_grad(alternative_objective)
variational_grad = jax.value_and_grad(alternative_objective_scalable)


# @partial(jax.jit, static_argnames=('alpha', 'model_type', 'zoptimizer', 'num_mc_samples', 'full_set_size'))
def optimize_step(z, x, map_model_state, alpha, opt_state, rng, zoptimizer, num_mc_samples, model_type, full_set_size=None):
    # print("Computing loss+grads")
    loss, grads = variational_grad(
        z, 
        x, 
        map_model_state, 
        alpha, 
        key=rng,
        # num_mc_samples=num_mc_samples,
        model_type=model_type, 
        full_set_size=full_set_size
    )
    # print("Finished computing loss+grads")
    updates, new_opt_state = zoptimizer.update(grads, opt_state)
    # print("Finished computing updates")
    new_params = optax.apply_updates(z, updates)
    # print("Finished applying updates")
    return new_params, new_opt_state, loss


def train_inducing_points(map_model_state, zinit, zoptimizer, dataloader, model_type, rng, num_mc_samples, alpha, num_steps, full_set_size, plot_full_dataset_fn_debug=None):
    z = zinit
    opt_state = zoptimizer.init(z)
    _iter = iter(dataloader)
    
    def get_next_sample(num_batches=1):
        nonlocal _iter 
        sample_batches = []
        for _ in range(num_batches):
            try:
                batch = next(_iter)
            except StopIteration:
                _iter = iter(dataloader)
                batch = next(_iter)
            sample_batches.append(batch)
        sample = list(zip(*sample_batches))
        sample = (jnp.concatenate(sample[0], axis=0), jnp.concatenate(sample[1], axis=0))
        return sample
    
    # ! bounding box for constraining inducing point range
    dataset_sample = get_next_sample(num_batches=5)[0]
    lb = dataset_sample.min(axis=0)
    ub = dataset_sample.max(axis=0)
    
    # todo for debugging
    fig, ax = plt.subplots(figsize=(12, 8))
    trajectory = [] 
    
    pbar = tqdm(range(num_steps))
    for step in pbar:
        dataset_sample = get_next_sample(num_batches=1)
        x_sample,y_sample = dataset_sample
        rng, rng_step = jax.random.split(rng)
        z, opt_state, loss = optimize_step(
            z, 
            x_sample, 
            map_model_state=map_model_state, 
            alpha=alpha, 
            opt_state=opt_state, 
            rng=rng_step,
            model_type=model_type,
            zoptimizer=zoptimizer, 
            num_mc_samples=num_mc_samples, 
            full_set_size=full_set_size
        )
        # ! Enforce constraints on x (and w, if necessary)
        # z = jnp.clip(z, lb, ub)
        
        pbar.set_description_str(f"Loss: {loss:.3f}", refresh=True)
        
        # todo for debug: every 10 steps, record & plot
        if step % 2 == 0:
            # convert to NumPy for plotting
            z_np = np.asarray(z)
            trajectory.append(z_np)

            traj = np.stack(trajectory)    # shape (n_points, 2)
            ax.clear()
            ax.plot(traj[:, :, 0], traj[:,:, 1], '-o', color="black", markersize=2, zorder=7)
            plot_full_dataset_fn_debug()
            ax.set_xlim(lb[0] - 1.0, ub[0] + 1.0)
            ax.set_ylim(lb[1] - 1.0, ub[1] + 1.0)
            ax.set_xlabel('z[0]')
            ax.set_ylabel('z[1]')
            ax.set_title(f'Latent Trajectory after {step} steps')
            scatterp(*z_np.T, color="yellow", zorder=8, marker="X", label="Inducing points")
            
            # force a draw
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.savefig("test.png")
        
    
    return z
