import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from lla import posterior_lla, compute_curvature_approx
from train_map import nl_likelihood_fun


def sample_params(mu, cov, rng):
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
        log_p_data = -nl_likelihood_fun(apply_fn, theta_sample, batch) # ! nll, therefore negate it...
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


def naive_objective(params, dataset, state, alpha, rng, num_mc_samples, full_set_size=None, reg_coeff=0):
    # Unpack the parameters: inducing points x and weights w
    x, w = params

    q, unravel_fn = posterior_lla(
        state,
        prior_std=alpha,# todo correct alpha used here?
        x=x,
        w=w,
        y=None,  # ? explicitly pass no labels
        full_set_size=full_set_size,
        return_unravel_fn=True
    )

    loglik_term = var_loglik_fun(q, dataset, state.apply_fn, unravel_fn, rng, num_mc_samples=num_mc_samples)
    kl_term = var_kl_fun(q, alpha)
    reg_term = 0 # ! reg_coeff * (jnp.sum(jnp.square(x)) + jnp.sum(jnp.square(w)))

    return - (loglik_term - kl_term) + reg_term


def alternative_objective(params, dataset, state, alpha, full_set_size=None):
    # Unpack the parameters: inducing points x and weights w
    xind, w = params
    # ! compute synthetic labels for inducing points
    yind,logvar = jax.lax.stop_gradient(state.apply_fn(state.params, xind))
    
    prior_std = alpha**(-0.5) # σ = 1/sqrt(⍺) = ⍺^(-1/2)
    S_full_inv, *_ = compute_curvature_approx(state, dataset, prior_std=prior_std, w=jnp.ones_like(dataset[0]), full_set_size=full_set_size, return_Hinv=True)
    S_induc,    *_ = compute_curvature_approx(state, (xind, yind), prior_std=prior_std, w=w, full_set_size=full_set_size, return_Hinv=False)
    
    """
    ========================
    Compute KL[ q || qfull ]
    ========================
    """
    trace_term = jnp.linalg.trace(S_full_inv @ S_induc)
    
    # log_det_term = jnp.log( 1 / (jnp.linalg.det(S_full_inv) * jnp.linalg.det(S_induc)) ) # ! problematic, super ill-conditioned?
    sign_full, logdet_full = jnp.linalg.slogdet(S_full_inv)
    sign_induc, logdet_induc = jnp.linalg.slogdet(S_induc)
    # todo use signs to signal if determinants are nonpositive - does not play well with JIT
    log_det_term = - logdet_full - logdet_induc
    
    D = 0 # todo const - does it matter for optimization?
    return 0.5 * (trace_term - D + log_det_term)

variational_grad = jax.value_and_grad(alternative_objective)


def optimize_step(params, dataset, map_model_state, alpha, opt_state, rng, xoptimizer, num_mc_samples, full_set_size=None):
    loss, grads = variational_grad(
        params, 
        dataset, 
        map_model_state, 
        alpha, 
        # rng,
        # num_mc_samples=num_mc_samples,
        full_set_size=full_set_size
    )
    updates, new_opt_state = xoptimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


# ? JIT optimize_step here, since the static_argnames is problematic in the decorator?
# print("WARNING! optimize_step not JITted for debug")
optimize_step = jax.jit(optimize_step, static_argnames=['xoptimizer', 'num_mc_samples', 'full_set_size']) # ! not JITted for debug


def train_inducing_points(map_model_state, xinit, winit, xoptimizer, dataloader, rng, num_mc_samples, alpha, num_steps):
    logvar = map_model_state.params['params']['logvar']
    # todo delete map_model_state.params['params']['logvar'] from params.
    
    params = (xinit, winit)
    opt_state = xoptimizer.init(params)
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
    
    dataset_sample = get_next_sample(num_batches=5)
    # todo better strategy for enforcing constraints?
    lb, ub = dataset_sample[0].min(), dataset_sample[0].max()  # ! data support range (1D case)
    N = len(dataloader) * len(next(iter(dataloader))[0])
    
    pbar = tqdm(range(num_steps))
    for step in pbar:
        dataset_sample = get_next_sample(num_batches=1)
        rng, rng_step = jax.random.split(rng)
        params, opt_state, loss = optimize_step(
            params, 
            dataset_sample, 
            map_model_state, 
            alpha, 
            opt_state, 
            rng_step,
            xoptimizer=xoptimizer, 
            num_mc_samples=num_mc_samples, 
            full_set_size=N
        )
        # Unpack parameters:
        x, w = params
        # ! Enforce constraints on x (and w, if necessary)
        x = jnp.clip(x, lb, ub)
        params = (x, w)
        
        if step == 0:
            print(f"Initial loss: {loss:.3f}")
        pbar.set_description_str(f"Loss: {loss:.3f}", refresh=True)
    
    return params
