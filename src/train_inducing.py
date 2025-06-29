from functools import partial
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

from src.stochtrace import hutchpp, hutchpp_v2
from src.lla import posterior_lla_dense, compute_curvature_approx_dense, compute_curvature_approx
from src.ggn import build_WTWz, compute_W_vps, build_WTW
# from src.train_map import nl_likelihood_fun_regression
from src.utils import count_model_params
from src.toydata import plot_binary_classification_data
from src.data import make_iter
from src.nplot import plot_color, scatterp, plot_grayscale



def alternative_objective_scalable_exact(Z, X, state, alpha, model_type, key, full_set_size=None,
                                   st_samples=256, slq_samples=2, slq_num_matvecs=None):
    """ MATRIX FREE
    =========================================
    Compute KL[ q(theta|Z) || q(theta|data) ]
    =========================================
    """
    N = full_set_size
    M = Z.shape[0]
    K = X.shape[0]
    beta = N / M
    gamma = N / K
    alpha_inv = 1.0 / alpha
    beta_inv = 1.0 / beta
    
    D = count_model_params(state.params['params'])
    if model_type == 'regressor':
        D -= 1 # ! subtract logvar parameter!
    
    # compute matrix free linear operator oracles
    S_vp  = compute_curvature_approx(
        state, X, alpha=alpha, model_type=model_type, 
        full_set_size=N)
    Sz_vp = compute_curvature_approx(
        state, Z, alpha=alpha, model_type=model_type, 
        full_set_size=N)
    Wz, WzT = compute_W_vps(
        state, Z, model_type=model_type, 
        full_set_size=None)
    W, WT = compute_W_vps(
        state, X, model_type=model_type, 
        full_set_size=None)
    
    
    
    # pdb.set_trace()
    dummy = WzT(jnp.zeros(D))
    inner_shape = dummy.shape
    d_z           = dummy.size
    I_d_z         = jnp.eye(d_z, dtype=float)
    WzTWz = build_WTW(Wz, WzT, inner_shape, d_z, dtype=float, block=1) # ! build dense WTW in blocks to lower memory pressure
    
    _,logdet_WTW = jnp.linalg.slogdet(I_d_z + beta*alpha_inv*WzTWz)
    logdet_term = logdet_WTW + D*jnp.log(alpha) # ! drop last term since it does not matter for optimization

    dummy = WT(jnp.zeros(D))
    d           = dummy.size
    WTWz = build_WTWz(WT, Wz, inner_shape, d=d, dtype=float, block=1)
    
    M  = beta_inv*I_d_z + alpha_inv*WzTWz
    L  = jnp.linalg.cholesky(M)
    S1 = jax.scipy.linalg.cho_solve((L, True), WzTWz)
    S2 = jax.scipy.linalg.cho_solve((L, True), WTWz.T)
    
    trace1 = jnp.linalg.trace(S1)
    trace2 = jnp.vdot(WTWz, S2.T)
    trace_term = - alpha_inv*trace1 - gamma*alpha_inv**2*trace2
    
    return logdet_term + trace_term


def alternative_objective_scalable(Z, X, state, alpha, model_type, key, full_set_size=None,
                                   st_samples=256, slq_samples=2, slq_num_matvecs=None):
    """ MATRIX FREE
    =========================================
    Compute KL[ q(theta|Z) || q(theta|data) ]
    =========================================
    """
    N = full_set_size
    M = Z.shape[0]
    K = X.shape[0]
    beta = N / M
    gamma = N / K
    alpha_inv = 1.0 / alpha
    beta_inv = 1.0 / beta
    
    # D = count_model_params(state.params['params'])
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
    Wz, WzT = compute_W_vps(
        state, Z, model_type=model_type, 
        full_set_size=None)
    # W, WT = compute_W_vps(
    #     state, X, model_type=model_type, 
    #     full_set_size=None)
    
    
    dummy = WzT(jnp.zeros(D))
    inner_shape = dummy.shape
    d_z           = dummy.size
    I_d_z         = jnp.eye(d_z, dtype=float)
    WzTWz = build_WTW(Wz, WzT, inner_shape, d_z, dtype=float, block=1) # ! build dense WTW in blocks to lower memory pressure
    def Sz_inv_vp_woodbury_dense(v):
        u = WzT(v).reshape(d_z)
        x   = jax.scipy.linalg.solve(
            beta_inv*I_d_z + alpha_inv*WzTWz,
            u)
        return alpha_inv*v - alpha_inv**2*Wz(x.reshape(inner_shape))
    
    def composite_vp(v):
        return S_vp(Sz_inv_vp_woodbury_dense(v))
    
    # # ! use same random vectors for StochTrace and SLQ
    x0 = jnp.ones((D,), dtype=float)
    sampler = matfree_stochtrace.sampler_rademacher(x0, num=st_samples)
    probes = sampler(key)
    st_sampler =  lambda _: probes
    slq_sampler = lambda _: probes[:slq_samples]
    
    stoch_trace = lambda vp: hutchpp_v2(vp, st_sampler, s1=st_samples-16, s2=16)
    trace_term = stoch_trace(composite_vp)
    
    # ! use stochastic Lanczos quadrature
    slq_num_matvecs = slq_num_matvecs if slq_num_matvecs is not None else int(M*0.8)
    def slq_logdet(Xfun):
        # Adapted directly from https://pnkraemer.github.io/matfree/Tutorials/1_compute_log_determinants_with_stochastic_lanczos_quadrature/
        # Old tridiagonal formulation:
        # tridiag_sym = decomp.tridiag_sym(slq_num_matvecs)
        # problem = integrand_funm_sym_logdet(tridiag_sym)
        
        # New bidiagonal reformulation:
        bidiag_sym = decomp.bidiag(slq_num_matvecs)
        problem = funm.integrand_funm_product_logdet(bidiag_sym)
        
        estimator = matfree_stochtrace.estimator(problem, sampler=slq_sampler)
        estimate = partial(estimator, Xfun)
        keys = jax.random.split(key, slq_samples)
        logdets = jax.lax.map(jax.checkpoint(estimate),keys)
        return logdets.mean()
    # logdet_term = slq_logdet(Sz_vp)
                          
    sqrt_alpha = jnp.sqrt(alpha)
    def bidiag_target(v):
        x, unravel_fn = jax.flatten_util.ravel_pytree(WzT(v))
        return jnp.concatenate([sqrt_alpha * v, x])

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
    # updates, new_opt_state = zoptimizer.update(grads, opt_state)  # ? ADAM, SGD, etc.
    updates, new_opt_state = zoptimizer.update(grads, opt_state, Z) # ? ADAMW
    new_params = optax.apply_updates(Z, updates)
    return new_params, new_opt_state, loss


def train_inducing_points(map_model_state, zinit, zoptimizer, dataloader, model_type, rng, num_mc_samples, alpha, num_steps, full_set_size, scalable, plot_type=None,
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
    
    if plot_type in ['spiral', 'xor', 'banana']:
        fig, ax = plt.subplots(figsize=(12, 8))
        trajectory = [] 
        dataset_sample = get_next_sample(num_batches=32)[0]
        lb = dataset_sample.min(axis=0)
        ub = dataset_sample.max(axis=0)
        del dataset_sample
    
    pbar = tqdm(range(num_steps))
    for step in pbar:
        dataset_sample = get_next_sample(num_batches=1)
        x_sample,y_sample = dataset_sample
        
        # ! Common Random Numbers - does it work???
        # if step % 4 == 0:
        #     rng = jax.random.fold_in(rng, step) # ? TEST holding probes constant
        
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
        
        pbar.set_description_str(f"Loss: {loss:.3f}", refresh=True)
        
        # todo for debug: every 2 steps, record & plot
        if (plot_type is not None) and (step % 4 == 0):
            z_np = np.asarray(z)
            
            if plot_type in ['mnist', 'fmnist']:
                plot_grayscale(z_np[:32].squeeze(), step, name=plot_type)
            
            elif plot_type in ['cifar10']:
                plot_color(z_np[:32].squeeze(), step, name=plot_type)
                
            elif plot_type in ['spiral', 'xor', 'banana']:
                trajectory.append(z_np)

                traj = np.stack(trajectory)    # shape (n_points, 2)
                ax.clear()
                ax.plot(traj[:, :, 0], traj[:,:, 1], '-o', color="black", markersize=2, zorder=7)
                ax.set_xlim(lb[0] - 1.0, ub[0] + 1.0)
                ax.set_ylim(lb[1] - 1.0, ub[1] + 1.0)
                ax.set_xlabel('z[0]')
                ax.set_ylabel('z[1]')
                ax.set_title(f'Inducing Point Trajectory after {step} steps')
                scatterp(*z_np.T, color="yellow", zorder=8, marker="X", label="Inducing points")
                plot_binary_classification_data(dataset_sample[0], dataset_sample[1].squeeze())
                
                # force a draw
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.savefig(f"fig/toy/ips.png")
                
                trajectory = trajectory[-3:]
        
    
    return z
