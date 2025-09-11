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

from matfree import stochtrace as matfree_stochtrace

from src.scalemodels import TrainState
from src.train_alpha import train_alpha
from src.lla import compute_curvature_approx_dense, compute_curvature_approx, predict_lla_scalable
from src.ggn import compute_W_vps, build_WTW
from src.utils import count_model_params
from src.toydata import plot_binary_classification_data
from src.data import make_iter
from src.nplot import plot_color, scatterp, plot_grayscale, plot_lla_2D_classification_single
from src.slq import estimate_logdet_slq


def ip_objective_mf(Z, X, state, alpha, model_type, key, full_set_size=None,
                                   st_samples=256, slq_samples=2, slq_num_matvecs=None):
    """ MATRIX FREE
    =========================================
    Compute KL[ q(theta|Z) || q(theta|data) ]
    =========================================
    """
    N = full_set_size
    M = Z.shape[0]
    beta = N / M
    alpha_inv = 1.0 / alpha
    beta_inv = 1.0 / beta
    
    D = count_model_params(state.params)
    if model_type == 'regressor':
        D -= 1 # subtract logvar parameter!
    
    ggn_full  = compute_curvature_approx(
        state, X, alpha=alpha, model_type=model_type, 
        full_set_size=N)
    ggn_ip  = compute_curvature_approx(
        state, Z, alpha=alpha, model_type=model_type, 
        full_set_size=N)
    W, WT = compute_W_vps(
        state, Z, model_type=model_type, 
        full_set_size=None)
    
    dummy = WT(jnp.zeros(D))
    inner_shape = dummy.shape
    d_z           = dummy.size
    I_d_z         = jnp.eye(d_z, dtype=float)
    WTW = build_WTW(W, WT, inner_shape, d_z, dtype=float, block=64) # ! build dense WTW in blocks to lower memory pressure

    def ggn_ip_inv(v):
        # Woodbury inversion
        u = WT(v).reshape(d_z)
        x = jax.scipy.linalg.solve(beta_inv * I_d_z + alpha_inv * WTW, u)
        return alpha_inv * v - alpha_inv**2 * W(x.reshape(inner_shape))
    
    def composite_vp(v):
        return ggn_full(ggn_ip_inv(v))

    key_trace, key_slq = jax.random.split(key, 2)
    x0 = jnp.zeros((D,), dtype=float)
    
    # Hutchinson
    trace_integrand = matfree_stochtrace.integrand_trace()
    trace_sampler = matfree_stochtrace.sampler_rademacher(x0, num=st_samples)
    trace_estimator = partial(
        matfree_stochtrace.estimator(trace_integrand, sampler=trace_sampler),
        composite_vp
    )
    trace_term = jax.checkpoint(trace_estimator)(key_trace)

    # SLQ
    slq_num_matvecs = min(slq_num_matvecs, M)
    logdet_term = estimate_logdet_slq(
        ggn_ip,
        D=D,
        M=M,
        key=key_slq,
        slq_samples=slq_samples,
        slq_num_matvecs=slq_num_matvecs,
    )
    
    return trace_term + logdet_term


def ip_objective_dense(Z, X, state, alpha, model_type, key, full_set_size=None):
    """
    =========================================
    Compute KL[ q(theta|Z) || p(theta|data) ]
    =========================================
    """
    S, *_ = compute_curvature_approx_dense(state, X, alpha=alpha, model_type=model_type, full_set_size=full_set_size)
    S_z,    *_ = compute_curvature_approx_dense(state, Z, alpha=alpha, model_type=model_type, full_set_size=full_set_size)
    S_z_inv = jnp.linalg.inv(S_z)
    
    trace_term = jnp.linalg.trace(S @ S_z_inv)
    
    _, S_logdet = jnp.linalg.slogdet(S)
    _, S_z_inv_logdet = jnp.linalg.slogdet(S_z_inv)
    logdet_term = - S_logdet - S_z_inv_logdet
    
    return logdet_term + trace_term


variational_grad_dense = jax.value_and_grad(ip_objective_dense)
variational_grad_scalable = jax.value_and_grad(ip_objective_mf)


@partial(jax.jit, static_argnames=('model_type', 'zoptimizer', 'num_mc_samples', 'full_set_size', 'scalable', 'st_samples', 'slq_samples', 'slq_num_matvecs'))
def optimize_step(Z, X, map_model_state, alpha, opt_state, rng, zoptimizer, num_mc_samples, model_type, full_set_size=None, scalable=True,
                  st_samples=256, slq_samples=2, slq_num_matvecs=None):
    if scalable:
        rng = jax.random.fold_in(rng, 2)
        grad_fun = variational_grad_scalable
        loss, grads = grad_fun(
            Z, 
            X, 
            map_model_state, 
            alpha, 
            key=rng,
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
            model_type=model_type, 
            full_set_size=full_set_size,
        )
    updates, new_opt_state = zoptimizer.update(grads, opt_state, Z) # ? ADAMW
    new_params = optax.apply_updates(Z, updates)
    return new_params, new_opt_state, loss


def train_inducing_points(map_state, zinit, zoptimizer, dataloader, model_type, rng, num_mc_samples, alpha, num_steps, full_set_size, scalable, plot_type=None,
                          st_samples=256, slq_samples=2, slq_num_matvecs=None):
    Z = zinit
    opt_state = zoptimizer.init(Z)
    
    # make state for optimizing alpha
    alpha_tx = optax.adam(learning_rate=1e-2)
    log_alpha_state = TrainState.create(
        apply_fn=lambda p: p, 
        params={'log_alpha': jnp.log(alpha)},
        tx=alpha_tx,
    )
    
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
        fig, ax = plt.subplots(figsize=(10, 8))
        trajectory = [] 
        dataset_sample = get_next_sample(num_batches=32)[0]
        lb = dataset_sample.min(axis=0)
        ub = dataset_sample.max(axis=0)
        del dataset_sample
    
    pbar = tqdm(range(num_steps))
    for step in pbar:
        dataset_sample = get_next_sample(num_batches=1)
        x_sample,y_sample = dataset_sample
        
        rng = jax.random.fold_in(rng, step)
        
        Z, opt_state, loss = optimize_step(
            Z, 
            x_sample,
            map_model_state=map_state, 
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
        
        # Jointly optimize alpha by interleaving steps.
        # After a burnin period, every x'th step, optimize alpha for y steps.
        alpha_steps_every = 5
        alpha_steps_per_call = 5
        if (step % alpha_steps_every == 0) and step > 20:
            rng, alpha_rng = jax.random.split(rng)
            log_alpha_state, map_state = train_alpha(
                map_state=map_state,
                log_alpha_state=log_alpha_state,
                Z=Z,
                train_loader=dataloader,
                test_loader=None,          # could pass a test loader
                model_type=model_type,
                num_steps=alpha_steps_per_call,
                rng=alpha_rng,
                slq_samples=slq_samples, 
                slq_num_matvecs=slq_num_matvecs,
                full_set_size=full_set_size
            )
            alpha = jnp.exp(log_alpha_state.params['log_alpha']).item()
        
        pbar.set_description_str(f"⍺: {alpha:.3e} |  Loss: {loss:.3f}", refresh=True)
        
        if (plot_type is not None) and (step % 6 == 0):
            z_np = np.asarray(Z)
            
            if plot_type in ['mnist', 'fmnist']:
                plot_grayscale(z_np[:32].squeeze(), step, name=plot_type)
            
            elif plot_type in ['cifar10']:
                plot_color(z_np[:32].squeeze(), step, name=plot_type)
                
            elif plot_type in ['spiral', 'xor', 'banana']:
                trajectory.append(z_np)
                traj = np.stack(trajectory)
                ax.clear()
                ax.plot(traj[:, :, 0], traj[:,:, 1], '-o', color="black", markersize=2, zorder=7)
                ax.set_xlim(lb[0] - 1.0, ub[0] + 1.0)
                ax.set_ylim(lb[1] - 1.0, ub[1] + 1.0)
                ax.set_xlabel('z[0]')
                ax.set_ylabel('z[1]')
                ax.set_title(f'Inducing Point Trajectory after {step} steps')
                scatterp(*z_np.T, color="yellow", zorder=8, marker="X", label="Inducing points")

                # expensive backdrop
                plot_lla_2D_classification_single(fig, ax, map_state,dataset_sample[0], dataset_sample[1].squeeze(), z_np, alpha, matrix_free=True, num_mc_samples=32, mode='ip_lla', key=rng, plot_Z=True, cbar=False)
                
                plot_binary_classification_data(dataset_sample[0], dataset_sample[1].squeeze())
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.savefig(f"fig/toy/ips.png")
                
                trajectory = trajectory[-3:]
        
    
    return Z
