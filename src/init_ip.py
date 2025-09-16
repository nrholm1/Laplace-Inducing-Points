from functools import partial
import pdb
import jax
import jax.numpy as jnp

from matfree import stochtrace as matfree_stochtrace
from matplotlib import pyplot as plt

from src.data import make_iter
from src.slq import estimate_logdet_slq
from src.ggn import build_WTW, compute_W_vps, compute_ggn_vp
from src.nplot import scatterp
from src.utils import flatten_nn_params

def get_initial_points(
        Z,
        train_loader_ip, 
        m_ip,

        state,
        key,
        model_type,
        alpha,
        full_set_size=None, 
        st_samples=256, 
        slq_samples=2, 
        slq_num_matvecs=None,
    ):
    flat_params, unravel_fn = flatten_nn_params(state.params)
    
    _iter = make_iter(train_loader_ip)
    
    def get_next_sample(num_batches=1):
        nonlocal _iter 
        sample_batches = []
        for _ in range(num_batches):
            try:
                batch = next(_iter)
            except StopIteration:
                _iter = make_iter(train_loader_ip)
                batch = next(_iter)
            sample_batches.append(batch)
        sample = list(zip(*sample_batches))
        sample = (jnp.concatenate(sample[0], axis=0), jnp.concatenate(sample[1], axis=0))
        return sample
    
    scatterp(*Z.T, color="yellow", zorder=8, marker="X", s=100)
    
    num_steps = m_ip - int(Z.shape[0]) # todo - this is the initial method.
    for i in range(num_steps):
        N = full_set_size
        M = Z.shape[0]
        beta = N / M
        alpha_inv = 1.0 / alpha
        beta_inv = 1.0 / beta

        D = sum(x.size for x in jax.tree_util.tree_leaves(unravel_fn(flat_params)))
        if model_type == 'regressor': D -= 1

        Z_const = jax.lax.stop_gradient(Z)
        G_to_n = compute_ggn_vp(state, Z_const, model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn,
                                            full_set_size=None)
        W, WT    = compute_W_vps(state, Z_const, model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn,
                                            full_set_size=None)
        def G_bar(v):
            return G_to_n(v) / M
        

        x0          = jnp.zeros((D,), dtype=jnp.float32)
        dummy       = WT(x0)
        inner_shape = dummy.shape
        d_z         = dummy.size
        I_d         = jnp.eye(d_z, dtype=jnp.float32)
        WTW         = build_WTW(W, WT, inner_shape, d_z, dtype=jnp.float32, block=min(M,32))

        def ggn_ip_inv(v):
            # Woodbury inversion
            u = WT(v).reshape(d_z)
            x = jax.scipy.linalg.solve(beta_inv * I_d + alpha_inv * WTW, u)
            return alpha_inv * v - alpha_inv**2 * W(x.reshape(inner_shape))
        
        
        # todo map below over candidate z's
        key, key_step = jax.random.split(key)
        key_trace, key_slq = jax.random.split(key_step)
        x0 = jnp.zeros((D,), dtype=jnp.float32)

        trace_integrand = matfree_stochtrace.integrand_trace()
        trace_sampler   = matfree_stochtrace.sampler_rademacher(x0, num=st_samples)
        slq_num_matvecs = min(slq_num_matvecs, M)
        
        @jax.jit
        def score_candidate(z_single):
            z_single = jax.lax.stop_gradient(z_single[None,...])
            
            G_new  = compute_ggn_vp(state, z_single, model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn,
                                                full_set_size=None)
            
            def trace_inner(v):
                return G_new(ggn_ip_inv(v))
            trace_estimator = partial(matfree_stochtrace.estimator(trace_integrand, sampler=trace_sampler), trace_inner)
            trace_term      = jax.checkpoint(trace_estimator)(key_trace)

            def logdet_inner(v):
                w = ggn_ip_inv(v)
                return v + N / (M+1) * (G_new(w) - G_bar(w))
            logdet_term     = estimate_logdet_slq(logdet_inner, D=D, M=M, key=key_slq,
                                                slq_samples=slq_samples, slq_num_matvecs=slq_num_matvecs)
            
            return trace_term - logdet_term
        
        X_pool, _ = get_next_sample(num_batches=2) # todo hyperparam
        
        best_val = -jnp.inf
        best_idx = 0
        for j in range(X_pool.shape[0]):
            val = score_candidate(X_pool[j])
            best_idx = jnp.where(val > best_val, j, best_idx)
            best_val = jnp.maximum(best_val, val)
        
        best_z = X_pool[best_idx][None, ...]
        Z = jnp.concatenate([Z, best_z], axis=0)
        
        print(f"Step {i+1}/{num_steps}: best score={float(best_val):.3f}")
        scatterp(*best_z.T, color="yellow", zorder=8, marker="X", s=100)
        plt.savefig(f"fig/toy/ips.png")
        
    return Z
        
        # todo find/implement convergence check (too little change over previous Z)