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
    
    # scatterp(*Z.T, color="yellow", zorder=8, marker="X", s=100) # ! for debug
    
    D = sum(x.size for x in jax.tree_util.tree_leaves(unravel_fn(flat_params)))
    if model_type == "regressor":
        D -= 1
    
    x0 = jnp.zeros((D,), dtype=jnp.float32)
    trace_integrand = matfree_stochtrace.integrand_trace()
    trace_sampler   = matfree_stochtrace.sampler_rademacher(x0, num=st_samples)
    trace_estimator = matfree_stochtrace.estimator(trace_integrand, sampler=trace_sampler)
    
    @partial(
        jax.jit,
        static_argnames=("model_type", "slq_samples", "D", "N"),
    )
    def score_candidates_batch(
        X_pool,
        Z_const,
        state,
        flat_params,
        model_type: str,
        N: int,
        alpha: float,
        slq_samples: int,
        key_trace,
        key_slq,
        D: int,
    ):
        M = Z_const.shape[0]
        n_next = M + 1
        beta = N / M
        alpha_inv = 1.0 / alpha
        beta_inv = 1.0 / beta

        G_to_n = compute_ggn_vp(
            state,
            Z_const,
            model_type=model_type,
            flat_params=flat_params,
            unravel_fn=unravel_fn,
            full_set_size=None,
        )

        def G_bar(v):
            return G_to_n(v) / M

        W, WT = compute_W_vps(
            state,
            Z_const,
            model_type=model_type,
            flat_params=flat_params,
            unravel_fn=unravel_fn,
            full_set_size=None,
        )

        x0 = jnp.zeros((D,), dtype=jnp.float32)
        inner = WT(x0)
        inner_shape = inner.shape
        d_z = inner.size
        I_d = jnp.eye(d_z, dtype=jnp.float32)
        WTW = build_WTW(W, WT, inner_shape, d_z, dtype=jnp.float32, block=min(M, 32))

        def ggn_ip_inv(v):
            u = WT(v).reshape(d_z)
            x = jax.scipy.linalg.solve(beta_inv * I_d + alpha_inv * WTW, u, assume_a="pos")
            return alpha_inv * v - (alpha_inv**2) * W(x.reshape(inner_shape))

        slq_samples = min(slq_samples, M)

        def score_one(z_single):
            z_single = jax.lax.stop_gradient(z_single[None, ...])

            G_new = compute_ggn_vp(
                state,
                z_single,
                model_type=model_type,
                flat_params=flat_params,
                unravel_fn=unravel_fn,
                full_set_size=None,
            )

            def trace_inner(v):
                return G_new(ggn_ip_inv(v))

            trace_term = trace_estimator(trace_inner, key_trace)

            def logdet_inner(v):
                w = ggn_ip_inv(v)
                return v + (N / n_next) * (G_new(w) - G_bar(w))

            logdet_term = estimate_logdet_slq(
                logdet_inner,
                D=D,
                M=M,
                key=key_slq,
                slq_samples=slq_samples,
                slq_num_matvecs=slq_num_matvecs,
            )

            return trace_term - logdet_term

        scores = jax.vmap(score_one, in_axes=0)(X_pool)
        return scores
    
    
    num_steps = m_ip - int(Z.shape[0]) # todo - this is the initial method.
    for i in range(num_steps):
        X_pool, _ = get_next_sample(num_batches=1) # todo num_batches is actually a hyperparam of how extensive the candidate search should be.
        
        key, key_step = jax.random.split(key)
        key_trace, key_slq = jax.random.split(key_step)
        
        Z_const = jax.lax.stop_gradient(Z)
        
        scores = score_candidates_batch(
            X_pool=X_pool,
            Z_const=Z_const,
            state=state,
            flat_params=flat_params,
            model_type=model_type,
            N=int(full_set_size),
            alpha=float(alpha),
            slq_samples=int(slq_samples),
            key_trace=key_trace,
            key_slq=key_slq,
            D=int(D),
        )

        best_idx = int(jnp.argmax(scores))
        best_z = X_pool[best_idx][None, ...]
        best_val = scores[best_idx]
        Z = jnp.concatenate([Z, best_z], axis=0)
        
        print(f"Step {i+1}/{num_steps}: best score={float(best_val):.3f}")
        # scatterp(*best_z.T, color="yellow", zorder=8, marker="X", s=100)
        # plt.savefig(f"fig/toy/ips.png")
        
    return Z
        
        # todo find/implement convergence check (too little change over previous Z)