from functools import partial
import pdb
from typing import Iterable, Optional
import jax
import jax.numpy as jnp
import optax

from matfree import decomp, funm, stochtrace as matfree_stochtrace
from tqdm import tqdm

from src.data import make_iter
from src.train_map import log_joint
from src.lla import compute_curvature_approx
from src.ggn import compute_W_vps
from src.utils import count_model_params


@partial(jax.jit, static_argnames=('model_type', 'slq_samples', 'slq_num_matvecs'))
def optimize_alpha_step(
        *,
        log_alpha_state,
        Z,
        map_state,
        batch,
        model_type: str,
        key,
        slq_samples: int = 1,
        slq_num_matvecs: int = 32,
    ):
    # log joint given data batch
    def log_joint_term(alpha, batch_stats):
        return log_joint(map_state.params,
                         batch_stats,
                         map_state,
                         batch, 
                         alpha, 
                        model_type)
    
    # log determinant
    D = count_model_params(map_state.params) # precompute
    def logdet_term(alpha):
        nonlocal slq_num_matvecs, D
        W, WT = compute_W_vps(
            map_state, Z, model_type=model_type, 
            full_set_size=None)
        
        M = Z.shape[0]
        if model_type == 'regressor':
            D -= 1 # subtract logvar parameter!
        x0 = jnp.ones((D,), dtype=float)
        sampler = matfree_stochtrace.sampler_rademacher(x0, num=slq_samples)
        
        slq_num_matvecs = min(slq_num_matvecs, M)
        def slq_logdet(Xfun):
            # Adapted from https://pnkraemer.github.io/matfree/Tutorials/1_compute_log_determinants_with_stochastic_lanczos_quadrature/
            # BUT using bidiagonal reformulation. See paper/thesis for details.
            bidiag_sym = decomp.bidiag(slq_num_matvecs)
            problem = funm.integrand_funm_product_logdet(bidiag_sym)
            
            estimator = matfree_stochtrace.estimator(problem, sampler=sampler)
            estimate = partial(estimator, Xfun)
            keys = jax.random.split(key, slq_samples)
            logdets = jax.lax.map(jax.checkpoint(estimate),keys)
            return logdets.mean()
                            
        sqrt_alpha = jnp.sqrt(alpha)
        def bidiag_target(v):
            x, unravel_fn = jax.flatten_util.ravel_pytree(WT(v))
            return jnp.concatenate([sqrt_alpha * v, x])

        return slq_logdet(bidiag_target)
    
    def loss_and_aux(params, batch_stats):
        log_alpha = params['log_alpha']
        alpha = jnp.exp(log_alpha)
        (neg_log_post, new_bs) = log_joint_term(alpha, batch_stats)
        prior_normalizer = -0.5 * D * log_alpha
        logdet = logdet_term(alpha)
        # pdb.set_trace()
        return neg_log_post + prior_normalizer + .5 * logdet, new_bs
    
    (loss, new_bs), grad_alpha = jax.value_and_grad(
            loss_and_aux, argnums=0, has_aux=True
        )(log_alpha_state.params, map_state.batch_stats)

    new_log_alpha_state = log_alpha_state.apply_gradients(grads=grad_alpha)
    new_map_state = map_state.replace(batch_stats=new_bs)
    return new_log_alpha_state, new_map_state, loss
    
    


def train_alpha(
    map_state,
    log_alpha_state,
    Z,
    train_loader: Iterable,
    test_loader: Optional[Iterable] = None,  # currently unused
    *,
    model_type: str,
    num_steps: int,
    rng=None,
    slq_samples: int = 1,
    slq_num_matvecs: int = 32,
):
    """
    Optimizes alpha for `num_steps` passes over `train_loader`.

    Returns:
      alpha_state, map_state
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # pbar = tqdm(range(num_steps), ncols=80)
    # for _ in pbar:
    for _ in range(num_steps):
        for batch in make_iter(train_loader):
            rng, subkey = jax.random.split(rng)
            log_alpha_state, map_state, loss = optimize_alpha_step(
                log_alpha_state=log_alpha_state,
                Z=Z,
                map_state=map_state,
                batch=batch,
                model_type=model_type,
                key=subkey,
                slq_samples=slq_samples,
                slq_num_matvecs=slq_num_matvecs,
            )

        # pbar.set_description(f"α: {jnp.exp(log_alpha_state.params['log_alpha']):.7f}  loss: {float(loss):.3f}")
        # pbar.refresh()

    return log_alpha_state, map_state