from functools import partial
from typing import Iterable, Optional
import jax
import jax.numpy as jnp

from src.data import make_iter
from src.train_map import log_joint
from src.lla import compute_curvature_approx
from src.utils import count_model_params, flatten_nn_params
from src.slq import estimate_logdet_slq


@partial(jax.jit, static_argnames=('model_type', 'slq_samples', 'slq_num_matvecs', 'full_set_size', 'unravel_fn'))
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
        full_set_size = None,
        flat_params, 
        unravel_fn,
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
    D = count_model_params(map_state.params)
    if model_type == "regressor":
        D -= 1
    M = int(Z.shape[0])
    def logdet_term(alpha):
        Sz_vp = compute_curvature_approx(
            map_state, Z, alpha=alpha, model_type=model_type, full_set_size=full_set_size, flat_params=flat_params, unravel_fn=unravel_fn,
        )
        return estimate_logdet_slq(
            Sz_vp,
            D=D,
            M=M,
            key=key,
            slq_samples=slq_samples,
            slq_num_matvecs=slq_num_matvecs,
        )

    def loss_and_aux(params, batch_stats):
        log_alpha = params['log_alpha']
        alpha = jnp.exp(log_alpha)
        (neg_log_post, new_bs) = log_joint_term(alpha, batch_stats)
        prior_normalizer = -0.5 * D * log_alpha
        logdet = logdet_term(alpha)
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
    get_batch_fn,
    *,
    model_type: str,
    num_steps: int,
    rng=None,
    slq_samples: int = 1,
    slq_num_matvecs: int = 32,
    full_set_size = None
):
    """
    Optimizes alpha for `num_steps` passes over `train_loader`.

    Returns:
      alpha_state, map_state
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    flat_params, unravel_fn = flatten_nn_params(map_state.params)

    for _ in range(num_steps):
        batch = get_batch_fn()
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
            full_set_size=full_set_size,
            flat_params=flat_params, 
            unravel_fn=unravel_fn,
        )

    return log_alpha_state, map_state