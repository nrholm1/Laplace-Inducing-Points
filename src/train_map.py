from functools import partial
import functools
from typing import Callable, Iterable, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.linen import softmax
from tqdm import tqdm
from flax.core.frozen_dict import unfreeze
from flax.traverse_util   import flatten_dict

from src.data import make_iter

# --------------------------------------------------------------------------- #
#  Helper: L2 negative log-prior  (½ α ‖θ‖²)                                  #
# --------------------------------------------------------------------------- #

def _l2_tree(params, weight_precision: float, bias_precision: float = 0.0) -> jnp.ndarray:
    """
    Sum of ½·precision·‖x‖² over all params.
    - weight_precision: applied to any leaf whose name != "bias"
    - bias_precision:   applied to leaves named "bias"
    """
    flat = flatten_dict(unfreeze(params), keep_empty_nodes=True)
    total = 0.0
    for path, x in flat.items():
        # path is a tuple of key names, e.g. ("Dense_0", "kernel") or ("Dense_0", "bias")
        is_bias = (path[-1] == "bias")
        prec   = bias_precision if is_bias else weight_precision
        total += 0.5 * prec * jnp.sum(x**2)
    return total

def nl_prior_fun(params, *,
                 weight_precision: float,
                 bias_precision:   float = 0.0) -> jnp.ndarray:
    """
    Negative log-prior = ½·λ_w·‖weights‖² + ½·λ_b·‖biases‖².
    By default λ_b = 0 ⇒ biases unpenalized.
    """
    return _l2_tree(params, weight_precision, bias_precision)



# --------------------------------------------------------------------------- #
#  Regression likelihood / posterior                                          #
# --------------------------------------------------------------------------- #
def nl_likelihood_fun_regression(apply_fn: Callable, params, batch) -> jnp.ndarray:
    x, y = batch
    y_hat, log_var = apply_fn(params, x)           # model returns mean & log-var
    var = jnp.exp(log_var)
    se  = jnp.square(y_hat - y)
    return 0.5 * jnp.mean(jnp.log(2 * jnp.pi * var) + se / var)

@jax.jit
def nl_posterior_fun_regression(state, params, batch, prior_precision):
    nll = nl_likelihood_fun_regression(state.apply_fn, params, batch)
    nlp = nl_prior_fun(params, weight_precision=prior_precision)
    return nll + nlp


# --------------------------------------------------------------------------- #
#  Classification likelihood / posterior                                      #
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnames=("apply_fn",))
def nl_likelihood_fun_classification(apply_fn: Callable, params, batch) -> jnp.ndarray:
    x, y = batch
    y         = y.squeeze()
    logits    = apply_fn(params, x)
    one_hot_y = jax.nn.one_hot(y, logits.shape[-1])
    return jnp.mean(optax.softmax_cross_entropy(logits, one_hot_y))


@jax.jit
def nl_posterior_fun_classification(state, params, batch, prior_precision):
    nll = nl_likelihood_fun_classification(state.apply_fn, params, batch)
    nlp = nl_prior_fun(params, weight_precision=prior_precision, bias_precision=prior_precision)
    return nll + nlp

@functools.partial(jax.jit, static_argnums=2) 
def _map_step(state, batch, posterior_fn, prior_precision):
    def loss_fn(params):
        # Treat `state` and everything else as constants; only `params` has grads.
        return posterior_fn(state, params, batch, prior_precision)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state   = state.apply_gradients(grads=grads)
    return new_state, loss


@jax.jit
def _eval_regression(state, batch):
    return nl_likelihood_fun_regression(state.apply_fn, state.params, batch)

@jax.jit
def _eval_classification(state, batch):
    x, y = batch
    logits = state.apply_fn(state.params, x)
    preds  = softmax(logits, axis=-1)
    acc    = jnp.mean(preds.argmax(axis=1) == y.squeeze())
    nll    = nl_likelihood_fun_classification(state.apply_fn, state.params, batch)
    return nll, acc


from tqdm import tqdm
import jax.numpy as jnp

def train_map(
    state,                                             # ← existing TrainState
    train_loader,
    test_loader,
    *,
    model_type          : str,                         # "regressor" | "classifier"
    num_epochs          : int,
    alpha     : float = 0.05,               # α (larger ⇒ stronger decay)
):
    """
    MAP training loop using an already-constructed `TrainState`.
    Returns the updated state.
    """

    # choose posterior & evaluation fns once
    if model_type == "regressor":
        posterior_fn = nl_posterior_fun_regression
        eval_step    = _eval_regression
    else:
        posterior_fn = nl_posterior_fun_classification
        eval_step    = _eval_classification

    # training loop
    pbar = tqdm(range(num_epochs), ncols=80)
    for epoch in pbar:
        # ── optimise one epoch ────────────────────────────────────────────
        for batch in make_iter(train_loader):
            state, train_loss = _map_step(state, batch, posterior_fn, alpha)

        # ── evaluate every 10 epochs ──────────────────────────────────────
        if epoch % 10 == 0:
            test_loss = 0.0
            test_acc  = 0.0
            for batch in make_iter(test_loader):
                metrics   = eval_step(state, batch)
                test_loss += metrics[0]
                if model_type == "classifier":
                    test_acc += metrics[1]

            n = len(test_loader)
            if model_type == "classifier":
                descr = f"[NLL={test_loss/n:6.4f}  ACC={test_acc/n:5.3f}]"
            else:
                descr = f"[NLL={test_loss/n:6.4f}]"
            pbar.set_description(descr)

    return state

