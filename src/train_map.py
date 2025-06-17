from functools import partial
import functools
import pdb
from typing import Callable, Iterable, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.linen import softmax
from tqdm import tqdm
from flax.core.frozen_dict import unfreeze, FrozenDict
from flax.traverse_util   import flatten_dict

from src.data import make_iter
from src.scalemodels import TrainState


def _l2_tree(params: FrozenDict,
             weight_precision: float,
             bias_precision: float = 0.0) -> jnp.ndarray:
    flat = flatten_dict(unfreeze(params), keep_empty_nodes=True)
    total = 0.0
    for path, x in flat.items():
        is_bias = path[-1] == "bias"
        prec    = bias_precision if is_bias else weight_precision
        total  += 0.5 * prec * jnp.sum(x ** 2)
    return total


def _nl_prior(params: FrozenDict,
              *,
              weight_precision: float,
              bias_precision: float = 0.0) -> jnp.ndarray:
    return _l2_tree(params, weight_precision, bias_precision)


#  Model wrapper that handles BatchNorm collections
def _apply_model(state: TrainState, x, *, train: bool):
    # vars_in = {"params": state.params['params'], "batch_stats": state.batch_stats}
    vars_in = {"params": state.params, "batch_stats": state.batch_stats}
    if train:
        y, new_vars = state.apply_fn(vars_in, x, train=True,
                                       mutable=["batch_stats"])
        state = state.replace(batch_stats=new_vars["batch_stats"])
    else:
        y = state.apply_fn(vars_in, x, train=False, mutable=False)
    return state, y


@functools.partial(jax.jit, static_argnums=(2,))
def _map_step(state: TrainState,
             batch,
             model_type: str,
             prior_precision: float):
    """Performs one optimisation step and returns the new state & loss."""

    def loss_fn(params, batch_stats):
        tmp_state = state.replace(params=params, batch_stats=batch_stats)
        tmp_state, outputs = _apply_model(tmp_state, batch[0], train=True)

        # NLL
        if model_type == "classifier":
            y       = batch[1].squeeze()
            logits  = outputs
            one_hot = jax.nn.one_hot(y, logits.shape[-1])
            nll     = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))
            nlp     = _nl_prior(params,
                                 weight_precision=prior_precision,
                                 bias_precision=prior_precision)
        # MSE
        else:
            y       = batch[1]
            y_hat, log_var = outputs
            var     = jnp.exp(log_var)
            se      = jnp.square(y_hat - y)
            nll     = 0.5 * jnp.mean(jnp.log(2 * jnp.pi * var) + se / var)
            nlp     = _nl_prior(params, weight_precision=prior_precision)

        loss = nll + nlp
        return loss, tmp_state.batch_stats

    (loss, new_bs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, state.batch_stats)

    new_state = state.apply_gradients(grads=grads).replace(batch_stats=new_bs)
    return new_state, loss


@jax.jit
def _eval_classification(state: TrainState, batch):
    state, logits = _apply_model(state, batch[0], train=False)

    preds = softmax(logits, axis=-1)
    acc   = jnp.mean(preds.argmax(axis=1) == batch[1].squeeze())

    one_hot = jax.nn.one_hot(batch[1].squeeze(), logits.shape[-1])
    nll     = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))
    return nll, acc


@jax.jit
def _eval_regression(state: TrainState, batch):
    state, outputs = _apply_model(state, batch[0], train=False)
    y_hat, log_var = outputs

    var = jnp.exp(log_var)
    se  = jnp.square(y_hat - batch[1])
    nll = 0.5 * jnp.mean(jnp.log(2 * jnp.pi * var) + se / var)
    return nll

def train_map(state: TrainState,
              train_loader: Iterable,
              test_loader: Iterable,
              *,
              model_type: str,
              num_epochs: int,
              alpha:      float):

    eval_step = _eval_regression if model_type == "regressor" else _eval_classification

    pbar = tqdm(range(num_epochs), ncols=80)
    for epoch in pbar:
        # ── optimisation ────────────────────────────────────────────
        for batch in make_iter(train_loader):
            state, train_loss = _map_step(state, batch, model_type, alpha)

        # ── evaluation every 4 epochs ──────────────────────────────
        if epoch % 1 == 0:
            test_loss = 0.0
            test_acc  = 0.0
            for batch in make_iter(test_loader):
                metrics = eval_step(state, batch)
                test_loss += metrics[0]
                if model_type == "classifier":
                    test_acc += metrics[1]

            n = len(test_loader)
            if model_type == "classifier":
                descr = f"[NLL={test_loss / n:6.4f}  ACC={test_acc / n:5.3f}]"
            else:
                descr = f"[NLL={test_loss / n:6.4f}]"
            pbar.set_description(descr)

    return state
