from functools import partial
import pdb
import jax
import jax.numpy as jnp
import optax
from flax.linen import softmax
from tqdm import tqdm


# Regression Loss Functions
@partial(jax.jit, static_argnames=("apply_fn",))
def nl_likelihood_fun_regression(apply_fn, variables, batch):
    # For regression, we use a closed-form negative log-likelihood.
    x, y = batch
    yhat, logvar = apply_fn(variables, x)
    var = jnp.exp(logvar)
    se = optax.squared_error(yhat, y)
    return 0.5 * (jnp.log(2 * jnp.pi * var) + se / var).sum()

@jax.jit
def nl_prior_fun(variables, stdev):
    """Simple (negative log) L2 prior."""
    sum_of_squares = lambda t: jnp.sum(t**2)
    param_leaves = jax.tree_util.tree_leaves(variables["params"])
    # param_leaves = jax.tree_util.tree_leaves(variables)
    l2 = jnp.sum(jnp.array([sum_of_squares(p) for p in param_leaves]))
    return 0.5 / (stdev**2) * l2

@jax.jit
def nl_posterior_fun_regression(state, variables, batch, prior_std=1.0):
    # Posterior is exactly the likelihood plus prior (here both are jitted)
    nll = nl_likelihood_fun_regression(state.apply_fn, variables, batch)
    log_prior = nl_prior_fun(variables, prior_std)
    return nll + log_prior

# Regression Training and Evaluation Steps
@jax.jit
def map_step_regression(state, batch, prior_std=1.0):
    grad_fn = jax.value_and_grad(nl_posterior_fun_regression, argnums=1, has_aux=False)
    loss, grads = grad_fn(state, state.params, batch, prior_std)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step_regression(state, batch):
    loss = nl_likelihood_fun_regression(state.apply_fn, state.params, batch)
    return loss


# Classification Loss Functions
@partial(jax.jit, static_argnames=("apply_fn",))
def nl_likelihood_fun_classification(apply_fn, params, batch):
    """
    Negative log likelihood for classification.
    """
    x, y = batch
    y = y.squeeze()
    logits = apply_fn(params, x)
    num_classes = logits.shape[-1]
    one_hot_y = jax.nn.one_hot(y, num_classes)
    return optax.softmax_cross_entropy(logits=logits, labels=one_hot_y).mean()

@jax.jit
def nl_posterior_fun_classification(state, params, batch, prior_std=1.0):
    """
    Negative log posterior for classification (likelihood plus L2 prior).
    """
    nll = nl_likelihood_fun_classification(state.apply_fn, params, batch)
    nl_prior = nl_prior_fun(state.params, prior_std)
    return nll + nl_prior

# Classification Training and Evaluation Steps
@jax.jit
def map_step_classification(state, batch, prior_std=1.0):
    grad_fn = jax.value_and_grad(nl_posterior_fun_classification, argnums=1, has_aux=False)
    loss, grads = grad_fn(state, state.params, batch, prior_std)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step_classification(state, batch):
    data_input, labels = batch
    logits = state.apply_fn(state.params, data_input)
    preds = jax.nn.softmax(logits, axis=-1)
    pred_labels = preds.argmax(axis=1).astype(jnp.float32)
    loss = nl_likelihood_fun_classification(state.apply_fn, state.params, batch)
    acc = (pred_labels == labels.squeeze()).mean()
    return loss, acc





def train_map(state, trainloader, testloader, model_type, *args, num_epochs=100, **kwargs):
    train_step = map_step_regression if model_type == "regressor" else map_step_classification
    eval_step = eval_step_regression if model_type == "regressor" else eval_step_classification
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        for batch in trainloader:
            state, loss = train_step(state, batch, *args, **kwargs)
        if (epoch) % 10 == 0:
            eloss,eacc = 0.,0.
            for batch in testloader:
                if model_type == "classifier":
                    loss, acc = eval_step(state, batch) # TODO make eval MAP step?
                    eacc += acc
                else:
                    loss = eval_step(state, batch) # TODO make eval MAP step?
                eloss += loss
            additional_info = f"[var={jnp.exp(state.params['logvar']['logvar']):.3f}] " if model_type == "regressor" else f"[Avg. eval acc={eacc/len(testloader):.3f}]"
            pbar.set_description(f"{additional_info}Avg. eval loss: {eloss/len(testloader):.3f} ")
    return state