import jax
import jax.numpy as jnp
import optax

from tqdm import tqdm


def nl_likelihood_fun(apply_fn, params, batch):
    # ! for regression! closed form:
    x,y = batch
    yhat, logvar = apply_fn(params, x)
    var = jnp.exp(logvar)               
    se = optax.squared_error(yhat,y)
    return 0.5 * (jnp.log(2*jnp.pi*var) + se/var).sum()


def nl_prior_fun(params, stdev):
    """Simple (negative log) L2 prior"""
    sum_of_squares = lambda t: jnp.sum(t**2)
    param_leaves = jax.tree_util.tree_leaves(params)
    l2 = jnp.sum(jnp.array([sum_of_squares(p) for p in param_leaves]))
    return .5 / (stdev**2) * l2


def nl_posterior_fun(state, params, batch, prior_std=1.0):
    """(negative log) Gaussian likelihood"""
    nll = nl_likelihood_fun(state.apply_fn, params, batch) # Gaussian likelihood => NLL ~ MSE
    log_prior = nl_prior_fun(state.params, prior_std)
    return nll + log_prior


@jax.jit
def regression_map_step(state, batch, prior_std=1.0):
    grad_fn = jax.value_and_grad(nl_posterior_fun, argnums=1, has_aux=False)
    loss,grads = grad_fn(state, state.params, batch, prior_std)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, batch):
    loss = nl_likelihood_fun(state.apply_fn, state.params, batch)
    return loss


def train_map(state, trainloader, testloader, train_step, *args, num_epochs=100, **kwargs):
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        for batch in trainloader:
            state, loss = train_step(state, batch, *args, **kwargs)
        if (epoch) % 10 == 0:
            eloss = 0
            for batch in testloader:
                eloss += eval_step(state, batch) # TODO make eval MAP step?
            pbar.set_description(f"[var={jnp.exp(state.params['params']['logvar']):.3f}] Avg. eval loss: {eloss/len(testloader):.3f}")
    return state