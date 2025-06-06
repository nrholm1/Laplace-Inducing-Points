import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
from typing import Iterable, Tuple

from src.data import make_iter
from src.ggn  import build_WTW, compute_W_vps
from src.lla  import compute_curvature_approx
from src.utils import count_model_params, flatten_nn_params
from src.train_map import _eval_classification, _eval_regression, _map_step

# ────────────────────────────────────────────────────────────────────────────
#   Laplace log-marginal likelihood  (α-dependent part only)
# ────────────────────────────────────────────────────────────────────────────
def log_marginal_likelihood(
        alpha: float,
        X,                    # batch X
        state,                   # TrainState (flax / haiku)
        model_type: str,
        full_set_size: int | None = None
) -> jnp.ndarray:
    """Return log p(D|α) up to α-independent constants."""
    N = full_set_size or X.shape[0]
    rescale = N / X.shape[0]
    
    # parameter count (minus the learned noise var. for regression)
    D = count_model_params(state.params['params'])
    if model_type == "regressor":
        D -= 1

    # ---- curvature pieces --------------------------------------------------
    W, WT = compute_W_vps(state, X, model_type, full_set_size=None)

    # dense   Wᵀ W   (   d × d  , d ≪ D   )
    dummy  = WT(jnp.zeros(D, dtype=float))
    d      = dummy.size
    WTW    = build_WTW(W, WT, dummy.shape, d, dtype=float, block=1)

    # log |H|  where  H = αI + β WᵀW
    _, logdet_lowrank = jnp.linalg.slogdet(jnp.eye(d) + rescale / alpha * WTW)
    logdet_term       = logdet_lowrank + D * jnp.log(alpha)

    # ---- Gaussian log-prior (α-dependent part) -----------------------------
    flat_p, _ = flatten_nn_params(state.params['params'])
    quad      = -0.5 * alpha * jnp.dot(flat_p, flat_p)
    norm      = 0.5 * D * jnp.log(alpha)     #  −½ D log 2π cancels later
    log_prior = quad + norm

    # final Laplace evidence (no constant terms in α)
    return log_prior - 0.5 * logdet_term


# ────────────────────────────────────────────────────────────────────────────
#   One Adam step on log α   (no JIT, optax objects aren't pytrees)
# ────────────────────────────────────────────────────────────────────────────
def update_alpha(
        log_alpha: jnp.ndarray,
        opt_state: optax.OptState,
        opt:       optax.GradientTransformation,
        *lm_args
) -> Tuple[jnp.ndarray, optax.OptState]:
    """Gradient-ascent on log α (implemented as optax *descent* on −L)."""
    def loss_fn(lalpha):
        return -log_marginal_likelihood(jnp.exp(lalpha), *lm_args)
    grad   = jax.grad(loss_fn)(log_alpha)
    updates, new_state = opt.update(grad, opt_state, log_alpha)
    new_log_alpha      = optax.apply_updates(log_alpha, updates)
    return new_log_alpha, new_state


# ────────────────────────────────────────────────────────────────────────────
#   Full training loop: interleave MAP steps on θ with α hyper-steps
# ────────────────────────────────────────────────────────────────────────────
def train_map_then_alpha(
        state,
        train_loader: Iterable,
        test_loader:  Iterable,
        *,
        model_type:    str,
        num_epochs:    int   = 500,
        alpha0:        float = 1.0,
        alpha_lr:      float = 5e-2,
        alpha_every:   int   = 5,
        burnin:        int   = 100,
        full_set_size: int | None = None):

    log_alpha = jnp.array(jnp.log(alpha0), dtype=float)
    opt_h     = optax.adam(alpha_lr)
    opt_hs    = opt_h.init(log_alpha)

    eval_step = _eval_regression if model_type == "regressor" else _eval_classification
    pbar      = tqdm(range(num_epochs), ncols=95)

    for epoch in pbar:
        # ── MAP optimisation of θ ──────────────────────────────────────────
        for batch in make_iter(train_loader):
            state, _ = _map_step(state, batch, model_type, jnp.exp(log_alpha))

        # ── Hyper-step on α every k epochs ────────────────────────────────
        if (epoch >= burnin) and ((epoch + 1) % alpha_every == 0):
            log_alpha, opt_hs = update_alpha(
                log_alpha, 
                opt_hs, 
                opt_h,
                batch[0],
                state,
                model_type,
                full_set_size
            )

        # ── Periodic evaluation ───────────────────────────────────────────
        if epoch % 4 == 0:
            test_loss = test_acc = 0.0
            for batch in make_iter(test_loader):
                metrics = eval_step(state, batch)
                test_loss += metrics[0]
                if model_type == "classifier":
                    test_acc += metrics[1]
            n = len(test_loader)
            if model_type == "classifier":
                descr = (f"[NLL={test_loss / n:6.4f}  "
                         f"ACC={test_acc / n:5.3f}  "
                         f"α={jnp.exp(log_alpha):6.4f}]")
            else:
                descr = (f"[NLL={test_loss / n:6.4f}  "
                         f"α={jnp.exp(log_alpha):6.4f}]")
            pbar.set_description(descr)

    # return state.replace(alpha=jnp.exp(log_alpha)), jnp.exp(log_alpha)
    return state, jnp.exp(log_alpha)
