import jax
import jax.numpy as jnp
from functools import partial

from matfree import decomp


# @jax.jit
def stack_zero_then_XT(X: jnp.ndarray) -> jnp.ndarray:
    """A = [[0]; X^T]  -> shape (M+1, D) for X in R^{D x M}."""
    return jnp.vstack([jnp.zeros((1, X.shape[0]), dtype=X.dtype), X.T])

# @jax.jit
def rotate_rows(A: jnp.ndarray, i: int, j: int, c: float, s: float) -> jnp.ndarray:
    """Apply a single Givens to rows i and j of A (row-rotation)."""
    ri, rj = A[i], A[j]
    new_i = c * ri + s * rj
    new_j = -s * ri + c * rj
    A = A.at[i].set(new_i)
    A = A.at[j].set(new_j)
    return A

# @partial(jax.jit, static_argnames=("reverse",))
def apply_row_givens_sequence_signed(
    A: jnp.ndarray, c: jnp.ndarray, s: jnp.ndarray, *, reverse: bool = False
) -> jnp.ndarray:
    M = s.shape[0]

    def body(k, A_):
        idx = jnp.where(reverse, M - 1 - k, k)
        sign = jnp.where(reverse, -1.0, +1.0)
        return rotate_rows(A_, 0, idx + 1, c[idx], sign * s[idx])

    return jax.lax.fori_loop(0, M, body, A)


# @jax.jit
def givens_from_coeffs(a: jnp.ndarray, eps: float = 1e-12):
    """Back-solve s_k, c_k from a (requires ||a|| <= 1)."""
    M = a.shape[0]
    c = jnp.empty(M, dtype=a.dtype)
    s = jnp.empty(M, dtype=a.dtype)

    Cprod = 1.0
    # Python loop is fine here (O(M)); application is jitted.
    for k in range(M - 1, -1, -1):
        sk = a[k] / (Cprod + eps)
        sk = jnp.clip(sk, -1.0, 1.0)
        ck = jnp.sqrt(jnp.maximum(0.0, 1.0 - sk * sk))
        s = s.at[k].set(sk)
        c = c.at[k].set(ck)
        Cprod = Cprod * ck
    return c, s


# ---------- public API ----------

def _rotations_core(X: jnp.ndarray, w: jnp.ndarray, *, reverse: bool, tol: float = 1e-8):
    """Shared core for both downdate and update."""
    a, *_ = jnp.linalg.lstsq(X, w, rcond=None)
    # Optional checks (kept commented to mirror your code)
    # res = jnp.linalg.norm(X @ a - w)
    # if res > tol: raise ValueError(...)
    # if jnp.linalg.norm(a) > 1.0 + 1e-10: raise ValueError(...)
    A = stack_zero_then_XT(X)
    c, s = givens_from_coeffs(a)
    GA = apply_row_givens_sequence_signed(A, c, s, reverse=reverse)
    return GA[1:].T

def rotations_for_w_down(X: jnp.ndarray, w: jnp.ndarray, tol: float = 1e-8):
    """DOWndate: X -> X~ such that X~X~^T ≈ A - w w^T."""
    return _rotations_core(X, w, reverse=False, tol=tol)


def rotations_for_w_up(X: jnp.ndarray, w: jnp.ndarray, tol: float = 1e-8):
    """UPdate: X -> X^ such that X^X^T ≈ A + w w^T."""
    return _rotations_core(X, w, reverse=True, tol=tol)

@partial(jax.jit, static_argnames=("mode",))
def apply_rotations(WTp_dense, X, *, mode: str = "down"):
    steps = WTp_dense.reshape(-1, WTp_dense.shape[-1])  # (K, D)
    is_update = (mode == "up")

    def body(i, X_):
        return jax.lax.cond(
            is_update,
            lambda Xk: rotations_for_w_up(Xk, steps[i]),
            lambda Xk: rotations_for_w_down(Xk, steps[i]),
            X_,
        )

    return jax.lax.fori_loop(0, steps.shape[0], body, X)


def lanczos_tridiag(matvec, v0, m):
    tri = decomp.tridiag_sym(m)
    out = tri(matvec, v0)
    Q, T = out.Q_tall, out.J_small
    return Q, T


def eigvals_desc(A):
    return jnp.sort(jnp.linalg.eigvalsh(A))[::-1]







if __name__ == '__main__':
    # load MAP state
    import optax
    import matplotlib.pyplot as plt
    import seaborn as sns

    from src.utils import load_checkpoint, load_yaml, count_model_params
    from src.scalemodels import TrainState, EMPTY_STATS
    from src.toymodels import SimpleClassifier
    from src.toydata import get_dataloaders

    _key = lambda x: jax.random.PRNGKey(x)

    dataset = 'banana'
    model_name = f'toyclassifier_{dataset}'
    cfg_path = f'config/toy/{model_name}.yml'

    cfg = load_yaml(cfg_path)
    model_cfg = cfg['model']
    opt_cfg = cfg['optimization']
    alpha = opt_cfg["alpha"]
    map_cfg = opt_cfg["map"]
    N = opt_cfg["full_set_size"]

    model_type = model_cfg.get("name", "regressor")  # 'regressor' or 'classifier'
    num_h = model_cfg["num_h"]
    num_l = model_cfg["num_l"]
    num_c = model_cfg.get("num_c", 2) if model_type == "classifier" else 1
    rng_model = jax.random.PRNGKey(model_cfg["seed"])
    map_batch_size = 256 # map_cfg["batch_size"]
    epochs_map = map_cfg["epochs"]
    lr_map = map_cfg["lr"]

    model = SimpleClassifier(numh=num_h, numl=num_l, numc=num_c)

    train_loader, test_loader, _ = get_dataloaders(dataset=dataset, batch_size=map_batch_size)

    dummy_input = next(iter(train_loader))[0][:1]
    variables = model.init(rng_model, dummy_input)
    optimizer_map = optax.adam(1e-3)
    model_state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer_map,
        batch_stats = variables.get('batch_stats', EMPTY_STATS),
    )
    map_ckpt_prefix = f"map_{dataset}"

    map_state = load_checkpoint(
        ckpt_dir="checkpoint/map/",
        prefix=map_ckpt_prefix,
        target=model_state
    )

    D = count_model_params(variables)
    
    # compute W and GGN

    from src.ggn import compute_W_vps, compute_ggn_vp, compute_ggn_dense
    from src.utils import flatten_nn_params

    flat_params, unravel_fn = flatten_nn_params(map_state.params)

    num_mod_terms = 64

    FULL_DATA  = next(iter(train_loader))[0]
    GGN_DATA   = FULL_DATA[:-num_mod_terms]
    EXTRA_TERMS = FULL_DATA[-num_mod_terms:]
    # NEW_DATA   = next(iter(train_loader))[0][:num_mod_terms]
    
    M_full = FULL_DATA.shape[0]
    M = M_full - num_mod_terms
    beta      = N/M
    beta_full = N/M_full

    GGN_full_dense, *_ =  compute_ggn_dense(map_state, FULL_DATA, model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)
    GGN_dense, *_      =  compute_ggn_dense(map_state, GGN_DATA,  model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)

    GGN_full =  compute_ggn_vp(map_state, FULL_DATA,  model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)
    GGN      =  compute_ggn_vp(map_state, GGN_DATA,   model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)
    W, WT    =  compute_W_vps( map_state, GGN_DATA,   model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)
    W_extra, WT_extra  =  compute_W_vps( map_state, EXTRA_TERMS, model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)
    # W_new, WT_new  =  compute_W_vps( map_state, NEW_DATA, model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)
    
    def Msqrtsym(M, tol=1e-6):
        """Symmetric matrix square root"""
        E,V = jnp.linalg.eigh(M)
        S = jnp.where(E > tol, jnp.sqrt(E), 0.0)
        return (V*S) @ V.T    
    

    D = count_model_params(variables)
    M = 24
    key = _key(424242)

    v0 = jax.random.normal(key, (D,))

    Q, T = lanczos_tridiag(GGN_full, v0, M)
    # Tsqrt = jnp.linalg.cholesky(T + 1e-6*jnp.eye(len(T)))
    Tsqrt = Msqrtsym(T, tol=1e-6)
    X = Q @ Tsqrt
    
    # # ! DOWNDATE
    WT_extra_dense = jax.jacfwd(WT_extra)(v0)
    Xtilde = apply_rotations(WT_extra_dense, X, mode="down")
    
    # ! UPDATE BACK -> PROBLEMATIC :((
    # Xtilde = apply_rotations(WT_extra_dense, Xtilde, mode="up")
    
    # ! RECONSTRUCT
    GGN_hat = Xtilde@Xtilde.T
    
    fig, axs = plt.subplots(2, 3, figsize=(23, 8))

    #* set a smaller range
    _start = 000
    _end   = 150

    # for downdate
    _GGN_dense      = beta * GGN_dense[_start:_end, _start:_end]
    _GGN_hat        = beta * GGN_hat[_start:_end, _start:_end]
    _GGN_full_dense = beta_full * GGN_full_dense[_start:_end, _start:_end]
    
    # for update
    # _GGN_dense      =  GGN_full_dense[_start:_end, _start:_end]
    # _GGN_hat        =  GGN_hat[_start:_end, _start:_end]
    # _GGN_full_dense =  GGN_dense[_start:_end, _start:_end]

    diff_1 = jnp.abs(_GGN_dense - _GGN_hat)
    diff_2 = jnp.abs(_GGN_dense - _GGN_full_dense)
    vmax_diff = max(diff_1.max(), diff_2.max())
    diff_color = sns.color_palette('rocket', as_cmap=True) # 'seismic'

    # --- Row 1: dense vs hat ---

    vmin_1 = min(_GGN_hat.min(), _GGN_dense.min())
    vmax_1 = max(_GGN_hat.max(), _GGN_dense.max())

    im00 = axs[0, 0].imshow(_GGN_dense, vmin=vmin_1, vmax=vmax_1)
    im01 = axs[0, 1].imshow(_GGN_hat,   vmin=vmin_1, vmax=vmax_1)
    fig.colorbar(im01, ax=axs[0, 1])

    im02 = axs[0, 2].imshow(diff_1, cmap=diff_color, vmin=0, vmax=vmax_diff)
    fig.colorbar(im02, ax=axs[0, 2])

    axs[0, 0].set_title(f'Ground truth base GGN')
    axs[0, 1].set_title(f'Downdated GGN')
    axs[0, 2].set_title(f'Difference |dense - hat|, max={diff_1.max():.2f}')

    # --- Row 2: dense vs full_dense ---


    vmin_2 = min(_GGN_full_dense.min(), _GGN_dense.min())
    vmax_2 = max(_GGN_full_dense.max(), _GGN_dense.max())

    im10 = axs[1, 0].imshow(_GGN_dense,      vmin=vmin_2, vmax=vmax_2)
    im11 = axs[1, 1].imshow(_GGN_full_dense, vmin=vmin_2, vmax=vmax_2)
    fig.colorbar(im11, ax=axs[1, 1])

    im12 = axs[1, 2].imshow(diff_2, cmap=diff_color, vmin=0, vmax=vmax_diff)
    fig.colorbar(im12, ax=axs[1, 2])

    axs[1, 0].set_title(f'Ground truth base GGN')
    axs[1, 1].set_title(f'Full dense GGN')
    axs[1, 2].set_title(f'Difference |dense - full_dense|, max={diff_2.max():.2f}')

    plt.suptitle(f"Plot Range [{_start}:{_end}], #matvecs={M}")
    
    for axi in axs.flatten(): axi.grid()
    plt.show()
