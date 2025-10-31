import jax
import jax.numpy as jnp
from functools import partial

from matfree import decomp

# --- X <-> (U, lam) ---
def _orth_eig_from_X(X: jnp.ndarray):
    # X = Qx Rx, Gram in small space
    Qx, Rx = jnp.linalg.qr(X, mode="reduced")         # (D,M),(M,M)
    G = Rx @ Rx.T                                     # (M,M), PSD
    lam, V = jnp.linalg.eigh(G)                       # ascending
    lam = jnp.maximum(lam, 0.0)
    idx = jnp.argsort(lam)[::-1]
    lam = lam[idx]
    U = Qx @ V[:, idx]                                # (D,M), orthonormal
    return U, lam

def _X_from_orth_eig(U: jnp.ndarray, lam: jnp.ndarray):
    return U @ jnp.diag(jnp.sqrt(jnp.maximum(lam, 0.0)))

# --- core: rank-1 update in fixed (M+1)-dim space ---
def _rank1_update_smallspace(U: jnp.ndarray, lam: jnp.ndarray, w: jnp.ndarray, sign: float,
                             eps: float = 1e-12):
    """
    A = U diag(lam) U^T  ->  A' = A + sign * w w^T
    Returns (U_new, lam_new) with same column count M as U.
    """
    D, M = U.shape

    # Decompose w = U u + r, r ⟂ span(U)
    u = U.T @ w                                 # (M,)
    r = w - U @ u                               # (D,)
    rn = jnp.linalg.norm(r)
    rhat = r / (rn + eps)                       # safe: if rn=0, rhat=0

    # Always augment to M+1 (static shape)
    U_aug = jnp.concatenate([U, rhat[:, None]], axis=1)     # (D, M+1)
    z     = jnp.concatenate([u, jnp.array([rn])], axis=0)   # (M+1,)
    diag  = jnp.concatenate([lam, jnp.array([0.0])], axis=0)# (M+1,)

    # Small (M+1)x(M+1) update
    K = jnp.diag(diag) + sign * jnp.outer(z, z)             # symmetric
    evals, evecs = jnp.linalg.eigh(K)                        # ascending
    idx = jnp.argsort(evals)[::-1]                           # descending
    evals = jnp.maximum(evals[idx], 0.0)
    evecs = evecs[:, idx]

    # Keep top M and map back
    evals_M = evals[:M]
    evecs_M = evecs[:, :M]                                   # (M+1, M)
    U_new = U_aug @ evecs_M                                  # (D, M)
    U_new, _ = jnp.linalg.qr(U_new, mode="reduced")          # re-orthonormalize
    lam_new = evals_M
    return U_new, lam_new

# --- public API ---
def modify_X(X: jnp.ndarray, w: jnp.ndarray, mode: str = "up"):
    sign = +1.0 if mode == "up" else -1.0
    U, lam = _orth_eig_from_X(X)
    U_new, lam_new = _rank1_update_smallspace(U, lam, w, sign=sign)
    return _X_from_orth_eig(U_new, lam_new)

def batch_modify_X(X: jnp.ndarray, W: jnp.ndarray, mode: str = "up"):
    sign = +1.0 if mode == "up" else -1.0
    U, lam = _orth_eig_from_X(X)
    Wflat = W.reshape(-1, W.shape[-1])

    def f(carry, w_i):
        Uc, lc = carry
        Un, ln = _rank1_update_smallspace(Uc, lc, w_i, sign=sign)
        return (Un, ln), None

    (U_fin, lam_fin), _ = jax.lax.scan(f, (U, lam), Wflat)
    return _X_from_orth_eig(U_fin, lam_fin)




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

    Q, T = lanczos_tridiag(GGN_full, v0, M)      # ! full
    # Q, T = lanczos_tridiag(GGN, v0, M)      # ! base
    Tsqrt = jnp.linalg.cholesky(T + 1e-6*jnp.eye(len(T)))
    X = Q @ Tsqrt

    # Add back EXTRA terms (modified or not)
    WT_extra_dense = jax.jacfwd(WT_extra)(v0)     # (..., D)

    X_mod = batch_modify_X(X, WT_extra_dense, mode="down")
    X_mod = batch_modify_X(X_mod, WT_extra_dense, mode="up")
    # X_mod = batch_modify_X(X, WT_extra_dense, mode="up")
    GGN_hat = X_mod @ X_mod.T
    
    fig, axs = plt.subplots(2, 3, figsize=(23, 8))

    #* set a smaller range
    _start = 000
    _end   = 150

    # for downdate
    # _GGN_dense      = beta * GGN_dense[_start:_end, _start:_end]
    # _GGN_hat        = beta * GGN_hat[_start:_end, _start:_end]
    # _GGN_full_dense = beta_full * GGN_full_dense[_start:_end, _start:_end]
    
    # for update
    _GGN_dense      =  GGN_full_dense[_start:_end, _start:_end]
    _GGN_hat        =  GGN_hat[_start:_end, _start:_end]
    _GGN_full_dense =  GGN_dense[_start:_end, _start:_end]

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
