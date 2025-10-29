import pdb
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import ShapeDtypeStruct as SDS
from functools import partial

from matfree.lstsq import lsmr

from src.utils import count_model_params, flatten_nn_params


def get_f_apply(map_state, unravel_fn, model_type):
    def f_apply(flatp, x):
        p = unravel_fn(flatp)
        if model_type == "regressor":
            return map_state.apply_fn(p, x, return_logvar=False)
        else:
            variables = {"params": p, "batch_stats": map_state.batch_stats}
            return map_state.apply_fn(variables, x, train=False, mutable=False)
    return f_apply



def H(p):
    """Closed-form loss Hessian of softmax."""
    return jnp.diag(p) - jnp.outer(p, p)
    
def Hsqrt(p):
    """Non-Symmetric matrix S s.t. H = S Sᵀ"""
    q = jnp.sqrt(p)
    P = jnp.eye(p.size, dtype=p.dtype) - jnp.outer(q, q)
    S = q[:, None] * P
    return S

def Hsqrtsym(p, tol=1e-6):
    """Symmetric matrix S s.t. H = S Sᵀ"""
    _H = H(p)
    E,V = jnp.linalg.eigh(_H)
    S = jnp.where(E > tol, jnp.sqrt(E), 0.0)
    return (V*S) @ V.T

def Hinvsqrt(p, tol=1e-6):
    """Also symmetric matrix."""
    _H = H(p)
    E,V = jnp.linalg.eigh(_H)
    E_safe = jnp.maximum(E, tol)
    S = jnp.where(E > tol, 1.0/jnp.sqrt(E_safe), 0.0)
    return (V*S) @ V.T




def get_lsmr_system(data, _alpha_inv_sqrt, beta_sqrt, v, map_state, *, model_type='classifier'):
    # todo ATM hardcoded to model_type == classifier
    flat_params,_unravel_fn = flatten_nn_params(map_state.params)
    f_apply = get_f_apply(map_state, _unravel_fn, model_type)
    f_out, vj_fun = jax.vjp(lambda _p: f_apply(_p, data), flat_params)
    p1 = jax.nn.softmax(f_out, axis=1)
    B = beta_sqrt * jax.vmap(Hsqrt)(p1)
    
    _, unravel_fn = ravel_pytree(v)
    
    def vecmat(_flat_vec):
        vec = unravel_fn(_flat_vec)
        x0 = jnp.einsum('bij,bj->bi', B, vec)
        x1 = vj_fun(x0)[0]
        x2,_ = ravel_pytree(x1)
        return _alpha_inv_sqrt * x2
    
    v0 = jnp.einsum('bji,bj->bi', B, v)
    flat_v0,_ = ravel_pytree(v0)
    
    return vecmat, flat_v0




def get_K(_data, _alpha, _beta, map_state, *, atol=1e-3, btol=1e-3, ctol=1e-4):
    solve = lsmr(atol=atol, btol=btol, ctol=ctol)
    _alpha_inv_sqrt = jnp.sqrt(1.0 / _alpha)
    _beta_sqrt = jnp.sqrt(_beta)
    
    # @jax.jit
    def K(v):
        vecmat, u1 = get_lsmr_system(_data, _alpha_inv_sqrt, _beta_sqrt, v, map_state)
        xi, info = solve(vecmat, u1, damp=1.0)
        return _alpha_inv_sqrt * xi, info
    
    return K

# def get_K(data, alpha, beta, map_state, *, atol=1e-3, btol=1e-3, ctol=1e-4):
#     solve = lsmr(atol=atol, btol=btol, ctol=ctol)
#     alpha_inv_sqrt = jnp.sqrt(1.0 / alpha)
#     beta_sqrt = jnp.sqrt(beta)

#     # ----- build once -----
#     flat_params, unravel_params = flatten_nn_params(map_state.params)
#     f_apply = get_f_apply(map_state, unravel_params, 'classifier')

#     # one-time linearization (cheap to reuse in the matvecs)
#     f_out, vjp_fn = jax.vjp(lambda p: f_apply(p, data), flat_params)
#     p = jax.nn.softmax(f_out, axis=1)                      # [B, C]
#     B = beta_sqrt * jax.vmap(Hsqrt)(p)                     # [B, C, C]  (kept dense)
#     v_shape = f_out.shape                                  # (B, C)

#     # param-space flattening helper (structure fixed)
#     # Using reshape for v; for param cotangent we still ravel, but we avoid ravel for v.
#     def vecmat(flat_vec):                                   # A @ x
#         vec = flat_vec.reshape(v_shape)                     # R^{BC} -> [B,C]
#         x0 = jnp.einsum('bij,bj->bi', B, vec)               # [B,C]
#         x1 = vjp_fn(x0)[0]                                  # params-like pytree
#         x2, _ = ravel_pytree(x1)                            # R^{D}
#         return alpha_inv_sqrt * x2

#     def apply_BT(v):                                        # B^T @ v
#         return jnp.einsum('bji,bj->bi', B, v)               # [B,C]
#     # ----- end build once -----

#     def K(v):
#         # Only the RHS changes across calls
#         u1 = apply_BT(v).reshape(-1)                        # flatten to R^{BC}
#         xi, info = solve(vecmat, u1, damp=1.0)              # xi in R^{BC}
#         return alpha_inv_sqrt * xi, info

#     return K



def sample_theta(key, alpha, D, *, num_samples=1):
    alpha_inv_sqrt = jnp.sqrt(1.0 / alpha)
    return alpha_inv_sqrt * jax.random.normal(key, (num_samples, D))


def sample_logits_given_theta(key, theta0, real_data, map_state, *, beta, model_type='classifier'):
    num_samples = theta0.shape[0]
    flat_params,unravel_fn = flatten_nn_params(map_state.params)
    f_apply = get_f_apply(map_state, unravel_fn, model_type)
    
    def linearized_fun(_t0):
        _f_out,_jv = jax.jvp(
            lambda _p: f_apply(_p, real_data), (flat_params,), (_t0,)
        )
        return _f_out, _jv
    
    f_out, jv = jax.vmap(linearized_fun)(theta0)
    f_out = f_out[0] # since batch-dimension just means we have 'num_samples' copies
    
    p = jax.nn.softmax(f_out, axis=1)
    _Hinvsqrt = jax.vmap(Hinvsqrt)(p) / jnp.sqrt(beta)
    
    eps = jax.random.normal(key, (num_samples,) + f_out.shape)
    bmm1 = jnp.einsum("bij,nbj->nbi",_Hinvsqrt, eps)
    
    return bmm1 + jv


def get_conditional_theta_sampler(data, alpha, beta, map_state, *, atol=1e-3, btol=1e-3, ctol=1e-4):
    _K = get_K(data, alpha, beta, map_state, atol=atol, btol=btol, ctol=ctol)
    D = count_model_params(map_state.params)
    
    # @partial(jax.jit, static_argnames=("num_samples",))
    def sample_theta_given_data(key, *, num_samples):
        key_theta, key_logits = jax.random.split(key, 2)
        theta0 = sample_theta(key_theta, alpha, D, num_samples=num_samples)
        y0 = sample_logits_given_theta(key_logits, theta0, data, map_state, beta=beta)
        residuals = - y0
        corr,info = jax.vmap(_K)(residuals)
        return theta0 + corr#, info

    return sample_theta_given_data



"""
Preliminary test cases
"""

if __name__ == '__main__':
    import optax
    from tqdm import tqdm
    
    from matfree import stochtrace

    from src.utils import load_checkpoint, load_yaml, count_model_params
    from src.scalemodels import TrainState, EMPTY_STATS
    from src.toymodels import SimpleClassifier
    from src.toydata import get_dataloaders
    from src.ggn import compute_W_vps, compute_ggn_vp, compute_ggn_dense
    from src.utils import flatten_nn_params, print_dict
    
    _key = lambda x: jax.random.PRNGKey(x)

    dataset = 'banana'
    model_name = f'toyclassifier_{dataset}'
    cfg_path = f'config/toy/{model_name}.yml'

    cfg = load_yaml(cfg_path)
    print(f"Loaded config: {cfg_path}")
    print_dict(cfg)
    
    model_cfg = cfg['model']
    opt_cfg = cfg['optimization']
    alpha =  opt_cfg["alpha"]
    map_cfg = opt_cfg["map"]
    full_set_size = opt_cfg["full_set_size"]

    model_type = model_cfg.get("name", "regressor")  # 'regressor' or 'classifier'
    num_h = model_cfg["num_h"]
    num_l = model_cfg["num_l"]
    num_c = model_cfg.get("num_c", 2) if model_type == "classifier" else 1
    rng_model = jax.random.PRNGKey(model_cfg["seed"])
    map_batch_size = map_cfg["batch_size"]
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


    flat_params, unravel_fn = flatten_nn_params(map_state.params)

    REAL_DATA  = next(iter(train_loader))[0]
    IP_DATA    = next(iter(train_loader))[0]

    GGN_data_dense, *_ =  compute_ggn_dense(map_state, REAL_DATA, model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)
    GGN_ip_dense, *_   =  compute_ggn_dense(map_state, IP_DATA,  model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)

    GGN_data =  compute_ggn_vp(map_state, REAL_DATA,  model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)
    GGN_ip   =  compute_ggn_vp(map_state, IP_DATA,   model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)
    W, WT    =  compute_W_vps( map_state, REAL_DATA,   model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)
    Wip, WTip=  compute_W_vps( map_state, IP_DATA, model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)

    beta      = full_set_size / IP_DATA.shape[0]
    beta_sqrt = jnp.sqrt(beta)

    
    theta_sampler = get_conditional_theta_sampler(IP_DATA, alpha, beta, map_state, atol=1e-6, btol=1e-6, ctol=1e-7)
    
    
    """##
    STATS
    ##"""
    num_samples = 10000
    # alot_of_samples,info = theta_sampler(_key(43434310), num_samples=num_samples)
    alot_of_samples = theta_sampler(_key(43434310), num_samples=num_samples)
    
    sample_mean = alot_of_samples.mean(axis=0, keepdims=True)
    centered = (alot_of_samples - sample_mean)
    sample_cov = centered.T @ centered / (num_samples - 1)
    sample_precision = jnp.linalg.inv(sample_cov)
    iso_prior = alpha * jnp.eye(D)
    precision = iso_prior + beta*GGN_ip_dense
    GGN_ip_inv_dense = jnp.linalg.inv(precision)
    true_trace = jnp.linalg.trace((iso_prior + beta*GGN_data_dense) @ (GGN_ip_inv_dense))    

    # integrand = stochtrace.integrand_trace()
    # sampler = lambda __key: theta_sampler(__key, num_samples=10_000)
    # estimate = stochtrace.estimator(integrand, sampler)
    # est_trace = estimate(lambda v: alpha*v + beta*GGN_data(v), _key(188_88_88_88))

    # print(f"Est. trace    = {est_trace:.2f}")
    # print(f"True trace    = {true_trace:.2f}")
    
    # assert jnp.isclose(true_trace, est_trace, rtol=0.05), f"Estimated trace does not match ground truth! EST={est_trace:.1f} vs. TRUE={true_trace:.1f}"
    
    import matplotlib.pyplot as plt
    from matplotlib import colors
    # import seaborn as sns

    cmap = 'seismic' #sns.color_palette('vlag', as_cmap=True)

    _start = 0
    _end   = 200
    true_matrix =    precision[_start:_end, _start:_end] #cov[:_k,:_k]
    sampled_matrix = sample_precision[_start:_end, _start:_end] #sample_cov[:_k,:_k]
    diff = jnp.abs(true_matrix - sampled_matrix) / (jnp.abs(true_matrix) + 1e-0)

    fig, axs = plt.subplots(1, 3, figsize=(25, 6))

    # Make limits symmetric so 0 is centered
    m0 = jnp.abs(true_matrix).max()
    m1 = jnp.abs(sampled_matrix).max()
    norm  = colors.TwoSlopeNorm(vmin=-max(m0,m1), vcenter=0, vmax=max(m0,m1))
    norm0 = norm # colors.TwoSlopeNorm(vmin=-m0, vcenter=0, vmax=m0)
    norm1 = norm # colors.TwoSlopeNorm(vmin=-m1, vcenter=0, vmax=m1)

    im0 = axs[0].imshow(true_matrix, cmap=cmap, norm=norm0)
    im1 = axs[1].imshow(sampled_matrix, cmap=cmap, norm=norm1)
    im2 = axs[2].imshow(diff, cmap='viridis')

    axs[0].set_title("True Precision")
    axs[1].set_title("Sample Precision")
    axs[2].set_title("|true - sample| / |true|")

    # fig.colorbar(im0)
    fig.colorbar(im1)
    fig.colorbar(im2)

    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    # plt.show()
    
    plt.savefig("precision.png")
    
    # pdb.set_trace()
    