import math
import jax
import jax.numpy as jnp
import optax

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import load_array_checkpoint, load_yaml, ip_config_from_dict, flatten_nn_params, count_model_params
from src.scaledata import get_dataloaders as get_scale_dataloaders
from src.scalemodels import EMPTY_STATS, TrainState, get_model as get_scale_model
from src.ggn import compute_W_vps, compute_ggn_vp, compute_ggn_dense

from matfree import stochtrace
from src.sampling2 import get_conditional_theta_sampler


if __name__ == "__main__":
    _key = lambda x: jax.random.PRNGKey(x)

    dataset = 'mnist'
    model_name = f'mini_{dataset}'
    cfg_path = f'config/scale/{model_name}.yml'
    cfg = load_yaml(cfg_path)
    model_cfg = cfg["model"]
    opt_cfg = cfg["optimization"]
    
    model_type = model_cfg["type"]
    variant = "scale"

    # Common options
    alpha_default = opt_cfg["alpha"]
    full_set_size = opt_cfg["full_set_size"]

    # MAP config
    map_cfg = opt_cfg["map"]
    map_batch_size = map_cfg["batch_size"]
    epochs_map = map_cfg["epochs"]
    lr_map = map_cfg["lr"]

    # IP config
    ip_cfg_raw = opt_cfg["ip"]
    m_ip = ip_cfg_raw["m"]
    epochs_ip = ip_cfg_raw["epochs"]
    batch_size_ip = ip_cfg_raw["batch_size"]
    lr_ip = ip_cfg_raw["lr"]
    seed_ip = ip_cfg_raw["seed"]
    ip_cfg_pytree = ip_config_from_dict(
        ip_cfg_raw,
        model_type=model_type,
        scalable=variant != "toy-dense",
    )
    
    #* Create model state
    train_loader, test_loader, val_loader = get_scale_dataloaders(dataset, batch_size_ip, num_workers=None, aug=None)
    dummy_input = next(iter(train_loader))[0][:1]  # e.g., (1, 28, 28, 1)
    model_type = model_cfg["type"]
    model_seed = model_cfg["seed"]
    rng_model = jax.random.PRNGKey(model_seed)
    model = get_scale_model(model_cfg)
    variables = model.init(rng_model, dummy_input, train=True)
    optimizer_map = optax.adam(learning_rate=lr_map)
    
    map_state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer_map,
        batch_stats=variables.get("batch_stats", EMPTY_STATS),
    )
    
    
    #* GGN stuff
    D = count_model_params(variables)
    flat_params, unravel_fn = flatten_nn_params(map_state.params)
    
    alpha = alpha_default

    D = count_model_params(variables)
    key = _key(424242)

    REAL_DATA  = next(iter(train_loader))[0]
    IP_DATA    = next(iter(train_loader))[0]

    GGN_data_dense, *_ =  compute_ggn_dense(map_state, REAL_DATA, model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)
    GGN_ip_dense, *_   =  compute_ggn_dense(map_state, IP_DATA,  model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)

    GGN_data =  compute_ggn_vp(map_state, REAL_DATA,  model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)
    GGN_ip   =  compute_ggn_vp(map_state, IP_DATA,   model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)
    W, WT    =  compute_W_vps( map_state, REAL_DATA,   model_type=model_type, flat_params=flat_params, unravel_fn=unravel_fn, full_set_size=None)

    beta      = full_set_size / IP_DATA.shape[0]
    beta_sqrt = jnp.sqrt(beta)

    
    theta_sampler = get_conditional_theta_sampler(IP_DATA, alpha, beta, map_state, atol=1e-6, btol=1e-6, ctol=1e-7)
    
    

    """##
    STATS
    ##"""
    num_samples = 1000
    alot_of_samples = theta_sampler(_key(434343), num_samples=num_samples)
    
    sample_mean = alot_of_samples.mean(axis=0, keepdims=True)
    centered = (alot_of_samples - sample_mean)
    sample_cov = centered.T @ centered / (num_samples - 1)
    sample_precision = jnp.linalg.inv(sample_cov)
    iso_prior = alpha * jnp.eye(D)
    precision = iso_prior + beta*GGN_ip_dense
    
    
    # GGN_ip_inv_dense = jnp.linalg.inv(precision)
    # true_trace = jnp.linalg.trace((iso_prior + beta*GGN_data_dense) @ (GGN_ip_inv_dense))
    
    # pdb.set_trace()
    

    # integrand = stochtrace.integrand_trace()
    # sampler = lambda __key: theta_sampler(__key, num_samples=1)
    # estimate = stochtrace.estimator(integrand, sampler)
    # est_trace = estimate(lambda v: alpha*v + beta*GGN_data(v), _key(88_88_88_88))

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
    diff = jnp.abs(true_matrix - sampled_matrix)# / (jnp.abs(true_matrix) + 1e-0)

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
    axs[2].set_title("|true - sample|")

    # fig.colorbar(im0)
    fig.colorbar(im1)
    fig.colorbar(im2)

    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    # plt.show()
    
    plt.savefig("precision.png")