import argparse
import os
import yaml

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state, checkpoints
import optax
jax.config.update("jax_enable_x64", True)
import tensorflow_probability.substrates.jax as tfp

from tqdm import tqdm
import matplotlib.pyplot as plt
from seaborn import set_style
set_style('darkgrid')

from toymodels import SimpleRegressor, SimpleClassifier, print_summary
from toydata import JAXDataset, get_dataloaders
from nplot import Colors, linep, scatterp, plot_cinterval, plot_inducing_points_1D


def nll_fun(state, params, batch):
    inp,res = batch
    pred = state.apply_fn(params, inp)
    mse_loss = optax.squared_error(pred,res).mean()
    return mse_loss


def log_prior_fun(params, stdev):
    """Simple L2 prior. Note: returns negative log prior!"""
    def sum_of_squares(t): return jnp.sum(t**2)
    param_leaves = jax.tree_util.tree_leaves(params)
    l2 = jnp.sum(jnp.array([sum_of_squares(p) for p in param_leaves]))
    return .5 / stdev * l2


def log_posterior_fun(state, params, batch, prior_std=1.0):
    nll = nll_fun(state, params, batch) # Gaussian likelihood => NLL ~ MSE
    log_prior = log_prior_fun(state.params, prior_std)
    return nll + log_prior


@jax.jit
def train_map_step(state, batch, prior_std=1.0):
    grad_fn = jax.value_and_grad(log_posterior_fun, argnums=1, has_aux=False)
    loss,grads = grad_fn(state, state.params, batch, prior_std)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, batch):
    loss = nll_fun(state, state.params, batch)
    return loss


def train_model(state, trainloader, testloader, train_step, *args, num_epochs=100, **kwargs):
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        for batch in trainloader:
            state, loss = train_step(state, batch, *args, **kwargs)
        if (epoch) % 10 == 0:
            eloss = 0
            for batch in testloader:
                eloss += eval_step(state, batch) # TODO make eval MAP step?
            pbar.set_description(f"Avg. eval MSE: {eloss/len(testloader):.3f}")
    return state



def compute_ggn(state, x, prior_std, full_set_size=None):
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(state.params)
    
    def per_datum_jacobian(xi):
        def scalar_output(flat_p):
            p = unravel_fn(flat_p)
            return state.apply_fn(p, xi[None]).squeeze()
        return jax.jacobian(scalar_output)(flat_params)

    # todo add hessian term for classification!
    Js = jax.vmap(per_datum_jacobian)(x)
    JtJ = jnp.einsum('ni,nj->ij', Js, Js)
    prior_precision = 1.0 / (prior_std**2)
    
    # rescaling weight if subsampled/inducing points
    N = x.shape[0]
    M = full_set_size or N
    GGN = N/M * JtJ + prior_precision*jnp.eye(JtJ.shape[0])
    
    return GGN, flat_params, unravel_fn
    

def ensure_symmetry(X, jitter=1e-8):
    return 0.5 * (X + X.T) + jitter * jnp.eye(X.shape[0]) # ! ensure symmetry of GGN (numerical stability)


def posterior_lla(map_state, x, prior_std, full_set_size=None, return_unravel_fn=False):
    """
    Posterior parameter distribution.
    """
    GGN, flat_params_map, unravel_fn = compute_ggn(map_state, x, prior_std, full_set_size=full_set_size) 
    GGN = ensure_symmetry(GGN)
    H_approx = jnp.linalg.inv(GGN)
    posterior_dist = tfp.distributions.MultivariateNormalFullCovariance(
            loc=flat_params_map.astype(jnp.float64), # todo: cast to f64 or other to f32?
            covariance_matrix=H_approx
        )
    if return_unravel_fn:
        return posterior_dist, unravel_fn
    return posterior_dist


def predict_lla(map_state, xnew, xtrain, prior_std, full_set_size=None):
    """
    Posterior predictive distribution.
    """
    GGN, flat_params_map, unravel_fn = compute_ggn(map_state, xtrain, prior_std, full_set_size=full_set_size) 
    GGN = ensure_symmetry(GGN)
    H_approx = jnp.linalg.inv(GGN)
    
    @jax.jit
    def flat_apply_fn(flat_p, inputs):
        p = unravel_fn(flat_p)
        return map_state.apply_fn(p, inputs)
    
    @jax.jit
    def per_datum_jacobian(xi):
        return jax.jacobian(lambda fp: flat_apply_fn(fp, xi[None]))(flat_params_map)
    Jnew = jax.vmap(per_datum_jacobian)(xnew)
    
    f_mean = flat_apply_fn(flat_params_map, xnew)
    
    @jax.jit
    def per_datum_cov(Ji):
        return Ji @ H_approx @ Ji.T
    f_cov = jax.vmap(per_datum_cov)(Jnew.squeeze(axis=1))
    
    assert jnp.all(jnp.linalg.eigvals(f_cov) > 0), "Covariance matrix not PD!"
    
    return tfp.distributions.MultivariateNormalDiag(loc=f_mean.squeeze(), 
                                                    scale_diag=jnp.sqrt(f_cov).squeeze())
    

def sample_params(mu, cov, rng):
    eps = jax.random.normal(rng, shape=mu.shape)
    # Cholesky to map eps ~ N(0,I) -> theta ~ N(mu, cov)
    L = jnp.linalg.cholesky(cov)
    return mu + L @ eps


def loglik_dataset(params, apply_fn, xdata, ydata, alpha):
    # ! for regression! closed form:
    ypred = apply_fn(params, xdata)
    twovar = 2/alpha
    N = xdata.shape[0]
    return -N/2 *jnp.log(twovar*jnp.pi) - 1/twovar * ((ydata-ypred)**2).sum()


def var_loglik_fun(q, dataset, apply_fn, unravel_fn, rng, alpha, num_mc_samples):
    """
    # ! using MC sample(s) of parameters
    """
    x,y = dataset
    mu,cov = q.mean(), q.covariance()
    log_sum = 0.0
    for i in range(num_mc_samples):
        rng_i = jax.random.fold_in(rng, i)  # make a fresh key
        theta_sample = sample_params(mu, cov, rng_i)
        theta_sample = unravel_fn(theta_sample)
        log_p_data = loglik_dataset(theta_sample, apply_fn, x, y, alpha) # todo don't use *all* data
        log_sum += log_p_data
    
    return log_sum / num_mc_samples
    
    
def var_kl_fun(q, alpha):
    mu,cov = q.mean(), q.covariance()
    D = cov.shape[0]
    tr_term = alpha*jnp.linalg.trace(cov)
    norm_term =alpha*jnp.linalg.norm(mu)**2
    logdetp_term = jnp.log(D/alpha) # log(det( I * alpha^(-1) ))
    logdetq_term = jnp.log(jnp.linalg.det(cov))
    
    kl_term = 0.5 * (tr_term - D + norm_term + logdetp_term - logdetq_term)
    return kl_term


def naive_objective(xind, dataset, state, alpha, rng, num_mc_samples, full_set_size=None, reg_coeff=0):
    q,unravel_fn = posterior_lla(state, xind, 
                      prior_std=alpha, full_set_size=full_set_size,
                      return_unravel_fn=True)

    loglik_term = var_loglik_fun(q, dataset, state.apply_fn, unravel_fn, rng, alpha, num_mc_samples=num_mc_samples)
    kl_term = var_kl_fun(q, alpha)
    reg_term = reg_coeff * jnp.sum(jnp.square(xind))
    return - (loglik_term-kl_term) + reg_term

variational_grad = jax.value_and_grad(naive_objective)

def optimize_step(x, dataset, map_model_state, alpha, 
                  opt_state, rng, xoptimizer,
                  num_mc_samples, full_set_size=None):
    loss, grads = variational_grad(x, dataset, map_model_state, alpha, rng, num_mc_samples=num_mc_samples, full_set_size=full_set_size)
    updates, new_opt_state = xoptimizer.update(grads, opt_state)
    new_x = optax.apply_updates(x, updates)
    return new_x, new_opt_state, loss

# ? JIT optimize_step here, since the static_argnames is problematic in the decorator?
optimize_step = jax.jit(optimize_step, static_argnames=['xoptimizer', 'num_mc_samples', 'full_set_size'])


def train_inducing_points(map_model_state, xinit, 
                          xoptimizer, full_dataset, rng,
                          num_mc_samples=10, alpha=1.0, num_steps=100):
    opt_state = xoptimizer.init(xinit)
    x = xinit
    lb, ub = full_dataset[0].min(), full_dataset[0].max() # data support range #! 1D case here!
    N = full_dataset[0].shape[0]
    
    pbar = tqdm(range(num_steps))
    for step in pbar:
        rng, rng_step = jax.random.split(rng)
        dataset_sample = full_dataset # todo: mini batch here?
        x,opt_state,loss = optimize_step(x, dataset_sample, map_model_state, alpha, 
                                         opt_state, rng_step, xoptimizer=xoptimizer,
                                         num_mc_samples=num_mc_samples, full_set_size=N)
        x = jnp.clip(x, lb, ub) # ! hard constraint enforcement
        
        if step == 0:
            print(f"Initial loss: {loss:.3f}")
        if step % 1 == 0:
            pbar.set_description_str(f"Loss: {loss:.3f}", refresh=True)
    return x


def save_array_checkpoint(array, ckpt_dir, name, step):
    """
    Save a JAX array (e.g. for inducing points) as a .npy file:
      ckpt_dir/name_step.npy
    """
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    filename = os.path.join(ckpt_dir, f"{name}_{step}.npy")
    np.save(filename, np.array(array))  # convert to NumPy
    print(f"[checkpoint] Saved array checkpoint at step {step} in {filename}")


def load_array_checkpoint(ckpt_dir, name, step):
    """
    Load a .npy file and return as a JAX array (device_put).
      ckpt_dir/name_step.npy
    """
    ckpt_dir = os.path.abspath(ckpt_dir)
    filename = os.path.join(ckpt_dir, f"{name}_{step}.npy")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} not found")
    array = np.load(filename)
    print(f"[checkpoint] Loaded array checkpoint from {filename}")
    return jax.device_put(array)


def save_checkpoint(train_state, ckpt_dir, prefix, step):
    """
    Save a Flax TrainState to ckpt_dir with given prefix, e.g.:
      ckpt_dir/prefix_<step>
    """
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir,
        target=train_state,
        step=step,
        prefix=prefix + "_",
        overwrite=True
    )
    print(f"[checkpoint] Saved model checkpoint at step {step} in {ckpt_dir} (prefix={prefix})")


def load_checkpoint(ckpt_dir, prefix, target=None):
    """
    Load a Flax TrainState from ckpt_dir with the given prefix.
    If multiple steps are present, loads the latest by default.
    """
    ckpt_dir = os.path.abspath(ckpt_dir)
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=target,
        prefix=prefix + "_",
    )
    print(f"[checkpoint] Loaded model checkpoint from {ckpt_dir} (prefix={prefix})")
    return restored_state


def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="full_pipeline",
                        choices=["train_map", "train_inducing", "visualize", "full_pipeline"],
                        help="Which phase(s) to run.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to an .npz file containing x,y arrays.")
    parser.add_argument("--model_config", type=str, required=True,
                        help="Path to a YAML file with model hyperparams (e.g. config/toyregressor.yml).")
    parser.add_argument("--optimization_config", type=str, required=True,
                        help="Path to a YAML file with all optimization hyperparams (for MAP and inducing).")
    parser.add_argument("--ckpt_map", type=str, default="checkpoint/map/",
                        help="Directory for loading/saving the MAP model checkpoint.")
    parser.add_argument("--ckpt_induc", type=str, default="checkpoint/ind/",
                        help="Directory for loading/saving the inducing points checkpoint.")
    args = parser.parse_args()

    # Print selected options
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)
        
    # 1) Load data
    datafile = f"data/{args.dataset}.npz"
    if not os.path.exists(datafile):
        raise FileNotFoundError(f"Data file not found: {datafile}")
    data_npz = np.load(datafile)
    x = jax.device_put(data_npz["x"])
    y = jax.device_put(data_npz["y"])
    n_samples = x.shape[0]
    print(f"[INFO] Loaded dataset from {datafile} with {n_samples} samples.")

    # 2) Create train/test split
    trainsplit = int(0.9 * n_samples)
    xtrain, ytrain = x[:trainsplit], y[:trainsplit]
    xtest,  ytest  = x[trainsplit:], y[trainsplit:]

    # 3) Load model config
    model_cfg = load_yaml(args.model_config)
    model_type = model_cfg.get("model_type", "regressor")  # 'regressor' or 'classifier'
    num_h = model_cfg["num_h"]
    num_l = model_cfg["num_l"]
    num_c = model_cfg.get("num_c", 2)
    model_seed = model_cfg["rng_seed"]

    if model_type == "regressor":
        model = SimpleRegressor(numh=num_h, numl=num_l)
    elif model_type == "classifier":
        model = SimpleClassifier(numh=num_h, numl=num_l, numc=num_c)

    rng_model = jax.random.PRNGKey(model_seed)
    dummy_inp = jax.random.normal(rng_model, shape=(num_h, 1))
    params = model.init(rng_model, dummy_inp)

    print("== Model Summary ==")
    print_summary(params)

    # 4) Load optimization config (combined for MAP and inducing)
    opt_cfg = load_yaml(args.optimization_config)
    map_cfg = opt_cfg["map"]
    induc_cfg = opt_cfg["inducing"]

    batch_size = map_cfg["batch_size"]
    epochs_map = map_cfg["epochs_map"]
    lr_map = map_cfg["lr_map"]
    alpha_map = map_cfg["alpha"]
    seed_map = map_cfg["seed"]

    m_induc = induc_cfg["m_induc"]
    epochs_induc = induc_cfg["epochs_induc"]
    lr_induc = induc_cfg["lr_induc"]
    alpha_induc = induc_cfg["alpha"]
    mc_samples = induc_cfg["mc_samples"]
    seed_inducing = induc_cfg["seed"]

    # 5) Build train_state for MAP
    optimizer_map = optax.adam(lr_map)
    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer_map
    )
    map_ckpt_prefix = f"map_{args.dataset}"

    # =========== PART A: MAP TRAINING ===========
    if args.mode in ["train_map", "full_pipeline"]:
        train_dataset = JAXDataset(xtrain, ytrain)
        test_dataset  = JAXDataset(xtest,  ytest)
        train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size)
        
        map_model_state = train_model(
            model_state,
            train_loader,
            test_loader,
            train_step=train_map_step,
            prior_std=alpha_map**(-0.5),
            num_epochs=epochs_map
        )
        save_checkpoint(
            train_state=map_model_state,
            ckpt_dir=args.ckpt_map,
            prefix=map_ckpt_prefix,
            step=epochs_map
        )
        if args.mode == "train_map":
            print("[DONE] MAP training only.")
            return
    else:
        map_model_state = load_checkpoint(
            ckpt_dir=args.ckpt_map,
            prefix=map_ckpt_prefix,
            target=model_state
        )

    # =========== PART B: Inducing Points ===========
    indu_ckpt_name = f"ind_{model_type}"
    rng_inducing = jax.random.PRNGKey(seed_inducing)
    # xinit = jax.random.uniform(rng_inducing, shape=(m_induc,)) * 7.0 - 4.0
    xinit = jnp.linspace(-4, 3, m_induc)

    if args.mode in ["train_inducing", "full_pipeline"]:
        xoptimizer = optax.adam(lr_induc)
        full_dataset = (xtrain, ytrain)
        
        xinduc = train_inducing_points(
            map_model_state, 
            xinit, 
            xoptimizer,
            full_dataset=full_dataset,
            rng=rng_inducing,
            num_mc_samples=mc_samples,
            alpha=alpha_induc,
            num_steps=epochs_induc
        )

        save_array_checkpoint(
            array=xinduc,
            ckpt_dir=args.ckpt_induc,
            name=indu_ckpt_name,
            step=epochs_induc
        )
        if args.mode == "train_inducing":
            print("[DONE] Inducing training only.")
            return
    else:
        xinduc = load_array_checkpoint(
            ckpt_dir=args.ckpt_induc,
            name=indu_ckpt_name,
            step=epochs_induc
        )

    # =========== PART C: Visualization ===========
    if args.mode in ["visualize", "full_pipeline"]:
        xlin = jnp.linspace(-4, 3, 100)[:, None]
        postpreddist_full = predict_lla(
            map_model_state, xlin, xtrain, prior_std=1.0
        )
        postpreddist_rand = predict_lla(
            map_model_state, xlin, xinit, prior_std=1.0, full_set_size=xtrain.shape[0]
        )
        postpreddist_optimized = predict_lla(
            map_model_state, xlin, xinduc, prior_std=1.0, full_set_size=xtrain.shape[0]
        )
    
        fig, ax = plt.subplots(figsize=(8,5))
        plot_cinterval(xlin.squeeze(), postpreddist_full.mean(), postpreddist_full.stddev(), 
                       text="full", color='orange', zorder=5)
        plot_cinterval(xlin.squeeze(), postpreddist_rand.mean(), postpreddist_rand.stddev(), 
                       text="ind. init", color='red', zorder=3)
        plot_cinterval(xlin.squeeze(), postpreddist_optimized.mean(), postpreddist_optimized.stddev(), 
                       text="ind. optimized", color='green', zorder=4)
        scatterp(xtest, ytest, color="yellow", zorder=2, label='Test data')
        plot_inducing_points_1D(ax, xinduc, color='green', offsetp=0.00, zorder=3, label=None)
        plot_inducing_points_1D(ax, xinit, color='red', offsetp=0.00, zorder=3, label=None)
        plt.legend(loc='lower right')
        os.makedirs("fig", exist_ok=True)
        plt.savefig("fig/pdist-full_vs_init_vs_optimized.pdf")
        print("[DONE] Visualization complete.")

if __name__ == "__main__":
    main()


"""#!SAVING THIS CODE BC UNIMPLEMENTED PLOT"""
# m = 16
# subsample_key = jax.random.PRNGKey(seed=314159265)
# #%%
# xlin = jnp.linspace(-4.5,3.5,100)[:,None]
# ylin_map = model.apply(map_model_state.params, xlin)

# fig, ax = plt.subplots(figsize=(8,5))

# # xinit = jnp.linspace(xtest.min(), xtest.max(), num=m)
# # xinit = jax.random.normal(jax.random.PRNGKey(4), shape=(m,))* 2
# xinit = jax.random.uniform(jax.random.PRNGKey(4), shape=(m,))*7 - 4
# perm = jax.random.permutation(subsample_key, xtrain.shape[0])
# subsample = perm[:m]
# xtrain_subs = xinit
# predictive_dist_subsampled = predict_lla(map_model_state, xlin, 
#                                          xtrain_subs,
#                                          prior_std=1.0, full_set_size=xtrain.shape[0])
# predictive_dist =            predict_lla(map_model_state, xlin, 
#                                          xtrain,
#                                          prior_std=1.0)
# plot_cinterval(xlin.squeeze(), predictive_dist.mean(), predictive_dist.stddev(), 
#                text="full", color='orange',zorder=4)
# plot_cinterval(xlin.squeeze(), predictive_dist_subsampled.mean(), predictive_dist_subsampled.stddev(), 
#                text="subsampled", color='red', zorder=3)
# scatterp(xtest,ytest,color=Colors.yellow, zorder=2, label='Test data')
# # scatterp(xtrain_subs, ytrain_subs, color='red', zorder=4, label='Inducing data points')
# plot_inducing_points_1D(ax, xtrain_subs, offsetp=-0.05, zorder=3, label=None) # ! plot last to ensure correct ylim!
# plt.legend(loc='lower right')
# plt.show()