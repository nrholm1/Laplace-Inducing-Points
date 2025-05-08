import argparse
import os
import pdb

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from flax.training import train_state
import optax

import matplotlib.pyplot as plt
from seaborn import set_style
set_style('darkgrid')

from src.toymodels import SimpleRegressor, SimpleClassifier
from src.toydata import JAXDataset, get_dataloaders, plot_binary_classification_data
from src.nplot import plot_bc_boundary_contour, plot_bc_heatmap, plot_heatmap_averaged, scatterp, linep, plot_cinterval, plot_inducing_points_1D

from src.train_map import train_map
from src.train_inducing import train_inducing_points
from src.lla import materialize_covariance, posterior_lla_dense, predict_lla_dense, predict_lla_fun
from src.sample import sample
from src.utils import flatten_nn_params, load_yaml, save_checkpoint, load_checkpoint, save_array_checkpoint, load_array_checkpoint, print_summary, print_options
    
# jax.config.update("jax_enable_x64", True)


def plot_map(map_model_state, traindata, testdata, alpha, model_type="", dataset_name=""):
    # ? visualize MAP estimator
    fig, ax = plt.subplots(figsize=(8,5))
    plt.title(f"MAP estimator")
    
    xtrain, ytrain = traindata
    xtest, ytest = testdata
    
    if model_type == "regressor":
        xlin = jnp.linspace(xtrain.min(), xtrain.max(), 100, dtype=jnp.float64)[:, None]
        postpreddist_full = predict_lla_dense(
            map_model_state, xlin, xtrain, model_type="regressor", alpha=alpha
        )
        plot_cinterval(xlin.squeeze(), postpreddist_full.mean(), postpreddist_full.stddev(), 
                        text="full", color='orange', zorder=5)
        scatterp(xtest, ytest, color="yellow", zorder=2, label='Test data')
        scatterp(xtrain, ytrain, zorder=1, label='Train data')
        
    elif model_type == "classifier":
        from src.toydata import plot_binary_classification_data
        plot_binary_classification_data(xtrain, ytrain)
        tmin, tmax = xtrain.min() - 1.5, xtrain.max() + 1.5
        plot_bc_heatmap(fig, ax, map_model_state, tmin, tmax)
        # plot_bc_boundary_contour( map_model_state, tmin, tmax, color='#3f3', alpha=1., label='Decision boundary')
        
    plt.legend(loc='lower right', framealpha=1.0)
    plt.tight_layout()
    os.makedirs("fig", exist_ok=True)
    model_type = f"{model_type}_" if model_type is not None else ""
    dataset_name = f"{dataset_name}_" if dataset_name is not None else ""
    plt.savefig(f"fig/{dataset_name}{model_type}map.pdf")


def plot_inducing_dense(model_type, map_model_state, 
                  xtrain, ytrain, 
                  xtest, ytest,
                  zinduc, 
                  alpha, rng_inducing,
                  model, optimizer_map, 
                  m_induc, epochs_induc, dataset_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    # plt.title(f"Induced LLA / {m_induc} inducing points, {epochs_induc} steps")
    plt.title(f"Full LLA / {xtrain.shape[0]} data points")
    
    rng_theta_sample = jax.random.fold_in(rng_inducing, 0)

    if model_type == "regressor":  # 1D regression case
        # Create a linear grid for predictions
        xlin = jnp.linspace(xtrain.min(), xtrain.max(), 100, dtype=jnp.float64)[:, None]
        postpreddist_full = predict_lla_dense(
            map_model_state, xlin, xtrain, model_type=model_type, alpha=alpha
        )
        postpreddist_optimized = predict_lla_dense(
            map_model_state, xlin, zinduc, model_type=model_type, alpha=alpha,
            full_set_size=xtrain.shape[0]
        )
        
        plot_cinterval(xlin.squeeze(), postpreddist_full.mean(), postpreddist_full.stddev(), 
                       text="full", color='orange', zorder=5)
        plot_cinterval(xlin.squeeze(), postpreddist_optimized.mean(), postpreddist_optimized.stddev(), 
                       text="ind. optimized", color='limegreen', zorder=4)
        
        # Plot training and test data
        scatterp(xtest, ytest, color="yellow", zorder=2, label='Test data')
        scatterp(xtrain, ytrain, zorder=1, label='Train data')
        plot_inducing_points_1D(ax, zinduc, color='limegreen', offsetp=0.00, zorder=3)#, label=None)

    elif model_type == "classifier":  # 2D classification case
        # Plot the inducing points
        plot_binary_classification_data(xtrain, ytrain)
        
        
        # tmin, tmax = xtrain.min() - 1.5, xtrain.max() + 1.5
        tmin, tmax = xtrain.min() - 1.0, xtrain.max() + 1.0
        t = jnp.linspace(tmin, tmax, 100)
        X, Y = jnp.meshgrid(t, t, indexing="ij")
        pts = jnp.stack([X.ravel(), Y.ravel()], axis=-1)  # (num_pts**2, 2)
        
        logit_dist = predict_lla_dense(map_model_state,          # MAP state!
                               pts,
                            #    xtrain,                   # curvature set
                               zinduc,                   # curvature set
                               model_type="classifier",
                               alpha=alpha,
                               full_set_size=xtrain.shape[0])

        # 3) Monte-Carlo expectation of soft-max --------------------------------------
        num_mc = 2_000                                              # 30-50 is enough
        key    = jax.random.PRNGKey(0)
        logit_samples = logit_dist.sample(seed=key,
                                        sample_shape=(num_mc,))   # (M, N, K)
        probs = jax.nn.softmax(logit_samples, axis=-1)              # (M, N, K)
        mean_probs = probs.mean(axis=0)[:, 0]                       # P(class 1)

        Z = mean_probs.reshape(X.shape)

        # 4) plot ---------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
                "bwr", ["#111188", "white", "#881111"])
        cf = ax.contourf(X, Y, Z, levels=50, cmap=cmap, vmin=0., vmax=1.)
        fig.colorbar(cf, ax=ax, label=r"$E[y^*|x^*,D]$")

        plot_binary_classification_data(xtrain, ytrain)
        scatterp(*zinduc.T, color="yellow", zorder=8, marker="X", label="Inducing points")
        plt.title("GLM predictive mean")
        plt.legend(loc="lower right", framealpha=1.0)
        plt.tight_layout()

    
        # plot_heatmap_averaged(fig, ax, states, tmin, tmax, show_variance=False, opacity=1.0, cbarlabel=r"$E[y^*|x^*,D]$")

    # Adjust the legend to appear on top of the data points.
    leg = plt.legend(loc='lower right', framealpha=1.0)
    leg.set_zorder(10)
    plt.tight_layout()
    os.makedirs("fig", exist_ok=True)
    plt.savefig(f"fig/{dataset_name}_{model_type}_lla.pdf")


def plot_inducing_scalable(model_type, map_model_state, 
                  xtrain, ytrain, 
                  xtest, ytest,
                  zinduc, 
                  prior_precision, rng_inducing,
                  model, optimizer_map, 
                  m_induc, epochs_induc, dataset_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.title(f"Induced LLA / {m_induc} inducing points, {epochs_induc} steps")
    # plt.title(f"Full LLA / {xtrain.shape[0]} data points")
    
    rng_theta_sample = jax.random.fold_in(rng_inducing, 0)

    if model_type == "regressor":  # 1D regression case
        # Create a linear grid for predictions
        xlin = jnp.linspace(xtrain.min(), xtrain.max(), 100, dtype=jnp.float64)[:, None]
        fmu_full, fcov_vp_full = predict_lla_fun(
            map_model_state, xlin, xtrain, model_type=model_type, alpha=prior_precision
        )
        fmu_opt, fcov_vp_opt = predict_lla_fun(
            map_model_state, xlin, zinduc, model_type=model_type, alpha=prior_precision,
            full_set_size=xtrain.shape[0]
        )
        fcov_full = materialize_covariance(fcov_vp_full, *fmu_full.shape, mode='diag').squeeze()
        fcov_opt = materialize_covariance(fcov_vp_opt, *fmu_full.shape, mode='diag').squeeze()
        
        plot_cinterval(xlin.squeeze(), fmu_full.squeeze(), jnp.sqrt(fcov_full), 
                       text="full", color='orange', zorder=5)
        plot_cinterval(xlin.squeeze(), fmu_opt.squeeze(), jnp.sqrt(fcov_opt), 
                       text="ind. optimized", color='limegreen', zorder=4)
        
        # Plot training and test data
        scatterp(xtest, ytest, color="yellow", zorder=2, label='Test data')
        scatterp(xtrain, ytrain, zorder=1, label='Train data')
        plot_inducing_points_1D(ax, zinduc, color='limegreen', offsetp=0.00, zorder=3)#, label=None)

    elif model_type == "classifier":  # 2D classification case
        # Plot the inducing points
        plot_binary_classification_data(xtrain, ytrain)
        scatterp(*zinduc.T, color="yellow", zorder=8, marker="X", label='Inducing points')

        num_samples = 500
        rng_theta_sample = jax.random.fold_in(rng_inducing, 123)
        flat_params, unravel_fn = flatten_nn_params(map_model_state.params['params'])
        D = flat_params.shape[0]
        # theta_samples = sample(map_model_state, zinduc, D, 
        theta_samples = sample(map_model_state, xtrain, D, 
                               alpha=prior_precision, 
                               key=rng_theta_sample, 
                               model_type=model_type, 
                               num_samples=num_samples,
                               full_set_size=xtrain.shape[0],
                               num_proj_steps=50
                               )
        
        # Plot multiple boundary contours sampled from the posterior
        tmin, tmax = xtrain.min() - 1.5, xtrain.max() + 1.5
        states = []
        for i,theta_sample in enumerate(theta_samples):
            sampled_model_state = train_state.TrainState.create(
                apply_fn=model.apply,
                params=unravel_fn(theta_sample),
                tx=optimizer_map
            )
            states.append(sampled_model_state)
            # if i < 10:
            #     label = "Decision boundary samples" if i==0 else None
            #     plot_bc_boundary_contour(sampled_model_state, tmin, tmax,
            #                             color='yellow', alpha=0.5, zorder=6, label=label)
    
        plot_heatmap_averaged(fig, ax, states, tmin, tmax, cbarlabel=r"$E[y^*|x^*,D]$")
        # plot_bc_heatmap(fig, ax, map_model_state, tmin, tmax)

    # Adjust the legend to appear on top of the data points.
    leg = plt.legend(loc='lower right', framealpha=1.0)
    leg.set_zorder(10)
    plt.tight_layout()
    os.makedirs("fig", exist_ok=True)
    plt.savefig(f"fig/{dataset_name}_{model_type}_lla.pdf")



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
    print_options(args)
        
    # Load data
    datafile = f"data/{args.dataset}.npz"
    if not os.path.exists(datafile):
        raise FileNotFoundError(f"Data file not found: {datafile}")
    data_npz = np.load(datafile)
    x = jax.device_put(data_npz["x"])
    y = jax.device_put(data_npz["y"])
    n_samples = x.shape[0]
    print(f"[INFO] Loaded dataset from {datafile} with {n_samples} samples.")

    # Create train/test split
    trainsplit = int(0.9 * n_samples)
    xtrain, ytrain = x[:trainsplit], y[:trainsplit]
    xtest,  ytest  = x[trainsplit:], y[trainsplit:]

    # Load model config
    model_cfg = load_yaml(args.model_config)
    model_type = model_cfg.get("model_type", "regressor")  # 'regressor' or 'classifier'
    num_h = model_cfg["num_h"]
    num_l = model_cfg["num_l"]
    num_c = model_cfg.get("num_c", 2) if model_type == "classifier" else 1
    model_seed = model_cfg["rng_seed"]

    rng_model = jax.random.PRNGKey(model_seed)
    if model_type == "regressor":
        # rng_model = {'params': rng_model, 'logvar': zeros_rng}
        model = SimpleRegressor(numh=num_h, numl=num_l)
    elif model_type == "classifier":
        model = SimpleClassifier(numh=num_h, numl=num_l, numc=num_c)

    dummy_inp = jnp.ones((1, *xtrain[0].shape))
    variables = model.init(rng_model, dummy_inp)

    print("== Model Summary ==")
    print_summary(variables)

    # Load optimization config (combined for MAP and inducing)
    opt_cfg = load_yaml(args.optimization_config)
    alpha = opt_cfg["alpha"]
    map_cfg = opt_cfg["map"]
    inducing_cfg = opt_cfg["inducing"]

    batch_size = map_cfg["batch_size"]
    epochs_map = map_cfg["epochs_map"]
    lr_map = map_cfg["lr_map"]
    seed_map = map_cfg["seed"]

    m_inducing = inducing_cfg["m_induc"]
    epochs_inducing = inducing_cfg["epochs_induc"]
    lr_inducing = inducing_cfg["lr_induc"]
    mc_samples = inducing_cfg["mc_samples"]
    seed_inducing = inducing_cfg["seed"]

    # Build train_state for MAP
    optimizer_map = optax.adam(lr_map)
    # optimizer_map = optax.sgd(lr_map)
    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables,
        tx=optimizer_map
    )
    map_ckpt_prefix = f"map_{args.dataset}"

    train_dataset = JAXDataset(xtrain, ytrain)
    test_dataset  = JAXDataset(xtest,  ytest)
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size)
    # =========== PART A: MAP TRAINING ===========
    if args.mode in ["train_map", "full_pipeline"]:
        map_model_state = train_map(
            model_state,
            train_loader,
            test_loader,
            model_type=model_type,
            prior_precision=alpha,
            num_epochs=epochs_map
        )
        save_checkpoint(
            train_state=map_model_state,
            ckpt_dir=args.ckpt_map,
            prefix=map_ckpt_prefix,
            step=epochs_map
        )
        
        # ? visualize MAP estimator
        plot_map(map_model_state, 
                 (xtrain, ytrain), 
                 (xtest, ytest), 
                 alpha, 
                 model_type=model_type, 
                 dataset_name=args.dataset)
        
        print("[DONE] MAP training.")
        if args.mode == "train_map":
            return
    else:
        map_model_state = load_checkpoint(
            ckpt_dir=args.ckpt_map,
            prefix=map_ckpt_prefix,
            target=model_state
        )
        
    # pdb.set_trace()

    # =========== PART B: Inducing Points ===========
    induc_ckpt_name = f"ind_{args.dataset}"
    rng_inducing = jax.random.PRNGKey(seed_inducing)
    # m_inducing = min(m_inducing, len(test_dataset))
    train_loader_induc, test_loader = get_dataloaders(train_dataset, test_dataset, m_inducing)
    # zinit = next(iter(test_loader))[0]
    zinit = next(iter(train_loader_induc))[0]

    if args.mode in ["train_inducing", "full_pipeline"]:
        zoptimizer = optax.adam(lr_inducing)
        
        # with jax.profiler.trace("trace"):
        zinducing = train_inducing_points(
            map_model_state, 
            zinit, 
            zoptimizer,
            dataloader=train_loader,
            rng=rng_inducing,
            model_type=model_type,
            num_mc_samples=mc_samples,
            alpha=alpha,
            num_steps=epochs_inducing,
            full_set_size=xtrain.shape[0],
        )

        # Save the inducing points (zinduc)
        save_array_checkpoint(
            array=zinducing,
            ckpt_dir=args.ckpt_induc,
            name=induc_ckpt_name,
            step=epochs_inducing
        )
    
        print("[DONE] Inducing training.")
    else:
        # Load both the inducing points (zinduc)
        zinducing = load_array_checkpoint(
            ckpt_dir=args.ckpt_induc,
            name=induc_ckpt_name,
            step=epochs_inducing
        )

    # =========== PART C: Visualization ===========
    if args.mode in ["visualize", "train_inducing", "full_pipeline"]:
        plot_inducing_dense(model_type, map_model_state, 
        # plot_inducing_scalable(model_type, map_model_state, 
                      xtrain, ytrain, 
                      xtest, ytest,
                      zinducing, 
                      alpha, 
                      rng_inducing,
                      model, 
                      optimizer_map, 
                      m_inducing, 
                      epochs_inducing, 
                      dataset_name=args.dataset)
        print("[DONE] Visualization complete.")


if __name__ == "__main__":
    main()