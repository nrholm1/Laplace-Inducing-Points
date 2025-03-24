import argparse
import os
import pdb

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
import optax

import matplotlib.pyplot as plt
from seaborn import set_style
set_style('darkgrid')

from src.toymodels import SimpleRegressor, SimpleClassifier
from src.toydata import JAXDataset, get_dataloaders, plot_binary_classification_data
from src.nplot import plot_bc_boundary_contour, plot_bc_heatmap, scatterp, linep, plot_cinterval, plot_inducing_points_1D

from src.train_map import train_map
from src.train_inducing import train_inducing_points
from src.lla import posterior_lla_dense, predict_lla_dense
from src.utils import load_yaml, save_checkpoint, load_checkpoint, save_array_checkpoint, load_array_checkpoint, print_summary, print_options
    
# jax.config.update("jax_enable_x64", True)


def plot_map(map_model_state, traindata, testdata, alpha_map, model_type="", dataset_name=""):
    # ? visualize MAP estimator
    fig, ax = plt.subplots(figsize=(8,5))
    plt.title(f"MAP estimator")
    
    xtrain, ytrain = traindata
    xtest, ytest = testdata
    
    if model_type == "regressor":
        xlin = jnp.linspace(xtrain.min(), xtrain.max(), 100, dtype=jnp.float64)[:, None]
        postpreddist_full = predict_lla_dense(
            map_model_state, xlin, xtrain, jnp.array(1.), model_type="regressor", prior_std=alpha_map**(-0.5)
        )
        plot_cinterval(xlin.squeeze(), postpreddist_full.mean(), postpreddist_full.stddev(), 
                        text="full", color='orange', zorder=5)
        scatterp(xtest, ytest, color="yellow", zorder=2, label='Test data')
        scatterp(xtrain, ytrain, zorder=1, label='Train data')
        
    elif model_type == "classifier":
        from src.toydata import plot_binary_classification_data
        plot_binary_classification_data(xtrain, ytrain)
        plot_bc_heatmap(fig, ax, map_model_state, xtrain.min(), xtrain.max())
        plot_bc_boundary_contour( map_model_state, xtrain.min(), xtrain.max(), color='#3f3', alpha=1., label='Decision boundary')
        
    plt.legend(loc='lower right', framealpha=1.0)
    plt.tight_layout()
    os.makedirs("fig", exist_ok=True)
    model_type = f"{model_type}_" if model_type is not None else ""
    dataset_name = f"{dataset_name}_" if dataset_name is not None else ""
    plt.savefig(f"fig/{dataset_name}{model_type}map.pdf")


def plot_inducing(model_type, map_model_state, 
                  xtrain, ytrain, 
                  xtest, ytest,
                  zinduc, 
                  prior_std, rng_inducing,
                  model, optimizer_map, 
                  m_induc, epochs_induc, dataset_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.title(f"Induced LLA / {m_induc} inducing points, {epochs_induc} steps")
    # plt.title(f"Full LLA / {xtrain.shape[0]} data points")
    
    rng_theta_sample = jax.random.fold_in(rng_inducing, 0)

    if model_type == "regressor":  # 1D regression case
        # Create a linear grid for predictions
        xlin = jnp.linspace(xtrain.min(), xtrain.max(), 100, dtype=jnp.float64)[:, None]
        postpreddist_full = predict_lla_dense(
            map_model_state, xlin, xtrain, model_type=model_type, prior_std=prior_std
        )
        postpreddist_optimized = predict_lla_dense(
            map_model_state, xlin, zinduc, model_type=model_type, prior_std=prior_std,
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
        postdist, unravel_fn = posterior_lla_dense(
            map_model_state, zinduc, model_type=model_type, prior_std=prior_std,
            full_set_size=xtrain.shape[0], return_unravel_fn=True
        )
        
        # Plot the inducing points
        # plot_binary_classification_data(xtrain, ytrain)
        scatterp(*zinduc.T, color="yellow", zorder=8, marker="X", label='Inducing points')

        rng_theta_sample = jax.random.fold_in(rng_inducing, 0)
        # Plot multiple boundary contours sampled from the posterior
        for i in range(10):
            rng_theta_sample = jax.random.fold_in(rng_theta_sample, i)
            theta_sample = postdist.sample(seed=rng_theta_sample)
            sampled_model_state = train_state.TrainState.create(
                apply_fn=model.apply,
                params=unravel_fn(theta_sample),
                tx=optimizer_map
            )
            label = "Decision boundary samples" if i==0 else None
            plot_bc_boundary_contour(sampled_model_state, xtrain.min(), xtrain.max(),
                                       color='yellow', alpha=0.5, zorder=6, label=label)
    
        plot_bc_heatmap(fig, ax, map_model_state, xtrain.min(), xtrain.max())

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

    dummy_inp = jax.random.normal(rng_model, shape=(num_h, num_c))
    variables = model.init(rng_model, dummy_inp)

    print("== Model Summary ==")
    print_summary(variables)

    # Load optimization config (combined for MAP and inducing)
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
            prior_std=alpha_map**(-0.5),
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
                 alpha_map, 
                 model_type=model_type, 
                 dataset_name=args.dataset)
        
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
    induc_ckpt_name = f"ind_{args.dataset}"
    rng_inducing = jax.random.PRNGKey(seed_inducing)
    m_induc = min(m_induc, len(test_dataset))
    _, test_loader = get_dataloaders(train_dataset, test_dataset, m_induc)
    zinit = next(iter(test_loader))[0]

    if args.mode in ["train_inducing", "full_pipeline"]:
        zoptimizer = optax.adam(lr_induc)
        
        zinduc = train_inducing_points(
            map_model_state, 
            zinit, 
            zoptimizer,
            dataloader=train_loader,
            rng=rng_inducing,
            model_type=model_type,
            num_mc_samples=mc_samples,
            alpha=alpha_induc,
            num_steps=epochs_induc,
            full_set_size=xtrain.shape[0],
        )

        # Save the inducing points (zinduc)
        save_array_checkpoint(
            array=zinduc,
            ckpt_dir=args.ckpt_induc,
            name=induc_ckpt_name,
            step=epochs_induc
        )
    
        if args.mode == "train_inducing":
            print("[DONE] Inducing training only.")
            # return # ! don't return, visualize
    else:
        # Load both the inducing points (zinduc)
        zinduc = load_array_checkpoint(
            ckpt_dir=args.ckpt_induc,
            name=induc_ckpt_name,
            step=epochs_induc
        )

    # =========== PART C: Visualization ===========
    if args.mode in ["visualize", "train_inducing", "full_pipeline"]:
        prior_std = alpha_map**(-0.5) # todo verify this?
        plot_inducing(model_type, map_model_state, 
                      xtrain, ytrain, 
                      xtest, ytest,
                      zinduc, 
                      prior_std, rng_inducing,
                      model, optimizer_map, 
                      m_induc, epochs_induc, 
                      dataset_name=args.dataset)
        print("[DONE] Visualization complete.")


if __name__ == "__main__":
    main()