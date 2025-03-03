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

from toymodels import SimpleRegressor, SimpleClassifier, print_summary
from toydata import JAXDataset, get_dataloaders
from nplot import Colors, linep, scatterp, plot_cinterval, plot_inducing_points_1D

from train_map import train_map, regression_map_step
from train_inducing import train_inducing_points
from lla import predict_lla
from utils import load_yaml, save_checkpoint, load_checkpoint, save_array_checkpoint, load_array_checkpoint
    

jax.config.update("jax_enable_x64", True)


def plot_map(map_model_state, traindata, testdata, alpha_map):
    # ? visualize MAP estimator
    xtrain, ytrain = traindata
    xtest, ytest = testdata
    fig, ax = plt.subplots(figsize=(8,5))
    xlin = jnp.linspace(xtrain.min(), xtrain.max(), 100, dtype=jnp.float64)[:, None]
    postpreddist_full = predict_lla(
        map_model_state, xlin, xtrain, ytrain, prior_std=alpha_map**(-0.5)
    )
    plot_cinterval(xlin.squeeze(), postpreddist_full.mean(), postpreddist_full.stddev(), 
                    text="full", color='orange', zorder=5)
    scatterp(xtest, ytest, color="yellow", zorder=2, label='Test data')
    scatterp(xtrain, ytrain, zorder=1, label='Train data')
    plt.legend(loc='lower right')
    os.makedirs("fig", exist_ok=True)
    plt.savefig("fig/map_estimator.pdf")


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
    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
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
            train_step=regression_map_step,
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
        plot_map(map_model_state, (xtrain, ytrain), (xtest, ytest), alpha_map)
        
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
    # xinit = xtrain
    # xinit = jnp.linspace(xtrain.min(), xtrain.max(), m_induc)[:,None]
    _, test_loader = get_dataloaders(train_dataset, test_dataset, min(m_induc,len(test_dataset)))
    xinit = next(iter(test_loader))[0]

    if args.mode in ["train_inducing", "full_pipeline"]:
        xoptimizer = optax.adam(lr_induc)
        # print("WARNING! Training inducing with full dataset!")
        # train_loader, _ = get_dataloaders(train_dataset, test_dataset, len(train_dataset))
        
        xinduc = train_inducing_points(
            map_model_state, 
            xinit, 
            xoptimizer,
            dataloader=train_loader,
            rng=rng_inducing,
            num_mc_samples=mc_samples,
            alpha=alpha_induc,
            num_steps=epochs_induc
        )

        save_array_checkpoint(
            array=xinduc,
            ckpt_dir=args.ckpt_induc,
            name=induc_ckpt_name,
            step=epochs_induc
        )
        if args.mode == "train_inducing":
            print("[DONE] Inducing training only.")
            # return # ! don't return, visualize
    else:
        xinduc = load_array_checkpoint(
            ckpt_dir=args.ckpt_induc,
            name=induc_ckpt_name,
            step=epochs_induc
        )

    # =========== PART C: Visualization ===========
    if args.mode in ["visualize", "train_inducing", "full_pipeline"]:
        xlin = jnp.linspace(xtrain.min(), xtrain.max(), 100, dtype=jnp.float64)[:, None]
        prior_std = alpha_map**(-0.5) # todo verify this?
        postpreddist_full = predict_lla(
            map_model_state, xlin, xtrain, ytrain, prior_std=prior_std
        )
        postpreddist_rand = predict_lla(
            map_model_state, xlin, xinit, prior_std=prior_std, full_set_size=xtrain.shape[0]
        )
        postpreddist_optimized = predict_lla(
            map_model_state, xlin, xinduc, prior_std=prior_std, full_set_size=xtrain.shape[0]
        )
    
        fig, ax = plt.subplots(figsize=(8,5))
        plot_cinterval(xlin.squeeze(), postpreddist_full.mean(), postpreddist_full.stddev(), 
                       text="full", color='orange', zorder=5)
        plot_cinterval(xlin.squeeze(), postpreddist_rand.mean(), postpreddist_rand.stddev(), 
                       text="ind. init", color='red', zorder=3)
        plot_cinterval(xlin.squeeze(), postpreddist_optimized.mean(), postpreddist_optimized.stddev(), 
                       text="ind. optimized", color='green', zorder=4)
        scatterp(xtest, ytest, color="yellow", zorder=2, label='Test data')
        scatterp(xtrain, ytrain, zorder=1, label='Train data')
        plot_inducing_points_1D(ax, xinduc, color='green', offsetp=0.00, zorder=3, label=None)
        plot_inducing_points_1D(ax, xinit, color='red', offsetp=0.00, zorder=3, label=None)
        plt.legend(loc='lower right')
        os.makedirs("fig", exist_ok=True)
        plt.savefig("fig/pdist-full_vs_init_vs_optimized.pdf")
        print("[DONE] Visualization complete.")

if __name__ == "__main__":
    main()