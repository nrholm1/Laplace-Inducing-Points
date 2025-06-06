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

from src.scalemodels import TrainState, EMPTY_STATS
from src.toymodels import SimpleRegressor, SimpleClassifier
from src.toydata import get_dataloaders, load_toydata
from src.nplot import make_predictive_mean_figure, plot_binary_classification_data, plot_map_2D_classification, scatterp, linep, plot_cinterval, plot_inducing_points_1D, plot_lla_2D_classification

from src.train_map import train_map
from src.train_alpha import train_map_then_alpha
from src.train_inducing import train_inducing_points
from src.lla import materialize_covariance, posterior_lla_dense, predict_lla_dense, predict_lla_scalable
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
        plot_map_2D_classification(fig, ax, map_model_state, tmin, tmax)
        # plot_bc_boundary_contour( map_model_state, tmin, tmax, color='#3f3', alpha=1., label='Decision boundary')
        
    plt.legend(loc='lower right', framealpha=1.0)
    plt.tight_layout()
    os.makedirs("fig", exist_ok=True)
    model_type = f"{model_type}_" if model_type is not None else ""
    dataset_name = f"{dataset_name}_" if dataset_name is not None else ""
    plt.savefig(f"fig/{dataset_name}{model_type}map.pdf")


def plot_inducing_dense(model_type, map_model_state, 
                  Xtrain, 
                  ytrain, 
                  Xtest, 
                  ytest, 
                  zinduc, 
                  alpha, rng_inducing,
                  model, optimizer_map, 
                  m_induc, epochs_induc, dataset_name,
                  full_lla=False):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    if full_lla:
        fig.suptitle(f"Full LLA / {Xtrain.shape[0]} data points")
    else:
        fig.suptitle(f"IP LLA / {m_induc} inducing points, {epochs_induc} steps")
    

    if model_type == "regressor":  # 1D regression case
        # Create a linear grid for predictions
        xlin = jnp.linspace(Xtrain.min(), Xtrain.max(), 100, dtype=jnp.float64)[:, None]
        postpreddist_full = predict_lla_dense(
            map_model_state, xlin, Xtrain, model_type=model_type, alpha=alpha
        )
        postpreddist_optimized = predict_lla_dense(
            map_model_state, xlin, zinduc, model_type=model_type, alpha=alpha,
            full_set_size=Xtrain.shape[0]
        )
        
        plot_cinterval(xlin.squeeze(), postpreddist_full.mean(), postpreddist_full.stddev(), 
                       text="full", color='orange', zorder=5)
        plot_cinterval(xlin.squeeze(), postpreddist_optimized.mean(), postpreddist_optimized.stddev(), 
                       text="ind. optimized", color='limegreen', zorder=4)
        
        # Plot training and test data
        scatterp(Xtest, ytest, color="yellow", zorder=2, label='Test data')
        scatterp(Xtrain, ytrain, zorder=1, label='Train data')
        plot_inducing_points_1D(ax, Z, color='limegreen', offsetp=0.00, zorder=3)#, label=None)

        
    plt.tight_layout()
    os.makedirs("fig", exist_ok=True)
    plt.savefig(f"fig/{dataset_name}_{model_type}_lla.pdf")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="full_pipeline",
                        choices=["train_map", "train_inducing", "visualize", "full_pipeline"],
                        help="Which phase(s) to run.")
    parser.add_argument("--full", action="store_true",
                        help="If selected, compute full LLA.")
    parser.add_argument("--scalable", action="store_true",
                        help="Whether to use scalable (matrix free) IP optimization and LLA sampling.")
    parser.add_argument("--num_mc_samples_lla", type=int, default=1000,
                        help="Number of MC samples for LLA predictive dist.")
    parser.add_argument("--plot_Z", action="store_true",
                        help="Whether to plot inducing points.")
    parser.add_argument("--plot_X", action="store_true",
                        help="Whether to plot training points.")
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
    
    # Load model config
    cfg = load_yaml(args.model_config)
    model_cfg = cfg['model']
    model_type = model_cfg.get("name", "regressor")  # 'regressor' or 'classifier'
    num_h = model_cfg["num_h"]
    num_l = model_cfg["num_l"]
    num_c = model_cfg.get("num_c", 2) if model_type == "classifier" else 1
    model_seed = model_cfg["seed"]

    rng_model = jax.random.PRNGKey(model_seed)
    if model_type == "regressor":
        model = SimpleRegressor(numh=num_h, numl=num_l)
    elif model_type == "classifier":
        model = SimpleClassifier(numh=num_h, numl=num_l, numc=num_c)

    # Load optimization config (combined for MAP and inducing)
    # opt_cfg = load_yaml(args.optimization_config)
    opt_cfg = cfg['optimization']
    alpha = opt_cfg["alpha"]
    map_cfg = opt_cfg["map"]
    
    full_set_size = opt_cfg["full_set_size"]

    map_batch_size = map_cfg["batch_size"]
    epochs_map = map_cfg["epochs"]
    lr_map = map_cfg["lr"]
    seed_map = map_cfg["seed"]
    
    # Load data
    train_loader, test_loader = get_dataloaders(dataset=args.dataset, batch_size=map_batch_size)
    
    dummy_input = next(iter(train_loader))[0][:1]
    variables = model.init(rng_model, dummy_input)

    print("== Model Summary ==")
    print_summary(variables)

    ip_cfg = opt_cfg["ip"]
    m_inducing = ip_cfg["m"]
    epochs_inducing = ip_cfg["epochs"]
    inducing_batch_size = ip_cfg["batch_size"]
    lr_inducing = ip_cfg["lr"]
    mc_samples = ip_cfg["mc_samples"]
    seed_inducing = ip_cfg["seed"]
    st_samples      = ip_cfg.get("st_samples", 256)
    slq_samples     = ip_cfg.get("slq_samples", 4)
    slq_num_matvecs = ip_cfg.get("slq_num_matvecs", 32)

    # Build train_state for MAP
    optimizer_map = optax.adam(lr_map)
    model_state = TrainState.create(
        apply_fn=model.apply,
        params=variables,
        tx=optimizer_map,
        batch_stats = variables.get('batch_stats', EMPTY_STATS),
    )
    map_ckpt_prefix = f"map_{args.dataset}"

    # =========== PART A: MAP TRAINING ===========
    if args.mode in ["train_map", "full_pipeline"]:
        # map_model_state = train_map(
        #     model_state,
        #     train_loader,
        #     test_loader,
        #     model_type=model_type,
        #     alpha=alpha,
        #     num_epochs=epochs_map
        # )
        
        map_model_state, alpha = train_map_then_alpha(
            model_state,
            train_loader,
            test_loader,
            model_type=model_type,
            num_epochs=epochs_map,
            alpha0=alpha,
            alpha_lr=1e-2,
            alpha_every=5,
            burnin=100,
            full_set_size=full_set_size,
        )
        print(alpha)
        save_checkpoint(
            train_state=map_model_state,
            ckpt_dir=args.ckpt_map,
            prefix=map_ckpt_prefix,
            step=epochs_map
        )
        
        plot_map(map_model_state, 
                 *load_toydata(args.dataset), # get train/test data for plots
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

    # =========== PART B: Inducing Points ===========
    induc_ckpt_name = f"ind_{args.dataset}"
    rng_inducing = jax.random.PRNGKey(seed_inducing)
    train_loader_init, _ = get_dataloaders(dataset=args.dataset, batch_size=m_inducing)
    zinit = next(iter(train_loader_init))[0]
    zinit = jax.random.uniform(minval=-1, maxval=1, key=jax.random.PRNGKey(123), shape=zinit.shape)
    # zinit = jax.random.normal(key=jax.random.PRNGKey(123), shape=zinit.shape)
    train_loader_induc, _ = get_dataloaders(dataset=args.dataset, batch_size=inducing_batch_size)

    if args.mode in ["train_inducing", "full_pipeline"]:
        zoptimizer = optax.adam(lr_inducing)
        
        zinducing = train_inducing_points(
            map_model_state, 
            zinit, 
            zoptimizer,
            dataloader=train_loader_induc,
            rng=rng_inducing,
            model_type=model_type,
            num_mc_samples=mc_samples,
            alpha=alpha,
            num_steps=epochs_inducing,
            full_set_size=opt_cfg['full_set_size'],
            scalable=args.scalable,
            plot_type=args.dataset,
            st_samples=st_samples,
            slq_samples=slq_samples,
            slq_num_matvecs=slq_num_matvecs,
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
    if args.mode in ["visualize", "full_pipeline"]:
        os.makedirs("fig", exist_ok=True)
        
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        full_lla = args.full
        if full_lla:
            plt.title(f"Full LLA / {opt_cfg['full_set_size']} data points")
        else:
            plt.title(f"IP LLA / {m_inducing} inducing points, {epochs_inducing} steps")
        (xtrain,ytrain),_ = load_toydata(args.dataset)
        plot_lla_2D_classification(
            fig,
            ax,
            map_model_state,
            xtrain,
            ytrain,
            zinducing,
            alpha,
            mode="full_lla" if args.full else "ip_lla",
            matrix_free=args.scalable,
            num_mc_samples=args.num_mc_samples_lla,
            plot_Z=args.plot_Z,
            plot_X=args.plot_X,
        )
        # pdb.set_trace()
        plt.tight_layout()
        suffix_if_matrixfree = '_mf' if args.scalable else ''
        plt.savefig(f"fig/{args.dataset}_{model_type}_lla_{'full' if args.full else 'ip'}{suffix_if_matrixfree}.pdf")
        
        # ! LA vs LLA example plot!
        # make_predictive_mean_figure(map_model_state, xtrain, ytrain, alpha, num_mc_samples=args.num_mc_samples_lla)
        # plt.savefig(f"fig/la_vs_lla.pdf", dpi=300, bbox_inches="tight")
        
        print("[DONE] Visualization complete.")


if __name__ == "__main__":
    main()