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

from src.scaledata import get_dataloaders
from src.scalemodels import EMPTY_STATS, TrainState, get_model

from src.train_map import train_map
from src.grid_search import grid_search_alpha
from src.train_inducing import train_inducing_points
from src.utils import flatten_nn_params, load_yaml, save_checkpoint, load_checkpoint, save_array_checkpoint, load_array_checkpoint, print_summary, print_options
from src.train_alpha import train_map_then_alpha

# jax.config.update("jax_transfer_guard", "log")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="full_pipeline",
                        choices=["train_map", "train_inducing", "full_pipeline"],
                        help="Which phase(s) to run.")
    parser.add_argument("--continue", action="store_true",
                        help="Continue training from checkpoint") # todo !!! (might not be needed)
    parser.add_argument("--alpha_ip", type=float, default=None,
                        help="IP alpha to use - default is to grid search for it.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to a YAML file with model and optimization hyperparams.")
    parser.add_argument("--ckpt_map", type=str, default="checkpoint/map/",
                        help="Directory for loading/saving the MAP model checkpoint.")
    parser.add_argument("--ckpt_induc", type=str, default="checkpoint/ind/",
                        help="Directory for loading/saving the inducing points checkpoint.")
    args = parser.parse_args()

    # Print selected options
    print_options(args)

    # Get configs
    cfg = load_yaml(args.config)
    model_cfg = cfg['model']
    opt_cfg   = cfg['optimization']

    alpha         = opt_cfg["alpha"]
    full_set_size = opt_cfg['full_set_size'] # full dataset size

    map_cfg        = opt_cfg["map"]
    map_batch_size = map_cfg["batch_size"]
    epochs_map     = map_cfg["epochs"]
    lr_map         = map_cfg["lr"]
    seed_map       = map_cfg["seed"]

    # Initialize dataloaders
    train_loader, test_loader, val_loader = get_dataloaders(args.dataset, map_batch_size, num_workers=0)
    
    # Initialize model
    dummy_inp = next(iter(train_loader))[0][:1] # shape (1,28,28,1)
    model_type = model_cfg['type']
    model_seed = model_cfg['seed']
    rng_model = jax.random.PRNGKey(model_seed)
    model = get_model(model_cfg)
    variables = model.init(rng_model, dummy_inp, train=True)
    
    # Build train_state for MAP
    optimizer_map = optax.adam(lr_map)
    # model_state = train_state.TrainState.create(
    model_state = TrainState.create(
        apply_fn=model.apply,
        params=variables,
        tx=optimizer_map,
        batch_stats = variables.get('batch_stats', EMPTY_STATS),
    )
    map_ckpt_prefix = f"map_{args.dataset}"

    print("== Model Summary ==")
    print_summary(variables)
    
    # =========== PART A: MAP TRAINING ===========
    if args.mode in ["train_map", "full_pipeline"]:
        map_model_state = train_map(
            model_state,
            train_loader,
            test_loader,
            model_type=model_type,
            alpha=alpha,
            num_epochs=epochs_map
        )
        # map_model_state, alpha = train_map_then_alpha(
        #     model_state,
        #     train_loader,
        #     test_loader,
        #     model_type=model_type,
        #     num_epochs=epochs_map,
        #     alpha0=alpha,
        #     alpha_lr=1e-2,
        #     alpha_every=5,
        #     burnin=25,
        #     full_set_size=full_set_size,
        # )
        save_checkpoint(
            train_state=map_model_state,
            ckpt_dir=args.ckpt_map,
            prefix=map_ckpt_prefix,
            step=epochs_map
        )
        
        print("[DONE] MAP training.")
        if args.mode == "train_map":
            return
    else:
        map_model_state = load_checkpoint(
            ckpt_dir=args.ckpt_map,
            prefix=map_ckpt_prefix,
            target=model_state
        )

    del train_loader
    del test_loader

    # =========== PART B: Inducing Points ===========
    ip_cfg              = opt_cfg["ip"]
    m_inducing          = ip_cfg["m"]
    epochs_inducing     = ip_cfg["epochs"]
    inducing_batch_size = ip_cfg["batch_size"]
    lr_inducing         = ip_cfg["lr"]
    mc_samples          = ip_cfg["mc_samples"]
    seed_inducing       = ip_cfg["seed"]
    st_samples          = ip_cfg["st_samples"]
    slq_samples         = ip_cfg["slq_samples"]
    slq_num_matvecs     = ip_cfg["slq_num_matvecs"]
    
    
    induc_ckpt_name = f"ind_{args.dataset}"
    rng_inducing = jax.random.PRNGKey(seed_inducing)
    train_loader_init, *_ = get_dataloaders(args.dataset, m_inducing)
    zinit = next(iter(train_loader_init))[0] # Init IP from training data sample
    train_loader_induc, _, val_loader = get_dataloaders(args.dataset, inducing_batch_size)
    
    alpha_ip = args.alpha_ip
    if alpha_ip is None:
        alpha_ip = grid_search_alpha(map_model_state,
                             zinit,
                             val_loader,
                             full_set_size=full_set_size,
                             model_type=model_cfg["type"],
                             num_mc_samples=ip_cfg["mc_samples"],
                             scalable=True)

    

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
            full_set_size=full_set_size,
            scalable=True,
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
    # else:
    #     # Load both the inducing points (zinduc)
    #     zinducing = load_array_checkpoint(
    #         ckpt_dir=args.ckpt_induc,
    #         name=induc_ckpt_name,
    #         step=epochs_inducing
    #     )


if __name__ == "__main__":
    main()