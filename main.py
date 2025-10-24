from __future__ import annotations

import argparse
import math
import os
import pdb

import jax
import optax
import matplotlib.pyplot as plt

from src.scalemodels import EMPTY_STATS, TrainState, get_model as get_scale_model
from src.train_map import train_map
from src.train_inducing import IPConfig, train_inducing_points
from src.utils import (
    flatten_nn_params,
    load_yaml,
    print_dict,
    save_checkpoint,
    load_checkpoint,
    save_array_checkpoint,
    load_array_checkpoint,
    print_summary,
    print_options,
    ip_config_from_dict
)

# Toy stack
from src.toymodels import SimpleRegressor, SimpleClassifier
from src.toydata import get_dataloaders as get_toy_dataloaders, load_toydata
from src.nplot import plot_lla_2D_classification

# Scale stack
from src.scaledata import get_dataloaders as get_scale_dataloaders



def _build_model_and_vars(variant: str, model_cfg: dict, dummy_input):
    if variant in ["toy-dense", "toy-mf"]:
        model_type = model_cfg.get("name", "regressor")
        num_h = model_cfg["num_h"]
        num_l = model_cfg["num_l"]
        num_c = model_cfg.get("num_c", 2) if model_type == "classifier" else 1
        model_seed = model_cfg["seed"]
        rng_model = jax.random.PRNGKey(model_seed)
        if model_type == "regressor":
            model = SimpleRegressor(numh=num_h, numl=num_l)
        elif model_type == "classifier":
            model = SimpleClassifier(numh=num_h, numl=num_l, numc=num_c)
        else:
            raise ValueError(f"Unknown toy model_type: {model_type}")
        variables = model.init(rng_model, dummy_input)
        return model, variables, model_type
    else:  # scale
        model_type = model_cfg["type"]
        model_seed = model_cfg["seed"]
        rng_model = jax.random.PRNGKey(model_seed)
        model = get_scale_model(model_cfg)
        variables = model.init(rng_model, dummy_input, train=True)
        return model, variables, model_type


def _get_dataloaders(variant: str, dataset: str, batch_size: int, *, aug=None, num_workers=None):
    if variant == "toy":
        # toy dataloaders: (train, test, val)
        return get_toy_dataloaders(dataset=dataset, batch_size=batch_size)
    else:
        # scale dataloaders: allow aug/workers args
        if num_workers is None:
            num_workers = 0
        if aug is None:
            aug = True
        return get_scale_dataloaders(dataset, batch_size, num_workers=num_workers, aug=aug)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="full_pipeline",
        choices=["train_map", "train_inducing", "visualize", "full_pipeline"],
        help="Which phase(s) to run.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["toy-dense", "toy-mf", "scale"],
        help="Pipeline variant.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name or path (.npz for toy)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML with model + optimization hyperparams.",
    )
    parser.add_argument(
        "--alpha_ip",
        type=float,
        default=None,
        help="IP alpha override.",
    )
    parser.add_argument(
        "--ckpt_map",
        type=str,
        default="checkpoint/map/",
        help="Directory for MAP checkpoints.",
    )
    parser.add_argument(
        "--ckpt_induc",
        type=str,
        default="checkpoint/ind/",
        help="Directory for inducing-point checkpoints.",
    )
    # Toy-only niceties (safe to pass for scale; they’ll just be ignored)
    parser.add_argument(
        "--full",
        action="store_true",
        help="(toy) If set, compute full LLA in visualization.",
    )
    parser.add_argument(
        "--plot_Z",
        action="store_true",
        help="(toy) Plot inducing points in visualization.",
    )
    parser.add_argument(
        "--plot_X",
        action="store_true",
        help="(toy) Plot training points in visualization.",
    )
    args = parser.parse_args()

    print_options(args)

    # Load config (unified)
    cfg = load_yaml(args.config)
    print_dict(cfg)
    model_cfg = cfg["model"]
    opt_cfg = cfg["optimization"]

    # Variant
    variant = args.variant
    model_type = model_cfg["type"]

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
    

    # Loaders for MAP
    if variant in ["toy-dense", "toy-mf"]:
        train_loader, test_loader, val_loader = _get_dataloaders("toy", args.dataset, map_batch_size)
        dummy_input = next(iter(train_loader))[0][:1]
    else:
        train_loader, test_loader, val_loader = _get_dataloaders(
            "scale", args.dataset, map_batch_size, num_workers=0
        )
        dummy_input = next(iter(train_loader))[0][:1]  # e.g., (1, 28, 28, 1)
    is_matrix_free = args.variant != "toy-dense"

    # Build model + variables
    model, variables, model_type = _build_model_and_vars(variant, model_cfg, dummy_input)

    # Optimizer for MAP
    if variant == "scale":
        # Cosine decay for MAP
        steps_per_epoch = max(1, math.ceil(full_set_size / map_batch_size))
        total_steps = epochs_map * steps_per_epoch
        lr_schedule = optax.cosine_decay_schedule(
            init_value=lr_map,
            decay_steps=total_steps,
            alpha=0.08,  # final LR = alpha * init_value
        )
        optimizer_map = optax.adam(learning_rate=lr_schedule)
    else:
        # Constant LR for MAP
        optimizer_map = optax.adam(lr_map)

    model_state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer_map,
        batch_stats=variables.get("batch_stats", EMPTY_STATS),
    )
    map_ckpt_prefix = f"map_{args.dataset}"

    print("== Model Summary ==")
    print_summary(variables)

    # ======== A) MAP training ========
    if args.mode in ["train_map", "full_pipeline"]:
        map_state = train_map(
            model_state,
            train_loader,
            test_loader,
            model_type=model_type,
            alpha=alpha_default,
            num_epochs=epochs_map,
        )
        save_checkpoint(
            train_state=map_state,
            ckpt_dir=args.ckpt_map,
            prefix=map_ckpt_prefix,
            step=epochs_map,
        )
        if args.mode == "train_map":
            print("[DONE] MAP training.")
            return
    else:
        map_state = load_checkpoint(
            ckpt_dir=args.ckpt_map, prefix=map_ckpt_prefix, target=model_state
        )

    # Free the big loaders if desired
    del train_loader
    del test_loader

    # ======== B) Inducing points ========
    # Init Z from a batch of size m_ip; for scale we ensure aug=False to sample real data points
    if variant in ["toy-dense", "toy-mf"]:
        train_loader_init, _, _ = _get_dataloaders("toy", args.dataset, m_ip)
        zinit = next(iter(train_loader_init))[0]
        train_loader_ip, *_ = _get_dataloaders("toy", args.dataset, batch_size_ip)
    else:
        train_loader_init, *_ = _get_dataloaders("scale", args.dataset, m_ip, aug=False)
        zinit = next(iter(train_loader_init))[0]
        train_loader_ip, _, val_loader = _get_dataloaders(
            "scale", args.dataset, batch_size_ip, aug=False
        )

    # Choose alpha for IP:
    if args.alpha_ip is not None:
        alpha_ip = args.alpha_ip
    else:
        alpha_ip = alpha_default

    total_steps_ip = epochs_ip
    warmup_steps = int(0.1 * total_steps_ip)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr_ip,
        warmup_steps=warmup_steps,
        decay_steps=max(1, total_steps_ip - warmup_steps),
        end_value=lr_ip * 0.1,
    )
    zoptimizer = optax.adam(learning_rate=schedule)

    if args.mode in ["train_inducing", "full_pipeline"]:
        rng_ip = jax.random.PRNGKey(seed_ip)
        z_ip, alpha_ip, map_state = train_inducing_points(
            map_state=map_state,
            Z_init=zinit,
            optimizer=zoptimizer,
            data_loader=train_loader_ip,
            rng=rng_ip,
            alpha=alpha_ip,
            full_set_size=full_set_size,
            ip_cfg=ip_cfg_pytree,
            num_steps=epochs_ip,
        )

        save_array_checkpoint(
            array=z_ip,
            ckpt_dir=args.ckpt_induc,
            name=f"ind_{args.dataset}",
            step=epochs_ip,
        )

        print("[DONE] Inducing training.")
    else:
        z_ip = load_array_checkpoint(
            ckpt_dir=args.ckpt_induc, name=f"ind_{args.dataset}", step=epochs_ip
        )

    # ======== C) Visualization (toy only) ========
    if args.mode == "visualize":
        if variant not in ["toy-dense", "toy-mf"]:
            print("[WARN] Visualization is only implemented for the toy pipeline; skipping.")
            pdb.set_trace()
            plt.clf(); plt.imshow(z_ip[501], cmap='gray'); plt.grid(); plt.savefig("mnist.png")
            return

        os.makedirs("fig", exist_ok=True)
        flat_params_map, unravel_fn_map = flatten_nn_params(map_state.params)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        full_lla = args.full
        fig.suptitle(
            f"{'Full LLA' if full_lla else 'IP LLA'} / "
            f"{full_set_size if full_lla else m_ip} points, {epochs_ip if not full_lla else epochs_map} steps"
        )
        (xtrain, ytrain), *_ = load_toydata(args.dataset)
        plot_lla_2D_classification(
            fig,
            ax,
            map_state,
            xtrain,
            ytrain,
            z_ip,
            alpha_ip,
            key=jax.random.fold_in(jax.random.PRNGKey(seed_ip), 1),
            mode="full_lla" if full_lla else "ip_lla",
            matrix_free=args.variant != "toy-dense",
            num_mc_samples=ip_cfg_raw.get("mc_samples", 1000),
            plot_Z=args.plot_Z or (not full_lla),
            plot_X=args.plot_X,
            flat_params=flat_params_map,
            unravel_fn=unravel_fn_map,
        )
        plt.tight_layout()
        suffix_if_matrixfree = "_mf"
        filename = f"fig/{args.dataset}_{model_type}_lla_{'full' if full_lla else 'ip'}{suffix_if_matrixfree if is_matrix_free else ''}.pdf"
        plt.savefig(
            filename
        )
        print("[DONE] Visualization complete.")
        print(f"Saved to {filename}")
    

if __name__ == "__main__":
    main()
