from __future__ import annotations

import argparse
import os

import jax
import matplotlib.pyplot as plt
import optax

from src.scalemodels import TrainState, EMPTY_STATS
from src.toymodels import SimpleRegressor, SimpleClassifier
from src.toydata import get_dataloaders, load_toydata
from src.nplot import (
    plot_lla_2D_classification,
)
from src.train_map import train_map
from src.train_inducing import IPConfig, train_inducing_points
from src.utils import (
    flatten_nn_params,
    load_yaml,
    save_checkpoint,
    load_checkpoint,
    save_array_checkpoint,
    load_array_checkpoint,
    print_summary,
    print_options,
)


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
        "--full",
        action="store_true",
        help="If selected, compute full LLA.",
    )
    parser.add_argument(
        "--scalable",
        action="store_true",
        help="Use matrix-free scalable IP optimization and LLA sampling.",
    )
    parser.add_argument(
        "--num_mc_samples_lla",
        type=int,
        default=1000,
        help="Number of MC samples for LLA predictive dist.",
    )
    parser.add_argument(
        "--alpha_ip",
        type=float,
        default=None,
        help="Alpha for inducing-point objective. If omitted, falls back to config['optimization']['alpha'].",
    )
    parser.add_argument(
        "--plot_Z", action="store_true", help="Whether to plot inducing points."
    )
    parser.add_argument(
        "--plot_X", action="store_true", help="Whether to plot training points."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to an .npz file containing x,y arrays.",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to a YAML with model hyperparams (e.g. config/toyregressor.yml).",
    )
    parser.add_argument(
        "--optimization_config",
        type=str,
        required=True,
        help="Path to a YAML with all optimization hyperparams (for MAP and inducing).",
    )
    parser.add_argument(
        "--ckpt_map",
        type=str,
        default="checkpoint/map/",
        help="Directory for loading/saving the MAP model checkpoint.",
    )
    parser.add_argument(
        "--ckpt_induc",
        type=str,
        default="checkpoint/ind/",
        help="Directory for loading/saving the inducing points checkpoint.",
    )
    args = parser.parse_args()

    print_options(args)

    # Load model config
    cfg = load_yaml(args.model_config)
    model_cfg = cfg["model"]
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
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load optimization config
    opt_cfg = cfg["optimization"]
    alpha_default = opt_cfg["alpha"]
    alpha_ip = args.alpha_ip if args.alpha_ip is not None else alpha_default

    map_cfg = opt_cfg["map"]
    full_set_size = opt_cfg["full_set_size"]

    map_batch_size = map_cfg["batch_size"]
    epochs_map = map_cfg["epochs"]
    lr_map = map_cfg["lr"]
    seed_map = map_cfg["seed"]

    # Data
    train_loader, test_loader, _ = get_dataloaders(
        dataset=args.dataset, batch_size=map_batch_size
    )

    # Initialize model params
    dummy_input = next(iter(train_loader))[0][:1]
    variables = model.init(rng_model, dummy_input)

    print("== Model Summary ==")
    print_summary(variables)

    # IP hyperparams
    ip_cfg = opt_cfg["ip"]
    m_ip = ip_cfg["m"]
    epochs_ip = ip_cfg["epochs"]
    batch_size_ip = ip_cfg["batch_size"]
    lr_ip = ip_cfg["lr"]
    mc_samples = ip_cfg["mc_samples"]
    seed_ip = ip_cfg["seed"]
    st_samples = ip_cfg.get("st_samples", 256)
    slq_samples = ip_cfg.get("slq_samples", 4)
    slq_num_matvecs = ip_cfg.get("slq_num_matvecs", 32)
    ip_batch_frac = ip_cfg.get("ip_batch_frac", 0.25)

    # Build TrainState for MAP
    optimizer_map = optax.adam(lr_map)
    model_state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer_map,
        batch_stats=variables.get("batch_stats", EMPTY_STATS),
    )
    map_ckpt_prefix = f"map_{args.dataset}"

    # =========== PART A: MAP TRAINING ===========
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

        print("[DONE] MAP training.")
        if args.mode == "train_map":
            return
    else:
        map_state = load_checkpoint(
            ckpt_dir=args.ckpt_map, prefix=map_ckpt_prefix, target=model_state
        )

    # =========== PART B: Inducing Points ===========
    induc_ckpt_name = f"ind_{args.dataset}"
    rng_ip = jax.random.PRNGKey(seed_ip)

    # Loader for IP training
    train_loader_ip, *_ = get_dataloaders(
        dataset=args.dataset, batch_size=batch_size_ip
    )

    # Bootstrap initial inducing locations: take a full batch of size m_ip
    train_loader_init, _, _ = get_dataloaders(dataset=args.dataset, batch_size=m_ip)
    zinit = next(iter(train_loader_init))[0]

    # Learning-rate schedule with warmup + cosine decay
    if args.mode in ["train_inducing", "full_pipeline"]:
        total_steps = epochs_ip
        warmup_steps = int(0.1 * total_steps)  # 10% warmup

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr_ip,
            warmup_steps=warmup_steps,
            decay_steps=max(1, total_steps - warmup_steps),
            end_value=lr_ip * 0.1,
        )
        zoptimizer = optax.adam(learning_rate=schedule)

        # NEW: smaller, cleaner config for IP training (matches refactor)
        cfg_ip = IPConfig(
            st_samples=st_samples,
            slq_samples=slq_samples,
            slq_num_matvecs=slq_num_matvecs,
            ip_batch_frac=ip_batch_frac,
            scalable=args.scalable,
            model_type=model_type,
        )

        z_ip = train_inducing_points(
            map_state=map_state,
            Z_init=zinit,
            optimizer=zoptimizer,
            data_loader=train_loader_ip,
            rng=rng_ip,
            alpha=alpha_ip,
            full_set_size=full_set_size,
            cfg=cfg_ip,
            num_steps=epochs_ip,
        )

        save_array_checkpoint(
            array=z_ip,
            ckpt_dir=args.ckpt_induc,
            name=induc_ckpt_name,
            step=epochs_ip,
        )

        print("[DONE] Inducing training.")
    else:
        z_ip = load_array_checkpoint(
            ckpt_dir=args.ckpt_induc, name=induc_ckpt_name, step=epochs_ip
        )

    # =========== PART C: Visualization ===========
    if args.mode in ["visualize", "full_pipeline"]:
        os.makedirs("fig", exist_ok=True)

        flat_params_map, unravel_fn_map = flatten_nn_params(map_state.params)

        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        full_lla = args.full
        if full_lla:
            fig.suptitle(f"Full LLA / {opt_cfg['full_set_size']} data points")
        else:
            fig.suptitle(f"IP LLA / {m_ip} inducing points, {epochs_ip} steps")

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
            matrix_free=args.scalable,
            num_mc_samples=args.num_mc_samples_lla,
            plot_Z=not full_lla if args.plot_Z or True else False,  # default: show Z for IP
            plot_X=args.plot_X,
            flat_params=flat_params_map,
            unravel_fn=unravel_fn_map,
        )
        plt.tight_layout()
        suffix_if_matrixfree = "_mf" if args.scalable else ""
        plt.savefig(
            f"fig/{args.dataset}_{model_type}_lla_{'full' if full_lla else 'ip'}{suffix_if_matrixfree}.pdf"
        )

        print("[DONE] Visualization complete.")


if __name__ == "__main__":
    main()
