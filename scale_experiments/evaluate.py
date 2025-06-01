# evaluate.py
# ---------------------------------------------------------------------
import pdb
import argparse, functools, pathlib, time

import jax, jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from flax.training.train_state import TrainState
import optax
from tqdm import tqdm

from src.lla import predict_lla_scalable, predict_lla_dense
from src.scaledata   import get_dataloaders
from src.scalemodels import get_model
from src.utils       import (load_yaml, load_checkpoint,
                             load_array_checkpoint, print_options,
                             print_summary)
from src.nplot import plot_mnist
from src.toydata import get_dataloaders as get_toydataloaders, load_toydata

# ---------------------------------------------------------------------
def build_state(model_cfg, lr, dummy_input):
    rng   = jax.random.PRNGKey(model_cfg["seed"])
    model = get_model(model_cfg)

    variables = model.init(rng, dummy_input)    # {'params': …}
    params    = variables # variables["params"]

    tx     = optax.adam(lr)
    state  = TrainState.create(apply_fn=model.apply,
                               params=params,
                               tx=tx)
    return model, state


# ---------------------------  NLL utils  ------------------------------
@functools.partial(jax.jit, static_argnames=("model_type", "num_mc_samples", "alpha", "full_set_size", "scalable"))
def batch_nll(state, x, y, Z, *, alpha, full_set_size,
              model_type, num_mc_samples, rng, scalable=True):
    """Return (batch_nll, batch_correct)."""    

    if scalable:
        logit_samples = predict_lla_scalable(
            state, x,
            Z,
            model_type=model_type,
            alpha=alpha,
            full_set_size=full_set_size,
            num_samples=num_mc_samples,
            key=rng
        )                                 # (S, B, C)
    else: 
        logit_dist = predict_lla_dense(
            state, x,
            Z,
            model_type=model_type,
            alpha=alpha,
            full_set_size=full_set_size,
        )
        logit_samples = logit_dist.sample(seed=rng, sample_shape=(num_mc_samples,))
    
    # logit_samples = state.apply_fn(state.params, x)[None] # ! MAP estimator sanity check

    probs = jax.nn.softmax(logit_samples, axis=-1)      # (S,B,C)
    mean  = probs.mean(axis=0)                          # (B,C)

    one_hot_y = jax.nn.one_hot(y.squeeze(), logit_samples.shape[-1])
    nll = jnp.mean(optax.softmax_cross_entropy(logit_samples, one_hot_y))
    acc   = (mean.argmax(-1) == y.squeeze()).mean()

    # pdb.set_trace()

    return nll, acc


def eval_dataset(state, test_loader, Z, alpha,
                 full_set_size, model_type, num_mc_samples,
                 scalable=True):

    tot_nll, tot_correct, tot_N = 0.0, 0.0, 0
    rng = jax.random.PRNGKey(420) # static key

    pbar = tqdm(test_loader)
    for x_b, y_b in pbar:
        rng, sub = jax.random.split(rng)
        nll, acc = batch_nll(state, x_b, y_b,
                             Z,
                             alpha=alpha,
                             full_set_size=full_set_size,
                             model_type=model_type,
                             num_mc_samples=num_mc_samples,
                             rng=sub,
                             scalable=scalable)

        pbar.set_description(f"[NLL {nll:.3f}] [ACC {acc:.3f}]")
        bs        = x_b.shape[0]
        tot_nll  += float(nll)  * bs
        tot_correct += float(acc) * bs
        tot_N    += bs

    return tot_nll / tot_N, tot_correct / tot_N


# ------------------------------  CLI  ---------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Dataset name registered in get_dataloaders.")
    parser.add_argument("--config",  required=True,
                        help="YAML with model & optimisation hyper-params.")
    parser.add_argument("--ckpt_map",  default="checkpoint/map/",
                        help="Directory with the MAP checkpoint.")
    parser.add_argument("--ckpt_induc", default="checkpoint/ind/",
                        help="Directory with inducing-point checkpoint.")
    parser.add_argument("--scalable", action="store_true",
                        help="Whether to use scalable sampling for LLA.")
    args = parser.parse_args()
    print_options(args)

    # ---------------- config / loaders ----------------
    cfg        = load_yaml(args.config)
    model_cfg  = cfg["model"]
    opt_cfg    = cfg["optimization"]

    alpha          = opt_cfg["alpha"]
    full_set_size  = opt_cfg["full_set_size"]

    map_cfg        = opt_cfg["map"]
    batch_size     = map_cfg["batch_size"]
    lr_map         = map_cfg["lr"]

    if args.dataset in ["spiral", "banana"]: # todo implement more toy datasets?
        train_loader, test_loader = get_toydataloaders(args.dataset, batch_size)
    else:
        train_loader, test_loader = get_dataloaders(args.dataset, batch_size)

    # ---------------- model & MAP weights -------------
    dummy_input = next(iter(train_loader))[0][:1]          # (1,28,28,1)
    model, state = build_state(model_cfg, lr_map, dummy_input)

    map_ckpt_prefix = f"map_{args.dataset}"
    state = load_checkpoint(
            ckpt_dir=args.ckpt_map,
            prefix=map_ckpt_prefix,
            target=state
        )

    print("== Loaded MAP weights ==");  print_summary(state.params)

    # ---------------- inducing points -----------------
    ip_cfg   = opt_cfg["ip"]
    induc_ckpt_name = f"ind_{args.dataset}"
    epochs_inducing = ip_cfg["epochs"]
    Z = load_array_checkpoint(
            ckpt_dir=args.ckpt_induc,
            name=induc_ckpt_name,
            step=epochs_inducing
        )
    
    # todo for debugging
    # plot_mnist(Z[:32].squeeze())
    # exit()
    # (xtrain,_),_ = load_toydata(args.dataset)
    # print(xtrain.shape)
    xload, _ = get_dataloaders(args.dataset, 100)
    xtrain = next(iter(xload))[0]
    
    # --------------   evaluation   --------------------
    t0 = time.time()
    nll, acc = eval_dataset(state,
                            test_loader,
                            # Z,
                            xtrain,
                            alpha=alpha,
                            full_set_size=full_set_size,
                            model_type=model_cfg["type"],
                            num_mc_samples=ip_cfg["mc_samples"],
                            scalable=args.scalable)
    dt = time.time() - t0

    print(f"\nTest NLL : {nll:8.5f}"
          f"\nTest Acc : {acc*100:8.3f} %"
          f"\nTime     : {dt:6.1f} s")


if __name__ == "__main__":
    main()
