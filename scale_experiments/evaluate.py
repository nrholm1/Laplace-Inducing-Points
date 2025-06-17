import pdb
import argparse, functools, pathlib, time
import numpy as np

from sklearn.metrics import roc_auc_score          # pip install scikit-learn

import jax, jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
# from flax.training.train_state import TrainState
import optax
from tqdm import tqdm

from src.lla import predict_lla_scalable, predict_lla_dense
from src.scaledata   import get_dataloaders
from src.scalemodels import EMPTY_STATS, get_model, TrainState
from src.utils       import (load_yaml, load_checkpoint,
                             load_array_checkpoint, print_options,
                             print_summary)
from src.nplot import plot_grayscale
from src.toydata import get_dataloaders as get_toydataloaders, load_toydata

# Helper
def build_state(model_cfg, lr, dummy_input):
    rng   = jax.random.PRNGKey(model_cfg["seed"])
    model = get_model(model_cfg)

    variables = model.init(rng, dummy_input, train=True)
    params    = variables["params"]

    tx     = optax.adam(lr)
    state  = TrainState.create(apply_fn=model.apply,
                               params=params,
                               batch_stats = variables.get('batch_stats', EMPTY_STATS),
                               tx=tx)
    return model, state


# ----------------------  calibration / OOD utils  ---------------------
def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score (lower is better)."""
    one_hot = np.eye(probs.shape[-1])[np.astype(labels, int)]
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error (Naïve histogram, confidence vs. accuracy)."""
    confidences = probs.max(1)
    predictions = probs.argmax(1)
    accuracies  = (predictions == labels)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val   = 0.0
    N         = len(labels)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        bin_conf   = confidences[mask].mean()
        bin_acc    = accuracies[mask].mean()
        ece_val   += np.abs(bin_conf - bin_acc) * mask.mean()
    return ece_val

def ood_scores(probs: np.ndarray) -> np.ndarray:
    """Return a scalar OOD score per sample (higher ⇒ more ID-like)."""
    return -probs.max(1)


def auroc_ood(state, id_probs: np.ndarray, ood_loader, Z,
              alpha, full_set_size, model_type, num_mc_samples, rng,
              scalable=True):
    # rng = jax.random.PRNGKey(1337)
    ood_probs = []

    for xb, _ in tqdm(ood_loader, desc="OOD pass"):
        rng, sub = jax.random.split(rng)
        _, _, mean = batch_nll(state, xb, _,
                               Z,
                               alpha=alpha,
                               full_set_size=full_set_size,
                               model_type=model_type,
                               num_mc_samples=num_mc_samples,
                               rng=sub,
                               scalable=scalable,
                               return_mean=True)
        ood_probs.append(np.asarray(mean))
    ood_probs = np.concatenate(ood_probs, axis=0)

    scores = np.concatenate([ood_scores(id_probs),
                             ood_scores(ood_probs)])
    labels = np.concatenate([np.zeros(len(id_probs)),
                             np.ones(len(ood_probs))])
    return roc_auc_score(labels, scores)


# ---------------------------  NLL utils  ------------------------------
# @functools.partial(jax.jit, static_argnames=("model_type", "num_mc_samples", "alpha", "full_set_size", "scalable", "return_mean"))
def batch_nll(state, x, y, Z, *, alpha, full_set_size,
              model_type, num_mc_samples, rng, scalable=True, return_mean=False):
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
    # variables = {
    #     'params': state.params,
    #     'batch_stats': state.batch_stats
    # }
    # logit_samples = state.apply_fn(variables, x, train=False, mutable=False)[None] # ! MAP estimator sanity check
    
    S = logit_samples.shape[0]
    log_probs = jax.nn.log_softmax(logit_samples, axis=-1)          # (S,B,C)
    # gather the log-probability of the *true* class
    y_int = y.squeeze().astype(jnp.int32)                           # ensure int32
    log_p_true = jnp.take_along_axis(
        log_probs,
        y_int[None, :, None],                                       # (1,B,1)  broadcasts over S
        axis=-1
    ).squeeze(-1)                                                  # (S,B)

    # log of the MC-averaged predictive probability  log( 1/S Σ_s p_s )
    log_avg_prob = (
        jax.scipy.special.logsumexp(log_p_true, axis=0)             # (B,)
        - jnp.log(S)
    )

    nll = -jnp.mean(log_avg_prob)

    probs = jax.nn.softmax(logit_samples, axis=-1)      # (S,B,C)
    mean  = probs.mean(axis=0)                          # (B,C)

    # one_hot_y = jax.nn.one_hot(y.squeeze(), logit_samples.shape[-1])
    # nll = jnp.mean(optax.softmax_cross_entropy(logit_samples, one_hot_y))
    acc   = (mean.argmax(-1) == y.squeeze()).mean()

    if return_mean:
        return nll, acc, mean
    return nll, acc


def eval_dataset(state, dataloader, Z, alpha,
                 full_set_size, model_type, num_mc_samples,
                 rng,
                 scalable=True):

    tot_nll, tot_correct, tot_N = 0.0, 0.0, 0
    # rng = jax.random.PRNGKey(420) # static key

    pbar = tqdm(dataloader)
    for x_b, y_b in pbar:
        # pdb.set_trace()
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


def eval_dataset_extended(state, 
                          dataloader, 
                          Z, 
                          alpha,
                          full_set_size, 
                          model_type, 
                          num_mc_samples,
                          rng,
                          scalable=True):

    tot_nll, tot_correct, tot_N = 0.0, 0.0, 0
    all_probs, all_labels       = [], []
    # rng = jax.random.PRNGKey(4210)

    pbar = tqdm(dataloader)
    for x_b, y_b in pbar:
        rng, sub = jax.random.split(rng)
        nll, acc, mean_probs = batch_nll(state, x_b, y_b,
                                         Z,
                                         alpha=alpha,
                                         full_set_size=full_set_size,
                                         model_type=model_type,
                                         num_mc_samples=num_mc_samples,
                                         rng=sub,
                                         scalable=scalable,
                                         return_mean=True)

        bs        = x_b.shape[0]
        tot_nll  += float(nll)  * bs
        tot_correct += float(acc) * bs
        tot_N    += bs

        all_probs.append(np.asarray(mean_probs))
        all_labels.append(np.asarray(y_b).squeeze())

        pbar.set_description(f"[NLL {nll:.3f}] [ACC {acc:.3f}]")

    probs  = np.concatenate(all_probs,  axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # extra metrics
    bri  = brier_score(probs, labels)
    cal  = ece(probs, labels)

    return tot_nll / tot_N, tot_correct / tot_N, bri, cal, probs, labels



# ------------------------------  CLI  ---------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Dataset name registered in get_dataloaders.")
    parser.add_argument("--ood-dataset", default=None,
                        help="Dataset name registered in get_dataloaders for Out-Of-Distribution tests.")
    parser.add_argument("--config",  required=True,
                        help="YAML with model & optimisation hyper-params.")
    parser.add_argument("--ckpt_map",  default="checkpoint/map/",
                        help="Directory with the MAP checkpoint.")
    parser.add_argument("--ckpt_induc", default="checkpoint/ind/",
                        help="Directory with inducing-point checkpoint.")
    parser.add_argument("--scalable", action="store_true",
                        help="Whether to use scalable sampling for LLA.")
    parser.add_argument("--alpha_ip", default=None,
                        help="Provide an alpha value if it should not be read from the config.")
    args = parser.parse_args()
    print_options(args)

    # ---------------- config / loaders ----------------
    cfg        = load_yaml(args.config)
    model_cfg  = cfg["model"]
    opt_cfg    = cfg["optimization"]

    alpha          = float(args.alpha_ip) if args.alpha_ip is not None else opt_cfg["alpha"]
    print(f"⍺={alpha}")
    full_set_size  = opt_cfg["full_set_size"]

    map_cfg        = opt_cfg["map"]
    batch_size     = map_cfg["batch_size"]
    lr_map         = map_cfg["lr"]

    if args.dataset in ["spiral", "banana"]: # todo implement more toy datasets?
        train_loader, test_loader, _ = get_toydataloaders(args.dataset, batch_size)
        # train_loader, _, _ = get_toydataloaders(args.dataset, 400)
    else:
        train_loader, test_loader, _ = get_dataloaders(args.dataset, batch_size)
    
    if args.ood_dataset is not None:
        if args.ood_dataset in ["spiral", "banana", "ring"]:
            ood_loader, _, _ = get_toydataloaders(args.ood_dataset, batch_size)
        else:
            _, ood_loader, _ = get_dataloaders(args.ood_dataset, batch_size)
    else:
        ood_loader = None

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
    # ! plotting code
    # xtrain,ytrain = next(iter(train_loader))
    # pdb.set_trace()
    # from src.nplot import make_comparison_figure, plot_binary_classification_data
    # ood_loader, _, _ = get_toydataloaders(args.ood_dataset, 256)
    # xood,yood = next(iter(ood_loader))
    # fig,axs=make_comparison_figure(state, xtrain, ytrain, xtrain, alpha, matrix_free=False, num_mc_samples=500)
    # plot_binary_classification_data(xood, yood.squeeze(), axs[0])
    # plot_binary_classification_data(xood, yood.squeeze(), axs[1])
    # plt.savefig("fig/test.pdf", dpi=300, bbox_inches="tight")
    
    # --------------   evaluation   --------------------
    iters = 3
    rng = jax.random.PRNGKey(155858)
    for i in range(iters):
        t0 = time.time()
        rng = jax.random.fold_in(rng, i)
        nll, acc, bri, cal, probs, labels = eval_dataset_extended(state,
                                test_loader,
                                Z,
                                # xtrain,
                                rng=rng,
                                alpha=alpha,
                                full_set_size=full_set_size,
                                model_type=model_cfg["type"],
                                num_mc_samples=ip_cfg["mc_samples"],
                                scalable=args.scalable)
        dt = time.time() - t0

        print(f"\nTest NLL   : {nll:8.5f}"
            f"\nTest Acc   : {acc*100:8.3f} %"
            f"\nBrier      : {bri:8.5f}"
            f"\nECE (15bin): {cal:8.5f}"
            f"\nTime       : {dt:6.1f} s")

        rng = jax.random.fold_in(rng, i)
        if ood_loader is not None:
            auroc = auroc_ood(state,
                            probs,
                            ood_loader,
                            Z,
                            alpha=alpha,
                            rng=rng,
                            full_set_size=full_set_size,
                            model_type=model_cfg["type"],
                            num_mc_samples=ip_cfg["mc_samples"],
                            scalable=args.scalable)
            print(f"OOD AUROC  : {auroc*100:8.3f} %")

if __name__ == "__main__":
    main()
