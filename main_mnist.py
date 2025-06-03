import argparse
import pdb
# import pdb
from matplotlib import pyplot as plt
import numpy as np
import torch
from seaborn import set_style

import jax
import jax.numpy as jnp 
from flax.training import train_state
from flax.linen import softmax
import optax
from tqdm import tqdm

from src.train_map import train_map
from src.train_inducing import train_inducing_points
from src.scalemodels import CNN
from src.data import JAXDataset, jax_collate_fn, get_dataloaders, load_mnist_numpy
from src.utils import load_array_checkpoint, load_checkpoint, print_options, save_array_checkpoint, save_checkpoint

# Set random seeds for reproducibility.
torch.manual_seed(0)
np.random.seed(0)

# Set plotting style.
set_style('darkgrid')


def create_train_state(rng, model, learning_rate, momentum):
    """Initialize TrainState with model parameters and an optimizer."""
    dummy_inp = jnp.ones((1, 28, 28, 1), jnp.float32)
    params = model.init(rng, dummy_inp)
    tx = optax.adamw(learning_rate=learning_rate, b1=momentum)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)



def visualize_lla_mnist():
    ...



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="full_pipeline",
                        choices=["train_map", "train_inducing", "visualize", "full_pipeline"],
                        help="Which phase(s) to run.")
    parser.add_argument("--ckpt_map", type=str, default="checkpoint/map/",
                        help="Directory for loading/saving the MAP model checkpoint.")
    parser.add_argument("--ckpt_induc", type=str, default="checkpoint/ind/",
                        help="Directory for loading/saving the inducing points checkpoint.")
    args = parser.parse_args()

    # Print selected options
    print_options(args)
    
    epochs_map = 5
    batch_size = 32
    seed_inducing = 1337
    m_inducing = 256
    lr_inducing = 0.01
    model_type = "classifier"
    mc_samples = 0 # dummy param
    alpha = 0.05
    epochs_inducing = 100
    

    xtrain, ytrain = load_mnist_numpy(train=True)
    xtest, ytest = load_mnist_numpy(train=False)
    train_dataset = JAXDataset(xtrain, ytrain)
    test_dataset = JAXDataset(xtest, ytest)
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size, collate_fn=jax_collate_fn)
    
    map_ckpt_prefix = f"map_mnist"
    model = CNN()
    model_state = create_train_state(jax.random.PRNGKey(0), model, learning_rate=0.005, momentum=0.9)
    
    # =========== MAP Training ===========
    if args.mode in ["train_map", "full_pipeline"]:
        # Train MAP and save a checkpoint.
        map_model_state = train_map(model_state, train_loader, test_loader, model_type="classifier", num_epochs=epochs_map, alpha=alpha)
        save_checkpoint(map_model_state, ckpt_dir="./checkpoint/map", prefix=map_ckpt_prefix, step=epochs_map)
        print("[DONE] MAP training.")
        if args.mode == "train_map":
            return
    else:
        map_model_state = load_checkpoint(
            ckpt_dir=args.ckpt_map,
            prefix=map_ckpt_prefix,
            target=model_state
        )
    
    # =========== Inducing Points ===========
    induc_ckpt_name = f"ind_mnist"
    rng_inducing = jax.random.PRNGKey(seed_inducing)
    m_inducing = min(m_inducing, len(test_dataset))
    _, test_loader = get_dataloaders(train_dataset, test_dataset, m_inducing)
    zinit = next(iter(test_loader))[0]
    if args.mode in ["train_inducing", "full_pipeline"]:
        # Train inducing points and save a checkpoint
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
        
    
    if args.mode in ["train_inducing", "visualize", "full_pipeline"]:
        # Visualize LLA approximation
        # todo
        # 1. Sampling algorithm
        # 2. Sample from full LLA (if possible? Maybe we will have to train a smaller CNN)
        # 3. Sample from inducing point LLA
        # 4. Plot uncertainties over class distributions - maybe we will have to get a less certain MAP model.
        
        # Update global font settings
        plt.rcParams.update({
            'font.size': 12,                  # Base font size for text
            'font.family': 'serif',           # Use serif fonts
            # 'font.serif': ['Times New Roman', 'DejaVu Serif'],  # List of preferred serif fonts
            'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],  # List of preferred serif fonts
            'font.sans-serif': ['Arial', 'DejaVu Sans'],         # Preferred sans-serif fonts (if needed)
            'axes.titlesize': 20,             # Font size for axes titles
            'axes.labelsize': 14,             # Font size for x and y labels
            'xtick.labelsize': 14,            # Font size for x-axis tick labels
            'ytick.labelsize': 14,            # Font size for y-axis tick labels
            'legend.fontsize': 14,            # Font size for legend text
            'figure.titlesize': 24,            # Font size for the figure title
        })
        plt.rcParams['text.usetex'] = True

        
        nsamples = 16
        nrows = 2
        figsize = (16,8)
        colors = [
            "#e41a1c",  # Red
            "#377eb8",  # Blue
            "#4daf4a",  # Green
            "#984ea3",  # Purple
            "#ff7f00",  # Orange
            "#ffff33",  # Yellow
            "#a65628",  # Brown
            "#f781bf",  # Pink
            "#999999",  # Grey
            "#66c2a5"   # Teal
        ]
        # figsize = (30,15)
        
        # get a random test set batch to visualize
        supersample = max(nsamples, 1024) # make a superset of samples and choose the top nsamples highest entropy
        _, test_loader = get_dataloaders(train_dataset, test_dataset, supersample)
        # data_batch = next(iter(test_loader))
        _iter = iter(test_loader)
        next(_iter)
        data_batch = next(_iter)
        
        # todo visualize_lla_mnist(zinducing, map_model_state, data_batch)
        X,y = data_batch
        classes = jnp.arange(10)
        
        # compile prediction 
        @jax.jit
        def pred_step(params, X):
            logits = map_model_state.apply_fn(params, X)
            preds = logits.argmax(axis=1)
            probs = softmax(logits, axis=1)
            return preds, probs
        
        preds, probs = pred_step(map_model_state.params, X)
        entropies = jax.scipy.special.entr(probs).sum(axis=1)
        topk_entr = jnp.argsort(entropies, descending=True)[:nsamples]
        X = X[topk_entr]
        y = y[topk_entr]
        probs = probs[topk_entr]
        preds = preds[topk_entr]

        fig,axs = plt.subplots(2*nrows, int(nsamples/nrows), figsize=figsize)
        for i,ax in enumerate(axs[1::2].flatten()):
            # ax.set_title(f"Pred: {preds[i]}")
            # ax.set_title(f"Pred={preds[i]} ({int(y[i])})")
            ax.imshow(X[i], cmap='gray_r')
            ax.grid(False)
            ax.axis(False)
        
        for i,ax in enumerate(axs[::2].flatten()):
            ax.set_facecolor('white')
            ax.bar(classes, probs[i], color=colors, edgecolor='#333')
            ax.set_ylim(0.0, 1.02)
            # ax.grid(True, color='#ddd')
            ax.grid(False)
            ax.get_yaxis().set_visible(False)
            ax.set_xticks(classes)
            ax.set_xticklabels(classes)
        
        # plt.suptitle(f"[MNIST] Inducing LLA after {epochs_inducing} steps.")
        plt.suptitle(f"MNIST | High entropy samples.")
        plt.tight_layout()
        plt.savefig("fig/mnist.pdf")
    
    # pdb.set_trace()

if __name__ == '__main__':
    main()