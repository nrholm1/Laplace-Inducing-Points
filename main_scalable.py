import argparse
import pdb
import numpy as np
import torch
from seaborn import set_style

import jax
import jax.numpy as jnp 
from flax.training import train_state
import optax
from tqdm import tqdm

from src.train_map import train_map
from src.toymodels import CNN
from src.toydata import JAXDataset, jax_collate_fn, get_dataloaders, load_mnist_numpy
from src.utils import print_options, save_checkpoint

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



def train_map_wrapper(num_epochs):
    batch_size = 32

    train_images, train_labels = load_mnist_numpy(train=True)
    test_images, test_labels = load_mnist_numpy(train=False)
    train_dataset = JAXDataset(train_images, train_labels)
    test_dataset = JAXDataset(test_images, test_labels)
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size, collate_fn=jax_collate_fn)
    
    model = CNN()
    state = create_train_state(jax.random.PRNGKey(0), model, learning_rate=0.005, momentum=0.9)

    train_map(state, train_loader, test_loader, model_type="classifier", num_epochs=num_epochs)

    return state


if __name__ == '__main__':
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
    
    epochs_map = 1
    
    if args.mode in ["train_map", "full_pipeline"]:
        # Train MAP and save a checkpoint.
        map_model_state = train_map_wrapper(epochs_map)
        save_checkpoint(map_model_state, ckpt_dir="./checkpoint/map", prefix="mnist", step=epochs_map)
    
    
    if args.mode in ["train_inducing", "full_pipeline"]:
        # Train inducing points and save a checkpoint
        # todo
        ...
    
    if args.mode in ["train_inducing", "visualize", "full_pipeline"]:
        # Visualize LLA approximation
        # todo
        # 1. Sampling algorithm
        # 2. Sample from full LLA (if possible? Maybe we will have to train a smaller CNN)
        # 3. Sample from inducing point LLA
        # 4. Plot uncertainties over class distributions - maybe we will have to get a less certain MAP model.
        ...
    
    pdb.set_trace()