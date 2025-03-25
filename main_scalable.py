import argparse
# import pdb
import numpy as np
import torch
from seaborn import set_style

import jax
import jax.numpy as jnp 
from flax.training import train_state
import optax
from tqdm import tqdm

from src.train_map import train_map
from src.train_inducing import train_inducing_points
from src.toymodels import CNN
from src.toydata import JAXDataset, jax_collate_fn, get_dataloaders, load_mnist_numpy
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



def train_map_wrapper(model_state, num_epochs, train_loader, test_loader):
    train_map(model_state, train_loader, test_loader, model_type="classifier", num_epochs=num_epochs)
    return model_state



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
    
    epochs_map = 1
    batch_size = 32
    seed_inducing = 1337
    m_inducing = 300
    lr_inducing = 0.01
    model_type = "classifier"
    mc_samples = 0 # dummy param
    alpha_inducing = 0.5
    epochs_inducing = 1
    

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
        map_model_state = train_map_wrapper(model_state, epochs_map, train_loader, test_loader)
        save_checkpoint(map_model_state, ckpt_dir="./checkpoint/map", prefix="mnist", step=epochs_map)
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
        
        with jax.profiler.trace("trace"):
            zinducing = train_inducing_points(
                map_model_state, 
                zinit, 
                zoptimizer,
                dataloader=train_loader,
                rng=rng_inducing,
                model_type=model_type,
                num_mc_samples=mc_samples,
                alpha=alpha_inducing,
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
        ...
    
    # pdb.set_trace()

if __name__ == '__main__':
    main()