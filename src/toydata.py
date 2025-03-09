"""
Utility for creating or loading toy datasets.

Example usage:
  python toydata.py --dataset sine --n_samples 200 --noise 0.3 --split_in_middle \
                    --seed 999 --out_file data/sine_dataset.npz
"""

import argparse
import os
import jax
import jax.numpy as jnp
import numpy as np
import torch.utils.data as data

"""DATASET CLASSES AND UTILS"""

class JAXDataset(data.Dataset):
    def __init__(self, x, y):
        self.x = jnp.array(x, dtype=jnp.float32)
        self.y = jnp.array(y, dtype=jnp.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx] 

def jax_collate_fn(batch):
    """Ensures that the batch remains a JAX array instead of PyTorch tensors."""
    x_batch, y_batch = zip(*batch)  # Unzip data points
    x_batch = jnp.stack(x_batch)
    y_batch = jnp.stack(y_batch)
    return x_batch, y_batch


# todo not completely sure if it works!
def numpy_collate_fn(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate_fn(samples) for samples in transposed]
    else:
        return np.array(batch)


def get_dataloaders(train_dataset, test_dataset, batch_size, collate_fn=numpy_collate_fn):
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader


"""SYNTHETIC DATASETS"""

sine_wave_fun = lambda x: jnp.sin(2 * x) + x * jnp.cos(5 * x)

def sine_wave_dataset(n, key, noise=0.5, split_in_middle=False):
    datakey, noisekey = jax.random.split(key, 2)
    if not split_in_middle:
        x = jax.random.uniform(key=datakey, minval=-4.0, maxval=3.0, shape=(n,)).reshape(-1, 1)
    else:
        # Example of splitting the domain into two portions and merging them
        x1 = jax.random.uniform(key=datakey, minval=-4.0, maxval=-1.0, shape=(n//2,)).reshape(-1, 1)
        x2 = jax.random.uniform(key=datakey, minval=0.0, maxval=3.0, shape=(n//2,)).reshape(-1, 1)
        x = jnp.concatenate([x1,x2], axis=0)
        perm = jax.random.permutation(datakey, x.shape[0])
        x = x[perm]
    signal = sine_wave_fun(x)
    y = signal + jax.random.normal(noisekey, shape=signal.shape) * noise
    return x, y

def xor_dataset(n, key, noise=0.05):
    zkey, noisekey = jax.random.split(key, 2)
    z = jax.random.uniform(key=zkey, shape=(n,2))
    x = (z > 0.5).astype(jnp.float32) #* 2 - 1
    y = (x.sum(axis=1) == 1).astype(jnp.float32).squeeze()
    x += noise * jax.random.normal(key=noisekey, shape=z.shape)
    return x, y

def data_ex5():
    data = jnp.load('data/data_exercise5.npz')
    x, y = data['X'], data['y']
    return x, y

def data_mnist_subset_89():
    data = jnp.load('data/mnist_subset_89.npz')
    Xtrain = data['Xtrain']
    Xtest = data['Xtest']
    ytrain = data['ytrain']
    ytest = data['ytest']
    X = jnp.vstack([Xtrain, Xtest])
    y = jnp.hstack([ytrain, ytest])    
    return X,y


def plot_regression_data(x,y):
    from nplot import scatterp
    scatterp(x, y, label='data')

def plot_binary_classification_data(x,y):
    from nplot import scatterp
    scatterp(*x[y==0].T, label='Class 0', color='salmon', zorder=2)
    scatterp(*x[y==1].T, label='Class 1', zorder=2)
    
def plot_data(x,y,name,plotf):
    import matplotlib.pyplot as plt
    from seaborn import set_style
    set_style('darkgrid')
    plotf(x,y)
    plt.legend()
    plt.title(f"{name} dataset")
    os.makedirs("fig", exist_ok=True)
    plt.savefig(f"fig/{name}.pdf")


"""DATA FACTORY"""
def create_dataset(dataset_name, n, key, noise, split_in_middle=False):
    """
    Creates a dataset of size n according to dataset_name, 
    optionally with added noise. Returns (x, y).
    """
    if dataset_name == 'xor':
        x, y = xor_dataset(n, key, noise)
        plot_data(x,y,dataset_name,plot_binary_classification_data)
    elif dataset_name == 'sine':
        x, y = sine_wave_dataset(n, key, noise, split_in_middle=split_in_middle)
        plot_data(x,y,dataset_name,plot_regression_data)
    else:
        raise ValueError(f"Unknown dataset_name = {dataset_name}")
    return x, y


################################################################################
#                                MAIN
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Utility for creating synthetic toy datasets.")
    parser.add_argument("--dataset", type=str, default="sine", choices=["sine", "xor"],
                        help="Which dataset to create.")
    parser.add_argument("--n_samples", type=int, default=128,
                        help="Number of data samples to generate.")
    parser.add_argument("--noise", type=float, default=0.5,
                        help="Noise level (scale) for the dataset.")
    parser.add_argument("--split_in_middle", action="store_true",
                        help="If set, use the 'split' variant for the sine dataset.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data generation.")
    parser.add_argument("--out_file", type=str, default=None,
                        help="Where to save the resulting .npz file (contains x and y).")
    args = parser.parse_args()

    # 1) PRNG initialization
    rng_data = jax.random.PRNGKey(args.seed)
    
    # 2) Create the dataset
    x, y = create_dataset(
        dataset_name=args.dataset,
        n=args.n_samples,
        key=rng_data,
        noise=args.noise,
        split_in_middle=args.split_in_middle
    )
    
    # 3) Save as .npz
    savename = args.out_file or f"data/{args.dataset}.npz"
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    np.savez(savename, x=x, y=y)
    print(f"Saved {args.dataset} data at {savename} with shape x={x.shape}, y={y.shape}")

if __name__ == "__main__":
    main()
