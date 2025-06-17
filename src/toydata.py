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

from src.nplot import plot_regression_data, plot_binary_classification_data
from src.data import JAXDataset, NumpyDataset, get_dataloaders as _get_dataloaders, jax_collate_fn, numpy_collate_fn


"""DATASETS"""

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

def spiral_dataset(n, key, noise=0.05):
    assert n % 2 == 0, "n should be even so classes are balanced"
    n_per_class = n // 2
    k1, k2, k3 = jax.random.split(key, 3)

    # radii ∈ (0, 1]
    r = jax.random.uniform(k1, shape=(n_per_class, 1))
    # angles (each spiral winds 3/2 turns, i.e. up to 540°)
    theta = r * 3.0 * jnp.pi  # (n/2, 1)

    # class 0  …  θ
    x0 = jnp.concatenate([r * jnp.cos(theta), r * jnp.sin(theta)], axis=1)
    # class 1  …  θ + π  (rotated by 180°)
    x1 = jnp.concatenate([r * jnp.cos(theta + jnp.pi),
                          r * jnp.sin(theta + jnp.pi)], axis=1)

    # add i.i.d. Gaussian noise
    x0 += noise * jax.random.normal(k2, shape=x0.shape)
    x1 += noise * jax.random.normal(k3, shape=x1.shape)

    x = jnp.concatenate([x0, x1], axis=0).astype(jnp.float32)
    y = jnp.concatenate([jnp.zeros(n_per_class), jnp.ones(n_per_class)]
                        ).astype(jnp.float32)

    # random permutation so minibatches are mixed
    perm = jax.random.permutation(jax.random.fold_in(key, 42), n)
    return x[perm], y[perm]


def noisy_spiral_dataset(n, key, noise=0.05):
    assert n % 2 == 0, "n should be even so classes are balanced"
    n_per = n // 2
    k1, k2, k3 = jax.random.split(key, 3)
    # radii ∈ (0,1], angles up to 3π
    r     = jax.random.uniform(k1, (n_per,1))
    theta = r * 3.0 * jnp.pi
    x0    = jnp.concatenate([r*jnp.cos(theta),   r*jnp.sin(theta)], axis=1)
    x1    = jnp.concatenate([r*jnp.cos(theta + jnp.pi),
                             r*jnp.sin(theta + jnp.pi)], axis=1)
    x0   += noise * jax.random.normal(k2, x0.shape)
    x1   += noise * jax.random.normal(k3, x1.shape)
    x     = jnp.vstack([x0, x1]).astype(jnp.float32)
    y     = jnp.concatenate([jnp.zeros(n_per), jnp.ones(n_per)]).astype(jnp.int32)
    return x, y

def ring_dataset(n, key, radius=1.05, width=0.15, noise=0.02):
    """
    Uniformly sample n points in the annulus [radius, radius+width],
    add small Gaussian noise, and assign random labels.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    r      = radius + jax.random.uniform(k1, (n,1)) * width
    theta  = jax.random.uniform(k2, (n,1)) * 2.0 * jnp.pi
    x_ring = jnp.concatenate([r*jnp.cos(theta), r*jnp.sin(theta)], axis=1)
    x_ring += noise * jax.random.normal(k3, x_ring.shape)
    y_ring = jax.random.bernoulli(k3, p=0.5, shape=(n,)).astype(jnp.int32)
    return x_ring, y_ring

def xor_dataset(n, key, noise=0.05):
    zkey, noisekey = jax.random.split(key, 2)
    z = jax.random.uniform(key=zkey, shape=(n,2))
    x = (z > 0.5).astype(jnp.float32) #* 2 - 1
    y = (x.sum(axis=1) == 1).astype(jnp.float32).squeeze()
    x += noise * jax.random.normal(key=noisekey, shape=z.shape)
    return x, y

def banana_dataset(n, key, noise=0.05):
    x0key, x1key, noisekey = jax.random.split(key, 3)
    halfn = n // 2
    
    # generate the blue arch
    archn = int(halfn * 0.8)
    
    x01 = jax.random.uniform(x0key, shape=(archn, 1), minval=-1., maxval=1.)
    x02 = jnp.cos(1.5*x01)
    x02 -= .7
    x0arch = jnp.concat([x01,x02], axis=1)
    x0arch += jax.random.normal(key=noisekey, shape=x0arch.shape) * noise
    
    # generate the blue line
    x0key = jax.random.fold_in(x0key, 1)
    noisekey = jax.random.fold_in(noisekey, 1)
    linen = halfn - archn
    x01 = jax.random.uniform(key=x0key, shape=(linen, 1), minval=0., maxval=1.)
    x02 = 1.5 - x01 * 0.2
    x0line = jnp.concat([x01,x02], axis=1)
    x0line += jax.random.normal(key=noisekey, shape=x0line.shape) * noise
    
    y0 = jnp.ones(halfn)
    
    # generate the red arch
    archn = int(halfn * 0.6)
    
    x11 = jax.random.uniform(x1key, shape=(archn, 1), minval=-1.0, maxval=1.1)
    x12 = jnp.cos(1.6*x11)
    x12 -= 0#.25
    x1arch = jnp.concat([x11,x12], axis=1)
    x1arch += jax.random.normal(key=noisekey, shape=x1arch.shape) * noise
    
    # generate the red blob
    x1key = jax.random.fold_in(x1key, 1)
    nkey1,nkey2 = jax.random.split(noisekey, 2)
    blobn = halfn - archn
    x11 = jax.random.uniform(key=x1key, shape=(blobn, 1), minval=-.4, maxval=.5)
    x12 = jax.random.normal(nkey1, x11.shape) * noise * 1.5 - 0.5
    x1blob = jnp.concat([x11,x12], axis=1)
    x1blob += jax.random.normal(key=nkey2, shape=x1blob.shape) * noise
    
    y1 = jnp.zeros(halfn)
    
    x = jnp.concat([x0arch, x0line, x1arch, x1blob], axis=0)
    y = jnp.concat([y0, y1])
    
    randperm = jax.random.permutation(jax.random.fold_in(key, 1337), n)
    
    return x[randperm], y[randperm]


def data_ex5():
    data = jnp.load('data/data_exercise5.npz')
    x, y = data['X'], data['y']
    return x, y

def data_mnist_subset_89():
    """MNIST subset, top 2 PCA components."""
    data = jnp.load('data/mnist_subset_89.npz')
    Xtrain = data['Xtrain']
    Xtest = data['Xtest']
    ytrain = data['ytrain']
    ytest = data['ytest']
    X = jnp.vstack([Xtrain, Xtest])
    y = jnp.hstack([ytrain, ytest])    
    return X,y



"""TINY PLOTTING UTILS"""

    
def plot_data(x,y,name,plotf):
    import matplotlib.pyplot as plt
    from seaborn import set_style
    set_style('darkgrid')
    plotf(x,y)
    plt.legend()
    plt.title(f"{name} dataset")
    plt.tight_layout()
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
    elif dataset_name == 'banana':
        x, y = banana_dataset(n, key, noise)
        plot_data(x,y,dataset_name,plot_binary_classification_data)
    elif dataset_name == 'spiral':
        # reserve 10% of n for the ring (validation)
        n_val      = int(0.00 * n)
        n_spiral   = n - n_val
        k1, k2     = jax.random.split(key, 2)

        # build spiral for the first (n - n_val) pts
        x_sp, y_sp = noisy_spiral_dataset(n_spiral, k1, noise)
        # build ring for the last n_val pts
        x_rg, y_rg = ring_dataset(n_val,       k2,
                                  radius=1.05,
                                  width=0.15,
                                  noise=noise)

        # concatenate *without* shuffling so that the last 10% are ring
        x = jnp.concatenate([x_sp, x_rg], axis=0)
        y = jnp.concatenate([y_sp, y_rg], axis=0)
        plot_data(x, y, dataset_name, plot_binary_classification_data)
    elif dataset_name == 'ring':
        x, y = ring_dataset(n,
                                  key,
                                  radius=2.,
                                  width=0.15,
                                  noise=noise)
        plot_data(x,y,dataset_name,plot_binary_classification_data)
    elif dataset_name == 'sine':
        x, y = sine_wave_dataset(n, key, noise, split_in_middle=split_in_middle)
        plot_data(x,y,dataset_name,plot_regression_data)
    else:
        raise ValueError(f"Unknown dataset_name = {dataset_name}")
    return x, y


def load_toydata(dataset):
    datafile = f"data/{dataset}.npz"
    if not os.path.exists(datafile):
        raise FileNotFoundError(f"Data file not found: {datafile}")
    data_npz = np.load(datafile)
    x = jax.device_put(data_npz["x"])
    y = jax.device_put(data_npz["y"])
    n_samples = x.shape[0]

    # Create train/test/val split
    trainsplit = int(0.8 * n_samples)
    testsplit  = trainsplit + int(0.10 * n_samples)
    xtrain, ytrain = x[:trainsplit], y[:trainsplit]
    xtest,  ytest  = x[trainsplit:testsplit], y[trainsplit:testsplit]
    xval,   yval  = x[testsplit:], y[testsplit:]
    return (xtrain,ytrain), (xtest,ytest), (xval,yval)


def get_dataloaders(dataset, batch_size):
    (xtrain,ytrain), (xtest,ytest), (xval,yval) = load_toydata(dataset)
    
    train_dataset = JAXDataset(xtrain, ytrain)
    test_dataset  = JAXDataset(xtest,  ytest)
    val_dataset  = JAXDataset(xval,  yval)
    train_loader, test_loader, val_loader = _get_dataloaders(train_dataset, test_dataset, val_dataset, batch_size, collate_fn=jax_collate_fn)
    
    return train_loader, test_loader, val_loader


################################################################################
#                                MAIN
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Utility for creating synthetic toy datasets.")
    parser.add_argument("--dataset", type=str, #choices=["sine", "xor", "banana"],
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
