"""
Utility for creating or loading toy datasets.
"""

import jax
import jax.numpy as jnp
import torch.utils.data as data

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

def get_dataloaders(train_dataset, test_dataset, batch_size, collate_fn=jax_collate_fn):
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader,test_loader



"""SYNTHETIC DATA"""
sine_wave_fun = lambda x: jnp.sin(2 * x) + x * jnp.cos(5 * x)

def sine_wave_dataset(n, key, noise=0.5, split_in_middle=False):
    datakey, noisekey = jax.random.split(key, 2)
    if not split_in_middle:
        x = jax.random.uniform(key=datakey, minval=-3.0, maxval=3.0, shape=(n,)).reshape(-1, 1)
    else:
        x1 = jax.random.uniform(key=datakey, minval=-4.0, maxval=-1.0, shape=(n,)).reshape(-1, 1)
        x2 = jax.random.uniform(key=datakey, minval=0.0, maxval=3.0, shape=(n,)).reshape(-1, 1)
        x = jnp.concatenate([x1,x2],axis=0)
        perm = jax.random.permutation(key, x.shape[0])
        x = x[perm]
    signal = sine_wave_fun(x)
    y = signal + jax.random.normal(noisekey, shape=signal.shape) * noise
    return x, y


def xor_dataset(n, key, noise=0.05):
    zkey,noisekey = jax.random.split(key, 2)
    z = jax.random.uniform(key=zkey, shape=(n,2))
    x = (z > 0.5).astype(jnp.float32) * 2 - 1
    y = (x.prod(axis=1) == -1).astype(jnp.float32)
    x += noise * jax.random.normal(key=noisekey, shape=z.shape)
    return x, y
    
    


"""DATASETS"""
def data_ex5():
    data = jnp.load('data/data_exercise5.npz')
    x, y = data['X'], data['y']
    return x, y