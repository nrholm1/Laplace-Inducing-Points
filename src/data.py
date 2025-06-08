import flax
import flax.jax_utils
import jax
import jax.numpy as jnp
import numpy as np
import torch.utils.data as data
from flax import jax_utils
from collections import deque
from typing import Iterable, Iterator, Any, Deque


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


# # todo not completely sure if it works!
def numpy_collate_fn_old(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate_fn(samples) for samples in transposed]
    else:
        return np.array(batch)

class NumpyDataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x  # still a plain np.ndarray
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

def numpy_collate_fn(batch):
    xs, ys = zip(*batch)
    return np.stack(xs), np.stack(ys)


def get_dataloaders(train_dataset, 
                    test_dataset, 
                    val_dataset=None, 
                    batch_size=32,
                    collate_fn=numpy_collate_fn_old):
    train_loader = data.DataLoader(train_dataset, 
                                   batch_size=batch_size, 
                                   shuffle=True, 
                                   collate_fn=collate_fn,
                                   drop_last=True)
    test_loader = data.DataLoader(test_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=False, 
                                  collate_fn=collate_fn, 
                                  drop_last=True)
    if val_dataset is None:
        return train_loader, test_loader
    val_loader = data.DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                collate_fn=collate_fn, 
                                drop_last=True)
    return train_loader, test_loader, val_loader




def _fifo_prefetch(it: Iterable[Any], *, size: int) -> Iterator[Any]:
    """
    Keep `size` future elements from `it` parked on the current default device.

    Works for a single or multi-GPU host, but **does not shard** – each yielded
    item is copied to exactly one device.
    """
    dev   = jax.devices()[0]               # choose first visible device
    buf: Deque[Any] = deque()

    it = iter(it)
    try:                                   # prime the buffer once
        for _ in range(size):
            buf.append(jax.device_put(next(it), device=dev))
    except StopIteration:
        pass

    while buf:                             # main loop
        yield buf[0]                       # serve oldest prefetched batch
        try:                               # enqueue one more if available
            buf.append(jax.device_put(next(it), device=dev))
        except StopIteration:
            pass
        buf.popleft()

def make_iter(loader, *, prefetch: int = 2):
    """
    Turn a PyTorch `DataLoader` into an iterator of **DeviceArrays** with
    `prefetch` batches already waiting on the accelerator.
    """
    to_jnp = lambda b: jax.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), b)
    return _fifo_prefetch((to_jnp(b) for b in loader), size=prefetch)
