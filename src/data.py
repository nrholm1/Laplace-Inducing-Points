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


# # todo not completely sure if it works!
# def numpy_collate_fn(batch):
#     if isinstance(batch[0], np.ndarray):
#         return np.stack(batch)
#     elif isinstance(batch[0], (tuple,list)):
#         transposed = zip(*batch)
#         return [numpy_collate_fn(samples) for samples in transposed]
#     else:
#         return np.array(batch)

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


def get_dataloaders(train_dataset, test_dataset, batch_size, num_workers=1, collate_fn=numpy_collate_fn):
    train_loader = data.DataLoader(train_dataset, 
                                   batch_size=batch_size, 
                                   shuffle=True, 
                                   collate_fn=collate_fn,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=True)
    test_loader = data.DataLoader(test_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=False, 
                                  collate_fn=collate_fn, 
                                   num_workers=num_workers,
                                   pin_memory=True,
                                  drop_last=True)
    return train_loader, test_loader


def prefetch(loader, size):
    it = iter(loader)
    buf = []
    for _ in range(size):
        try:
            buf.append(jax.device_put(next(it)))
        except StopIteration:
            break
    for batch in it:
        yield buf.pop(0)
        buf.append(jax.device_put(batch))
    while buf:
        yield buf.pop(0)