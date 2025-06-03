import pdb
import jax
import jax.numpy as jnp
from flax import linen as nn

from src.toymodels import SimpleClassifier


class LeNet5(nn.Module):
    """LeNet-5 for MNIST / Fashion-MNIST (~60 k parameters)."""
    @nn.compact
    def __call__(self, x):
        # Ensure shape (batch, 28, 28, 1)
        if x.ndim == 3:                 # missing batch dim
            x = x[None, ...]
            
        # Pad to 32×32 to reproduce original LeNet geometry
        x = jnp.pad(x,                    # N H  W  C
                    ((0, 0), (2, 2), (2, 2), (0, 0)),
                    mode="constant")

        # C1: 6 × 5×5 conv, valid padding
        x = nn.Conv(features=6, kernel_size=(5, 5), strides=(1, 1),
                    padding="VALID")(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # C3: 16 × 5×5 conv
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(1, 1),
                    padding="VALID")(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten: 16 × 5 × 5 = 400
        x = x.reshape((x.shape[0], -1))

        # F5: 120-unit fully connected
        x = nn.Dense(features=120)(x)
        x = nn.relu(x)

        # F6: 84-unit fully connected
        x = nn.Dense(features=84)(x)
        x = nn.relu(x)

        # Output: 10 logits
        x = nn.Dense(features=10)(x)
        return x


class LargeClassifier(nn.Module):
    input_shape: tuple # shape of a single input
    numh: list # list of number of hidden units per layer
    numl: int # number of layers
    numc: int # number of classes
    
    @nn.compact
    def __call__(self, X):
        if X.shape == self.input_shape:
            X = X.reshape(-1)
        else:
            X = X.reshape(X.shape[0], -1)
        for j in range(self.numl):
            X = nn.tanh(nn.Dense(features=self.numh[j])(X))
        logits = nn.Dense(features=self.numc)(X)
        return logits


def get_model(model_cfg):
    model_name = model_cfg['name']
    
    if model_name == "LeNet5":
        return LeNet5()
    elif model_name == "large_classifier":
        input_shape = tuple(model_cfg["input_shape"])
        num_h = model_cfg["num_h"]
        num_l = model_cfg["num_l"]
        num_c = model_cfg.get("num_c")
        return LargeClassifier(input_shape=input_shape, numh=num_h, numl=num_l, numc=num_c)
    elif model_name == "classifier":
        num_h = model_cfg["num_h"]
        num_l = model_cfg["num_l"]
        num_c = model_cfg.get("num_c")
        return SimpleClassifier(numh=num_h, numl=num_l, numc=num_c)