import pdb
from flax import linen as nn
import jax

class SimpleRegressor(nn.Module):
    numh: int  # number of hidden units per layer
    numl: int  # number of layers

    @nn.compact
    def __call__(self, X, return_logvar: bool = True):
        for _ in range(self.numl):
            X = nn.gelu(nn.Dense(features=self.numh)(X))
        mu = nn.Dense(features=1)(X)
        if return_logvar:
            zeros_key = jax.random.PRNGKey(0)
            logvar = self.variable(
                col="logvar",
                name="logvar",
                init_fn=nn.initializers.zeros,
                key=zeros_key,
                shape=()
            )
            return mu, logvar.value
        else:
            return mu
    
    
class SimpleClassifier(nn.Module):
    numh: int # number of hidden units per layer
    numl: int # number of layers
    numc: int # number of classes
    
    @nn.compact
    def __call__(self, X):
        for _ in range(self.numl):
            X = nn.tanh(nn.Dense(features=self.numh)(X))
        logits = nn.Dense(features=self.numc)(X)
        return logits
    

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Input shape: (batch, 28, 28, 1)
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # -> (batch, 3136)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x