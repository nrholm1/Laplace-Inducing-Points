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
            logvar = self.param('logvar', nn.initializers.zeros, ())
            return mu, logvar
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