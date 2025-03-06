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
            X = nn.gelu(
                nn.Dense(features = self.numh)(X)
            )
        X = nn.Dense(features = self.numc)(X)
        return nn.softmax(X, axis=1)
    
    
count_model_params = lambda params: sum(x.size for x in jax.tree_util.tree_leaves(params))
def print_summary(params):
    num_model_params = count_model_params(params)    
    print(f"Param count     (D) : {num_model_params}")
    print(f"Cov. mat. size (D^2): {num_model_params**2:.3e}")