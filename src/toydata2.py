#%%
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

#%%
def three_moons(n_samples, seed, *, noise=0.1):
    import numpy as np
    np.random.seed(seed)
    x0,y0 = make_moons(2*n_samples//3, noise=noise)
    x1,y1 = make_moons(2*n_samples//3, noise=noise)
    x2 = x1[y1 == 1] + jnp.array([1.,1.])
    y2 = y1[y1 == 1]
    y2 = y2*2
    return jnp.concatenate([x0,x2]), jnp.concatenate([y0, y2])



if __name__ == '__main__':
    n_samples = 100
    noise = 0.1

    x,y = three_moons(n_samples, noise=noise)
    plt.scatter(*x[y==0].T)
    plt.scatter(*x[y==1].T)
    plt.scatter(*x[y==2].T)