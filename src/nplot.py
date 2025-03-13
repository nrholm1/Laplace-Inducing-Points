"""
Utility for making nice, homogenous plots.
"""

import jax.numpy as jnp
from flax.linen import softmax
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from seaborn import set_style
set_style('darkgrid')

class Colors(str, Enum):
    paleblue = '#8888FF'
    deepblue = '#375E97'
    darkorange = '#FB6542'
    yellow = '#FFBB00'
    darkgray = '#333'
    
def plot_bc_heatmap(fig, ax, map_model_state, tmin, tmax, sharp_boundary=False):
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", [Colors.paleblue, 'salmon'])
    t = jnp.linspace(tmin, tmax, 100)
    X,Y = jnp.meshgrid(t, t, indexing='ij')
    model_inputs = jnp.stack([X, Y], axis=-1)
    
    logits = map_model_state.apply_fn(map_model_state.params, model_inputs)
    preds = softmax(logits, axis=-1)[:,:,0]
    # co = plt.contourf(X, Y, preds, levels=100, cmap=cmap, alpha=0.5)
    co = plt.contourf(X, Y, preds, levels=100, cmap=cmap, alpha=1.)
    
    cbar = fig.colorbar(co, ax=ax)
    cbar.set_label(r"$θ_{MAP}$ prediction probability")
    
    
def plot_bc_boundary_contour(map_model_state, tmin, tmax, alpha=0.2, color="black",zorder=5, label=None):
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", [color, color])
    # levels = [0.5]
    t = jnp.linspace(tmin, tmax, 100)
    X, Y = jnp.meshgrid(t, t)
    points = jnp.vstack([X.ravel(), Y.ravel()]).T
    score = softmax(
        map_model_state.apply_fn(map_model_state.params, points),
        axis=-1
    )[:, 1].reshape(X.shape)
    plt.contour(X, Y, score, levels=1, cmap=cmap, zorder=zorder, alpha=alpha)
    if label is not None:
        plt.plot(float('nan'), color=color, label=label)


scatterp = lambda x,y,*args, color=Colors.paleblue, **kwargs: plt.scatter(x, y, edgecolor=Colors.darkgray, color=color, *args, **kwargs)
linep    = lambda x,y,*args, color=Colors.paleblue, **kwargs: plt.plot(x, y, color=color, linewidth=3, *args, **kwargs)

def plot_inducing_points_1D(ax, points, *args,
                            offsetp=0.1,
                            color='red', label='Inducing points', marker='X',
                            **kwargs):
    ymin, ymax = ax.get_ylim()
    offset = jnp.ceil(ymax + offsetp * (ymax - ymin))  # a little (offsetp amount) above the top

    ax.scatter(points, jnp.full_like(points, offset), 
            color=color, marker=marker, label=label, edgecolor=Colors.darkgray, *args, **kwargs)


def plot_cinterval(x, mu, sigma, color='orange', *args, zorder=1, text=None, **kwargs):
    """Plot 2 std. deviations out from a mean."""
    label = r"$2\sigma$"
    text = f" ({text})" if text is not None else ""
    label = f"{label}{text}"
    plt.fill_between(
        x,
        mu - 2 * sigma,
        mu + 2 * sigma,
        alpha=0.3,
        color=color,
        label=label,
        zorder=-1
    )
    linep(x, mu - 2 * sigma,color=color,linestyle='--', zorder=zorder)
    linep(x, mu + 2 * sigma,color=color,linestyle='--', zorder=zorder)