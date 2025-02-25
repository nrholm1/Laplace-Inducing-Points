"""
Utility for making nice, homogenous plots.
"""

import jax.numpy as jnp
from enum import Enum
import matplotlib.pyplot as plt
from seaborn import set_style
set_style('darkgrid')

class Colors(str, Enum):
    paleblue = '#8888FF'
    deepblue = '#375E97'
    darkorange = '#FB6542'
    yellow = '#FFBB00'
    darkgray = '#333'

scatterp = lambda x,y,*args, color=Colors.paleblue, **kwargs: plt.scatter(x, y, edgecolor=Colors.darkgray, color=color, *args, **kwargs)
linep    = lambda x,y,*args, color=Colors.paleblue, **kwargs: plt.plot(x, y, color=color, linewidth=3, *args, **kwargs)

def plot_inducing_points_1D(ax, points, *args,
                            offsetp=0.1,
                            color='red', label='Inducing points', marker='+',
                            **kwargs):
    ymin, ymax = ax.get_ylim()
    offset = jnp.ceil(ymax + offsetp * (ymax - ymin))  # a little (offsetp amount) above the top

    ax.scatter(points, jnp.full_like(points, offset), 
            color=color, marker=marker, label=label, *args, **kwargs)


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