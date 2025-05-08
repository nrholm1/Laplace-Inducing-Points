"""
Utility for making nice, homogenous plots.
"""

import jax
import jax.tree_util
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
    

    
def plot_bc_heatmap(fig, ax, map_model_state, tmin, tmax, alpha=1.0):
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ['#111188', "white", "#881111"])
    t = jnp.linspace(tmin, tmax, 100)
    X,Y = jnp.meshgrid(t, t, indexing='ij')
    model_inputs = jnp.stack([X, Y], axis=-1)
    
    logits = map_model_state.apply_fn(map_model_state.params, model_inputs)
    preds = softmax(logits, axis=-1)[:,:,0]
    co = plt.contourf(X, Y, preds, levels=100, cmap=cmap, alpha=alpha, vmin=0., vmax=1.)
    
    cbar = fig.colorbar(co, ax=ax)
    cbar.set_label(r"$θ_{MAP}$ prediction probability")


def plot_heatmap_averaged(fig, ax, states, tmin, tmax, opacity=1.0, num_pts=100, show_variance=False, cbarlabel=None, **kwargs):
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ['#111188', "white", "#881111"])
      # 1) Build grid and flatten
    t = jnp.linspace(tmin, tmax, num_pts)
    X, Y = jnp.meshgrid(t, t, indexing="ij")
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=-1)  # (num_pts**2, 2)

    # 2) Prepare storage
    n_states = len(states)
    preds_accum = jnp.zeros((n_states, pts.shape[0]))

    # 3) Loop over parameter samples
    for i, state in enumerate(states):
        logits = state.apply_fn(state.params, pts)                # (N_pts², n_classes)
        probs  = softmax(logits, axis=-1)[:, 0]                  # (N_pts²,)
        preds_accum = preds_accum.at[i].set(probs)

    # 4) Compute mean (and variance if requested)
    mean_pred = preds_accum.mean(axis=0).reshape((num_pts, num_pts))

    # 5) Plot mean contour
    if cmap is None:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "blue_white_red", ["#111188", "white", "#881111"]
        )
    cf = ax.contourf(
        X, Y, mean_pred,
        levels=50,
        cmap=cmap,
        alpha=opacity,
        vmin=0.0, vmax=1.0,
        **kwargs
    )

    # 6) Optional: overlay variance
    if show_variance:
        # e.g. a lighter gray-scale for uncertainty
        var_pred  = preds_accum.var(axis=0).reshape((num_pts, num_pts))
        cvar = ax.contour(
            X, Y, var_pred,
            levels=10,
            colors="k",
            linewidths=0.5,
            linestyles="--"
        )
        ax.clabel(cvar, fmt="%.2f", fontsize=8)

    # 7) Colorbar
    if cbarlabel:
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(cbarlabel)
    
    
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


scatterp = lambda x,y,*args, ax=plt, color=Colors.paleblue, **kwargs: ax.scatter(x, y, edgecolor=Colors.darkgray, color=color, *args, **kwargs)
linep    = lambda x,y,*args, ax=plt, color=Colors.paleblue, **kwargs: ax.plot(x, y, color=color, linewidth=3, *args, **kwargs)

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