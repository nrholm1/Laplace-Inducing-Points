"""
Utility for making nice, homogenous plots.
"""

import os
import pdb
import jax
import jax.tree_util
import jax.numpy as jnp
from flax.linen import softmax
from enum import Enum
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

from src.lla import predict_la_samples_dense, predict_lla_dense, predict_lla_scalable

sns.set_style('darkgrid')
mpl.rcParams.update({
    "text.usetex": True,                     # hand off all text to LaTeX
    "font.family": "serif",                  # use a serif face
    "font.serif": ["Computer Modern Roman"], # explicitly point to the CM font
    "text.latex.preamble":
        r"\usepackage[T1]{fontenc}"+         # proper font encoding
        r"\usepackage{lmodern}" +            # if your doc uses Latin Modern
        r"\usepackage{amsmath}"             # for \text{…} inside math
    ,
    "pdf.fonttype": 42,                      # embed Type-42 (TrueType) fonts in PDF
    "ps.fonttype": 42,
    "font.size": 15, #20
})

class Colors(str, Enum):
    paleblue = '#8888FF'
    deepblue = '#375E97'
    darkorange = '#FB6542'
    yellow = '#FFBB00'
    darkgray = '#333'


def get_palette():
    return sns.color_palette("icefire", as_cmap=True)
    # return sns.diverging_palette(250, 0, center="light",  as_cmap=True, s=200, l=35)

def plot_regression_data(x,y):
    scatterp(x, y, label='data')

def plot_binary_classification_data(x,y,ax=plt):
    scatterp(*x[y==0].T, label='Class 0', color='salmon', zorder=2, ax=ax)
    scatterp(*x[y==1].T, label='Class 1', zorder=2, ax=ax)


def plot_lla_2D_classification(
        fig,
        ax,
        state,
        Xtrain,
        ytrain,
        Z,
        alpha,
        matrix_free: bool,
        num_mc_samples: int,
        mode: str,
        plot_Z: bool = False,
        plot_X: bool = False, 
    ):
    assert mode in {"ip_lla", "full_lla"}, "Please select a mode. Options = [ip, full]"
    
    N = Xtrain.shape[0]
    tmin, tmax = Xtrain.min() - 1.0, Xtrain.max() + 1.0
    t = jnp.linspace(tmin, tmax, 150)
    X, Y = jnp.meshgrid(t, t, indexing="ij")
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    key = jax.random.PRNGKey(0) # todo handle?
    
    if matrix_free:
        logit_samples = predict_lla_scalable(
            state,
            pts,
            Xtrain if mode=="full_lla" else Z,
            model_type="classifier",
            alpha=alpha,
            full_set_size=N,
            num_samples=num_mc_samples
        )
    else:
        logit_dist = predict_lla_dense(
            state, 
            pts,
            Xtrain if mode=="full_lla" else Z,
            model_type="classifier",
            alpha=alpha,
            full_set_size=N
        )
        logit_samples = logit_dist.sample(seed=key, sample_shape=(num_mc_samples,))
        # logit_samples = logit_samples.at[jnp.isnan(logit_samples)].set(0)
    
    prob_samples  = jax.nn.softmax(logit_samples, axis=-1)[:,:,0]

    """Plot empirical Mean"""
    mean_probs = prob_samples.mean(0)
    Z1 = mean_probs.reshape(X.shape)
    cmap1 = get_palette()
    vmin1, vmax1 = 0.0, 1.0
    norm1 = mpl.colors.Normalize(vmin1, vmax1)
    cf1 = ax[0].pcolormesh(
        X, Y, Z1,
        cmap=cmap1,
        norm=norm1,
        rasterized=True
    )
    cbar1 = fig.colorbar(cf1, ax=ax[0], label=r"$\text{E}[\mathbf{y^*|x^*},\mathcal{D}]$", location="left")
    cbar1.set_ticks(jnp.linspace(vmin1, vmax1, 2))  # nice round ticks
    cbar1.ax.yaxis.set_ticks_position('right')
    cbar1.ax.yaxis.set_label_position('left')
    ax[0].set_title("LLA predictive mean")
    cbars = [cbar1]

    """Plot empirical Variance"""
    var_probs = prob_samples.var(0)
    Z2 = var_probs.reshape(X.shape)
    cmap2  = mpl.colors.LinearSegmentedColormap.from_list("bwr", ["white", "black"])
    vmin2, vmax2 = 0.0, jnp.round(Z2.max(),2)
    norm2 = mpl.colors.Normalize(vmin2, vmax2)
    cf2 = ax[1].pcolormesh(
        X, Y, Z2,
        cmap=cmap2,
        norm=norm2,
        rasterized=True
    )
    cbar2 = fig.colorbar(cf2, ax=ax[1], label=r"$\text{V}[\mathbf{y^*|x^*},\mathcal{D}]$", location="left")
    cbar2.set_ticks(jnp.linspace(vmin2, vmax2, 2))
    cbar2.ax.yaxis.set_ticks_position('right')
    cbar2.ax.yaxis.set_label_position('left')
    ax[1].set_title("LLA predictive variance")
    cbars.append(cbar2)


    for cbari in cbars:
        axi = cbari.ax
        for spine in ('top','bottom','left','right'):
            axi.spines[spine].set_visible(True)
            axi.spines[spine].set_linewidth(2.0)
            axi.spines[spine].set_color('#333')

    for axi in ax:
        axi.set_xticks([])
        axi.set_yticks([])
        axi.set_xlabel(r"$x_1$")
        axi.set_ylabel(r"$x_2$")
        for spine in ('top','bottom','left','right'):
            axi.spines[spine].set_visible(True)
            axi.spines[spine].set_linewidth(1.0)
            axi.spines[spine].set_color('#333')
    

    if plot_Z: scatterp(*Z.T, color="yellow", zorder=8, marker="X", label="Inducing points", ax=ax[0])
    if plot_Z: scatterp(*Z.T, color="yellow", zorder=8, marker="X", label="Inducing points", ax=ax[1])
    
    if plot_X: plot_binary_classification_data(Xtrain, ytrain, ax=ax[0])
    if plot_X: plot_binary_classification_data(Xtrain, ytrain, ax=ax[1])
        
    if plot_Z or plot_X: ax[0].legend(loc="lower right", framealpha=1.0)
    if plot_Z or plot_X: ax[1].legend(loc="lower right", framealpha=1.0)
    # pdb.set_trace()


def plot_map_2D_classification(fig, ax, map_model_state, tmin, tmax, colorbar=True):
    cmap = get_palette()
    t = jnp.linspace(tmin, tmax, 150)
    X,Y = jnp.meshgrid(t, t, indexing='ij')
    model_inputs = jnp.stack([X, Y], axis=-1)
    
    logits = map_model_state.apply_fn(map_model_state.params, model_inputs)
    preds = softmax(logits, axis=-1)[:,:,0]
    # co = plt.contourf(X, Y, preds, levels=100, cmap=cmap, vmin=0., vmax=1.)
    co = ax.pcolormesh(X, Y, preds, cmap=cmap, vmin=0., vmax=1., alpha=1.0, rasterized=True)
    
    if colorbar:
        cbar = fig.colorbar(co, ax=ax)
        cbar.set_label(r"$\theta_{\mathrm{MAP}}$ prediction probability")
        
    return co


def plot_lla_mean(
        fig,
        ax,
        state,
        Xtrain,
        ytrain,
        alpha,
        num_mc_samples: int,
        plot_X: bool = False, 
    ):
    N = Xtrain.shape[0]
    tmin, tmax = Xtrain.min() - 1.0, Xtrain.max() + 1.0
    t = jnp.linspace(tmin, tmax, 150)
    X, Y = jnp.meshgrid(t, t, indexing="ij")
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    key = jax.random.PRNGKey(0) # todo handle?
    
    logit_dist = predict_lla_dense(
        state, 
        pts,
        Xtrain,
        model_type="classifier",
        alpha=alpha,
        full_set_size=N
    )
    logit_samples = logit_dist.sample(seed=key, sample_shape=(num_mc_samples,))

    prob_samples  = jax.nn.softmax(logit_samples, axis=-1)[:,:,0]

    """Plot empirical Mean"""
    mean_probs = prob_samples.mean(0)
    Z1 = mean_probs.reshape(X.shape)
    cmap = get_palette()
    vmin, vmax = 0.0, 1.0
    norm = mpl.colors.Normalize(vmin, vmax)
    cf = ax.pcolormesh(
        X, Y, Z1,
        cmap=cmap,
        norm=norm,
        rasterized=True
    )

    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$x_1$")
    # ax.set_ylabel(r"$x_2$") # todo uncomment!
    for spine in ('top','bottom','left','right'):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color('#333')
    
    if plot_X: 
        plot_binary_classification_data(Xtrain, ytrain, ax=ax)
        ax.legend(loc="lower right", framealpha=1.0)
    
    return fig

def plot_la_sampled_mean(ax, state, Xtrain, pts, norm, cmap, alpha):
    key = jax.random.PRNGKey(42)
    logit_samples = predict_la_samples_dense(
        map_state=state,
        Xnew=pts,            # (Ngrid, 2)
        Z=Xtrain,            # or inducing points if IP LA
        model_type="classifier",
        alpha=alpha,
        full_set_size=Xtrain.shape[0],
        num_mc_samples=50,
        key=key,
    )
    prob_samples = jax.nn.softmax(logit_samples, axis=-1)[..., 0]  
    mean_probs = prob_samples.mean(0)
    Zla = mean_probs.reshape(X.shape)
    ax.pcolormesh(X, Y, Zla, cmap=cmap, norm=norm, rasterized=True)


def make_predictive_mean_figure(state, Xtrain, ytrain, alpha, num_mc_samples=100):
    """
    Build the 1x3 figure:
      [ MAP ] [ LA-MC ] [ LLA ]
    and add one shared colorbar on the left.
    """
    tmin, tmax = Xtrain.min() - 1, Xtrain.max() + 1
    t = jnp.linspace(tmin, tmax, 150)
    Xg, Yg = jnp.meshgrid(t, t, indexing='ij')
    
    global X, Y, G
    X, Y = Xg, Yg
    G = Xg.shape[0]
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    cmap = get_palette()
    norm = mpl.colors.Normalize(0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(13,4),
                            sharex=True, 
                            # sharey=True,
                            constrained_layout=True
                            )

    plot_binary_classification_data(Xtrain, ytrain.squeeze(), axs[0])
    plot_binary_classification_data(Xtrain, ytrain.squeeze(), axs[1])
    plot_binary_classification_data(Xtrain, ytrain.squeeze(), axs[2])
    
    axs[0].set_title("NN MAP")
    im0 = plot_map_2D_classification(fig, axs[0], state, tmin, tmax, alpha, colorbar=False)

    axs[1].set_title("Without Linearization")
    im1 = plot_la_sampled_mean(axs[1], state, Xtrain, pts, norm, cmap, alpha)

    axs[2].set_title("With Linearization")
    im2 = plot_lla_mean(fig, axs[2],
                        state, Xtrain, ytrain,
                        alpha,
                        num_mc_samples,
                        plot_X=False)

    for ax in axs:
        ax.set_xlabel(r"$x_1$")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#333')
            spine.set_linewidth(1.0)
    axs[0].set_ylabel(r"$x_2$")

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axs,               
        location="left",
        label=r"$\mathrm{E}[y^* \mid x^*, \mathcal{D}]$",
    )
    
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')
    cbar.set_ticks(jnp.linspace(0, 1, 2))

    # fig.subplots_adjust(top=0.85)
    # fig.suptitle("Predictive mean", fontsize=16)

    return fig

    
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
    
def plot_grayscale(batch, step='', name=''):
    """
    Plot a batch of 32 MNIST digits (shape: [32, 28, 28]).
    """
    # convert JAX array to NumPy (for matplotlib)
    imgs = np.array(batch)
    assert imgs.shape == (32, 28, 28), f"Expected batch shape (32,28,28), got {batch.shape}"
    
    # create a 4x8 grid of subplots
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(8, 4),
                             gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    
    # plot each image
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i], cmap='gray', interpolation='nearest')
        ax.axis('off')
    
    # save to PDF
    fig.savefig(f'fig/test/{name}_{step}.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_color(batch, step='', name=''):
    """
    Plot a batch of 32 RGB images (shape: [32, H, W, 3]).
    """
    # convert JAX array (or any array‐like) to NumPy
    imgs = np.array(batch)
    assert imgs.ndim == 4 and imgs.shape[0] == 32 and imgs.shape[-1] == 3, (
        f"Expected batch shape (32, H, W, 3), got {imgs.shape}"
    )
    # compute grid size
    nrows, ncols = 4, 8
    
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols, nrows),
        gridspec_kw={'wspace': 0.1, 'hspace': 0.1}
    )
    
    for i, ax in enumerate(axes.flatten()):
        # clip to [0,1] or [0,255] depending on data range
        img = imgs[i]
        if img.max() > 1.0:
            img = img.astype(np.uint8)
        ax.imshow(img, interpolation='nearest')
        ax.axis('off')
    
    # ensure output directory exists
    out_path = f'fig/test/{name}_{step}.png'
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved color grid to {out_path}")