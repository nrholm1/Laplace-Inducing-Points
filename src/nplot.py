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
# mpl.rcParams.update({
#     "text.usetex": True,                     # hand off all text to LaTeX
#     "font.family": "serif",                  # use a serif face
#     "font.serif": ["Computer Modern Roman"], # explicitly point to the CM font
#     "text.latex.preamble":
#         r"\usepackage[T1]{fontenc}"+         # proper font encoding
#         r"\usepackage{lmodern}" +            # if your doc uses Latin Modern
#         r"\usepackage{amsmath}"             # for \text{…} inside math
#     ,
#     "pdf.fonttype": 42,                      # embed Type-42 (TrueType) fonts in PDF
#     "ps.fonttype": 42,
#     # "font.size":  22, # 15
#     "font.size":  15
# })


def _make_grid(lo, hi, n):
    t = jnp.linspace(lo, hi, n)
    X, Y = jnp.meshgrid(t, t, indexing="ij")
    pts  = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    return X, Y, pts

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

def plot_binary_classification_data(x,y,ax=plt, c1='salmon', c2=Colors.paleblue):
    scatterp(*x[y==0].T, label='Class 0', color=c1, zorder=2, ax=ax)
    scatterp(*x[y==1].T, label='Class 1', color=c2, zorder=2, ax=ax)


def plot_lla_2D_classification_single(
    fig, ax, state, Xtrain, ytrain, Z, alpha,
    matrix_free: bool, num_mc_samples: int, mode: str, key,
    plot_Z: bool = False, plot_X: bool = False, cbar: bool = True,
    *, flat_params, unravel_fn,
):
    assert mode in {"ip_lla", "full_lla"}

    N = Xtrain.shape[0]
    lo, hi = Z.min() - 1.0, Z.max() + 1.0
    X, Y, pts = _make_grid(lo, hi, 64)

    if matrix_free:
        logit_samples = predict_lla_scalable(
            state,
            pts,
            Xtrain if mode == "full_lla" else Z,
            model_type="classifier",
            alpha=alpha,
            full_set_size=N,
            num_samples=num_mc_samples,
            flat_params=flat_params,
            unravel_fn=unravel_fn,
        )
    else:
        logit_dist = predict_lla_dense(
            state,
            pts,
            Xtrain if mode == "full_lla" else Z,
            model_type="classifier",
            alpha=alpha,
            full_set_size=N,
        )
        logit_samples = logit_dist.sample(seed=key, sample_shape=(num_mc_samples,))

    probs = jax.nn.softmax(logit_samples, axis=-1)[..., 0]
    mean_probs = probs.mean(0)
    Z1 = np.asarray(mean_probs.reshape(X.shape))  # matplotlib wants host arrays

    cf1 = ax.pcolormesh(np.asarray(X), np.asarray(Y), Z1,
                        cmap=get_palette(), norm=mpl.colors.Normalize(0.0, 1.0),
                        rasterized=True)
    if cbar:
        cbar1 = fig.colorbar(cf1, ax=ax, label=r"$\mathbb{E}[y^*|x^*,\mathcal{D}]$", location="left")
        cbar1.set_ticks([0.0, 1.0])
        cbar1.ax.yaxis.set_ticks_position('right')
        cbar1.ax.yaxis.set_label_position('left')
        ax.set_title("LLA predictive mean")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(1.0); s.set_color('#333')

    if plot_Z: scatterp(*Z.T, color="yellow", zorder=8, marker="X", label="Inducing points", ax=ax)
    if plot_X: plot_binary_classification_data(Xtrain, ytrain, ax=ax)

    
def plot_lla_2D_classification(
    fig, ax, state, Xtrain, ytrain, Z, alpha,
    matrix_free: bool, num_mc_samples: int, mode: str, key,
    plot_Z: bool = False, plot_X: bool = False, cbar: bool = True,
    *, flat_params, unravel_fn,
):
    assert mode in {"ip_lla", "full_lla"}

    N = Xtrain.shape[0]
    lo, hi = Xtrain.min() - 1.0, Xtrain.max() + 1.0
    X, Y, pts = _make_grid(lo, hi, 150)

    if matrix_free:
        logit_samples = predict_lla_scalable(
            state,
            pts,
            Xtrain if mode == "full_lla" else Z,
            model_type="classifier",
            alpha=alpha,
            full_set_size=N,
            num_samples=num_mc_samples,
            flat_params=flat_params,
            unravel_fn=unravel_fn,
        )
    else:
        logit_dist = predict_lla_dense(
            state, pts,
            Xtrain if mode == "full_lla" else Z,
            model_type="classifier",
            alpha=alpha,
            full_set_size=N,
            flat_params=flat_params,
            unravel_fn=unravel_fn,
        )
        logit_samples = logit_dist.sample(seed=key, sample_shape=(num_mc_samples,))

    probs = jax.nn.softmax(logit_samples, axis=-1)[..., 0]
    mean_probs = probs.mean(0).reshape(X.shape)
    var_probs  = probs.var(0).reshape(X.shape)

    Xh, Yh = np.asarray(X), np.asarray(Y)
    mean_h = np.asarray(mean_probs); var_h = np.asarray(var_probs)

    cf1 = ax[0].pcolormesh(Xh, Yh, mean_h, cmap=get_palette(),
                           norm=mpl.colors.Normalize(0.0, 1.0), rasterized=True)
    cf2 = ax[1].pcolormesh(Xh, Yh, var_h,
                           cmap=mpl.colors.LinearSegmentedColormap.from_list("bw", ["white","black"]),
                           norm=mpl.colors.Normalize(0.0, float(np.round(var_h.max(), 2)) or 1.0),
                           rasterized=True)

    if cbar:
        c1 = fig.colorbar(cf1, ax=ax[0], label=r"$\mathbb{E}[y^*|x^*,\mathcal{D}]$", location="left")
        c1.set_ticks([0.0, 1.0]); c1.ax.yaxis.set_ticks_position('right'); c1.ax.yaxis.set_label_position('left')
        c2 = fig.colorbar(cf2, ax=ax[1], label=r"$\mathrm{Var}[y^*|x^*,\mathcal{D}]$", location="left")
        c2.ax.yaxis.set_ticks_position('right'); c2.ax.yaxis.set_label_position('left')
        for c in (c1, c2):
            for s in c.ax.spines.values(): s.set_visible(True); s.set_linewidth(2.0); s.set_color('#333')

    for a in ax:
        a.set_xticks([]); a.set_yticks([])
        a.set_xlabel(r"$x_1$"); a.set_ylabel(r"$x_2$")
        for s in a.spines.values(): s.set_visible(True); s.set_linewidth(1.0); s.set_color('#333')

    if plot_Z:
        scatterp(*Z.T, color="yellow", zorder=8, marker="X", ax=ax[0])
        scatterp(*Z.T, color="yellow", zorder=8, marker="X", ax=ax[1])
    if plot_X:
        plot_binary_classification_data(Xtrain, ytrain, ax=ax[0])
        plot_binary_classification_data(Xtrain, ytrain, ax=ax[1])
    if plot_X:
        ax[0].legend(loc="lower right", framealpha=1.0)
        ax[1].legend(loc="lower right", framealpha=1.0)


def plot_map_2D_classification(fig, ax, map_model_state, tmin, tmax, colorbar=True):
    cmap = get_palette()
    t = jnp.linspace(tmin, tmax, 150)
    X,Y = jnp.meshgrid(t, t, indexing='ij')
    model_inputs = jnp.stack([X, Y], axis=-1)
    
    logits = map_model_state.apply_fn({'params': map_model_state.params}, model_inputs)
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

def plot_la_sampled_mean(ax, state, Xtrain, pts, norm, cmap, alpha, *, flat_params, unravel_fn):
    key = jax.random.PRNGKey(42)
    logit_samples = predict_la_samples_dense(
        map_state=state, Xnew=pts, Z=Xtrain, model_type="classifier",
        alpha=alpha, full_set_size=Xtrain.shape[0], num_mc_samples=50, key=key,
    )
    prob_samples = jax.nn.softmax(logit_samples, axis=-1)[..., 0]
    return prob_samples.mean(0)  # caller handles drawing



def make_predictive_mean_figure(state, Xtrain, ytrain, alpha, num_mc_samples=100, *, flat_params, unravel_fn):
    lo, hi = Xtrain.min() - 1, Xtrain.max() + 1
    X, Y, pts = _make_grid(lo, hi, 150)

    cmap = get_palette(); norm = mpl.colors.Normalize(0, 1)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharex=True, constrained_layout=True)

    for ax in axs: plot_binary_classification_data(Xtrain, ytrain.squeeze(), ax)

    axs[0].set_title("NN MAP")
    plot_map_2D_classification(fig, axs[0], state, lo, hi, colorbar=False)  # removed stray alpha arg

    axs[1].set_title("Without Linearization")
    mean_la = plot_la_sampled_mean(axs[1], state, Xtrain, pts, norm, cmap, alpha,
                                   flat_params=flat_params, unravel_fn=unravel_fn)
    axs[1].pcolormesh(np.asarray(X), np.asarray(Y), np.asarray(mean_la.reshape(X.shape)),
                      cmap=cmap, norm=norm, rasterized=True)

    axs[2].set_title("With Linearization")
    plot_lla_mean(fig, axs[2], state, Xtrain, ytrain, alpha, num_mc_samples, plot_X=False)

    for ax in axs:
        ax.set_xlabel(r"$x_1$"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(True); s.set_color('#333'); s.set_linewidth(1.0)
    axs[0].set_ylabel(r"$x_2$")

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, location="left",
                 label=r"$\mathrm{E}[y^* \mid x^*, \mathcal{D}]$").ax.yaxis.set_ticks_position('right')
    return fig


def make_predictive_mean_figure2(state, Xtrain, ytrain, alpha, num_mc_samples=100):
    """
    Build the 1x3 figure:
      [ MAP ] [ LLA ] [dataset]
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
    
    axs[1].set_title("NN MAP")
    im0 = plot_map_2D_classification(fig, axs[1], state, tmin, tmax, colorbar=False)

    axs[2].set_title("Full LLA")
    im2 = plot_lla_mean(fig, axs[2],
                        state, Xtrain, ytrain,
                        alpha,
                        num_mc_samples,
                        plot_X=False)
    
    axs[0].set_title("Dataset")
    axs[0].set_ylim(tmin, tmax)
    axs[0].legend(loc="lower center", ncols=2)

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


def make_comparison_figure(state, Xtrain, ytrain, Z, alpha, matrix_free, num_mc_samples=100):
    """
    Build the 2x1 figure:
      [ MEAN ]
      [ VAR  ]
    and optionally add one shared colorbar on the left.
    """

    M = Z.shape[0]

    # fig, axs = plt.subplots(1, 2, figsize=(12,5),
    fig, axs = plt.subplots(2, 1, figsize=(7,11),
    # fig, axs = plt.subplots(2, 1, figsize=(5.5,11),
                            sharex=True, 
                            sharey=True,
                            constrained_layout=True
                            )

    plot_lla_2D_classification(
        fig,
        axs,
        state,
        Xtrain,
        ytrain,
        Z,
        alpha,
        matrix_free,
        num_mc_samples,
        mode="ip_lla",
        key=jax.random.PRNGKey(123),
        # plot_Z=True,
        # cbar=False
    )

    axs[1].set_xlabel(r"$x_1$")
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#333')
            spine.set_linewidth(1.0)
        # ax.set_ylabel(None)

    # axs[0].set_title(f'{r"$M="}{M}{r"$"}')
    axs[0].set_title(None)
    axs[1].set_title(None)

    # plot_binary_classification_data(Xtrain, ytrain.squeeze(), axs[0])
    # plot_binary_classification_data(Xtrain, ytrain.squeeze(), axs[1])
    
    scatterp(*Z.T, color="yellow", s=200, zorder=8, marker="X", ax=axs[0])
    scatterp(*Z.T, color="yellow", s=200, zorder=8, marker="X", ax=axs[1])
    # fig.subplots_adjust(top=0.85)
    # fig.suptitle("Predictive mean", fontsize=16)

    return fig, axs
    
    
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