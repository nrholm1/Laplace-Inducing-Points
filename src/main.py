import argparse
import os
import pickle

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
jax.config.update("jax_enable_x64", True)
import tensorflow_probability.substrates.jax as tfp

from tqdm import tqdm
import matplotlib.pyplot as plt
from seaborn import set_style
set_style('darkgrid')

from toymodels import SimpleRegressor, SimpleClassifier, print_summary
from toydata import JAXDataset, get_dataloaders, sine_wave_dataset, sine_wave_fun, data_ex5, xor_dataset
from nplot import Colors, linep, scatterp, plot_cinterval, plot_inducing_points_1D


#%%
def nll_fun(state, params, batch):
    inp,res = batch
    pred = state.apply_fn(params, inp)
    mse_loss = optax.squared_error(pred,res).mean()
    return mse_loss


def log_prior_fun(params, stdev):
    """Simple L2 prior. Note: returns negative log prior!"""
    def sum_of_squares(t): return jnp.sum(t**2)
    param_leaves = jax.tree_util.tree_leaves(params)
    l2 = jnp.sum(jnp.array([sum_of_squares(p) for p in param_leaves]))
    return .5 / stdev * l2


def log_posterior_fun(state, params, batch, prior_std=1.0):
    nll = nll_fun(state, params, batch) # Gaussian likelihood => NLL ~ MSE
    log_prior = log_prior_fun(state.params, prior_std)
    return nll + log_prior


@jax.jit
def train_map_step(state, batch, prior_std=1.0):
    grad_fn = jax.value_and_grad(log_posterior_fun, argnums=1, has_aux=False)
    loss,grads = grad_fn(state, state.params, batch, prior_std)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, batch):
    loss = nll_fun(state, state.params, batch)
    return loss


def train_model(state, trainloader, testloader, train_step, *args, num_epochs=100, **kwargs):
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        for batch in trainloader:
            state, loss = train_step(state, batch, *args, **kwargs)
        if (epoch) % 25 == 0:
            eloss = 0
            for batch in testloader:
                eloss += eval_step(state, batch) # TODO make eval MAP step?
            pbar.set_description(f"Avg. eval MSE: {eloss/len(testloader):.3f}")
    return state



def compute_ggn(state, x, prior_std, full_set_size=None):
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(state.params)
    
    def per_datum_jacobian(xi):
        def scalar_output(flat_p):
            p = unravel_fn(flat_p)
            return state.apply_fn(p, xi[None]).squeeze()
        return jax.jacobian(scalar_output)(flat_params)

    Js = jax.vmap(per_datum_jacobian)(x)
    JtJ = jnp.einsum('ni,nj->ij', Js, Js)
    prior_precision = 1.0 / (prior_std**2)
    
    # rescaling weight if subsampled/inducing points
    N = x.shape[0]
    M = full_set_size or N
    
    GGN = N/M * JtJ + prior_precision*jnp.eye(JtJ.shape[0])
    
    return GGN, flat_params, unravel_fn
    

def ensure_symmetry(X, jitter=1e-8):
    return 0.5 * (X + X.T) + jitter * jnp.eye(X.shape[0]) # ! ensure symmetry of GGN (numerical stability)


def posterior_lla(map_state, x, prior_std, full_set_size=None, return_unravel_fn=False):
    """
    Posterior parameter distribution.
    """
    GGN, flat_params_map, unravel_fn = compute_ggn(map_state, x, prior_std, full_set_size=full_set_size) 
    GGN = ensure_symmetry(GGN)
    H_approx = jnp.linalg.inv(GGN)
    posterior_dist = tfp.distributions.MultivariateNormalFullCovariance(
            loc=flat_params_map.astype(jnp.float64), # todo: cast to f64 or other to f32?
            covariance_matrix=H_approx
        )
    if return_unravel_fn:
        return posterior_dist, unravel_fn
    return posterior_dist


def predict_lla(map_state, xnew, xtrain, prior_std, full_set_size=None):
    """
    Posterior predictive distribution.
    """
    GGN, flat_params_map, unravel_fn = compute_ggn(map_state, xtrain, prior_std, full_set_size=full_set_size) 
    GGN = ensure_symmetry(GGN)
    H_approx = jnp.linalg.inv(GGN)
    
    @jax.jit
    def flat_apply_fn(flat_p, inputs):
        p = unravel_fn(flat_p)
        return map_state.apply_fn(p, inputs)
    
    @jax.jit
    def per_datum_jacobian(xi):
        return jax.jacobian(lambda fp: flat_apply_fn(fp, xi[None]))(flat_params_map)
    Jnew = jax.vmap(per_datum_jacobian)(xnew)
    
    f_mean = flat_apply_fn(flat_params_map, xnew)
    
    @jax.jit
    def per_datum_cov(Ji):
        return Ji @ H_approx @ Ji.T
    f_cov = jax.vmap(per_datum_cov)(Jnew.squeeze(axis=1))
    
    assert jnp.all(jnp.linalg.eigvals(f_cov) > 0), "Covariance matrix not PD!"
    
    return tfp.distributions.MultivariateNormalDiag(loc=f_mean.squeeze(), 
                                                    scale_diag=jnp.sqrt(f_cov).squeeze())
    

def sample_params(mu, cov, rng):
    eps = jax.random.normal(rng, shape=mu.shape)
    # Cholesky to map eps ~ N(0,I) -> theta ~ N(mu, cov)
    L = jnp.linalg.cholesky(cov)
    return mu + L @ eps


def loglik_dataset(params, apply_fn, xdata, ydata):
    # ! for regression! closed form:
    ypred = apply_fn(params, xdata)
    twovar = 2/alpha
    N = xdata.shape[0]
    return -N/2 *jnp.log(twovar*jnp.pi) - 1/twovar * ((ydata-ypred)**2).sum()


def var_loglik_fun(q, dataset, apply_fn, unravel_fn, rng, num_samples):
    """
    # ! using MC sample(s) of parameters
    """
    x,y = dataset
    mu,cov = q.mean(), q.covariance()
    log_sum = 0.0
    for i in range(num_samples):
        rng_i = jax.random.fold_in(rng, i)  # make a fresh key
        theta_sample = sample_params(mu, cov, rng_i)
        theta_sample = unravel_fn(theta_sample)
        log_p_data = loglik_dataset(theta_sample, apply_fn, x, y) # todo don't use *all* data
        log_sum += log_p_data
    
    return log_sum / num_samples
    
    
def var_kl_fun(q, alpha):
    mu,cov = q.mean(), q.covariance()
    D = cov.shape[0]
    tr_term = alpha*jnp.linalg.trace(cov)
    norm_term =alpha*jnp.linalg.norm(mu)**2
    logdetp_term = jnp.log(D/alpha) # log(det( I * alpha^(-1) ))
    logdetq_term = jnp.log(jnp.linalg.det(cov))
    
    kl_term = 0.5 * (tr_term - D + norm_term + logdetp_term - logdetq_term)
    return kl_term


def naive_objective(xind, dataset, state, alpha, rng, num_mc_samples, full_set_size=None, reg_coeff=0):
    q,unravel_fn = posterior_lla(state, xind, 
                      prior_std=alpha, full_set_size=full_set_size,
                      return_unravel_fn=True)

    loglik_term = var_loglik_fun(q, dataset, state.apply_fn, unravel_fn, rng, num_samples=num_mc_samples)
    kl_term = var_kl_fun(q, alpha)
    reg_term = reg_coeff * jnp.sum(jnp.square(xind))
    return - (loglik_term-kl_term) + reg_term

variational_grad = jax.value_and_grad(naive_objective)

def optimize_step(x, dataset, map_model_state, alpha, opt_state, rng, num_mc_samples, full_set_size=None):
    loss, grads = variational_grad(x, dataset, map_model_state, alpha, rng, num_mc_samples=num_mc_samples, full_set_size=full_set_size)
    updates, new_opt_state = xoptimizer.update(grads, opt_state)
    new_x = optax.apply_updates(x, updates)
    return new_x, new_opt_state, loss
# JIT optimize_step here, since the static_argnames is problematic in the decorator?
optimize_step = jax.jit(optimize_step, static_argnames=['num_mc_samples', 'full_set_size'])


def train_inducing_points(map_model_state, xinit, full_dataset, rng, num_mc_samples=10, alpha=1.0, num_steps=100):
    opt_state = xoptimizer.init(xinit)
    x = xinit
    N = full_dataset[0].shape[0]
    
    pbar = tqdm(range(num_steps))
    for step in pbar:
        rng, rng_step = jax.random.split(rng)
        dataset_sample = full_dataset # todo: mini batch here?
        x,opt_state,loss = optimize_step(x, dataset_sample, map_model_state, alpha, opt_state, rng_step, num_mc_samples=num_mc_samples, full_set_size=N)
        x = jnp.clip(x, xtrain.min(), xtrain.max()) # ! hard constraint enforcement
        
        if step == 0:
            print(f"Initial loss: {loss:.3f}")
        if step % 1 == 0:
            pbar.set_description_str(f"Loss: {loss:.3f}", refresh=True)
    return x


def create_dataset(dataset_name, n, key, noise):
    """
    Creates a dataset of size n according to dataset_name, 
    optionally with added noise. 
    Returns (x, y).
    """
    if dataset_name == 'xor':
        x, y = xor_dataset(n, key, noise)
    elif dataset_name == 'sine':
        x, y = sine_wave_dataset(n, key, noise, split_in_middle=True)
    else:
        raise ValueError(f"Unknown dataset_name = {dataset_name}")
    return x, y

def save_checkpoint(obj, filename):
    """
    Save an object (e.g. model params, array, etc.) to disk.
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"[checkpoint] Saved to {filename}")

def load_checkpoint(filename):
    """
    Load a pickled object from disk.
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(f"[checkpoint] Loaded from {filename}")
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="full_pipeline",
                        choices=["train_map", "train_inducing", "full_pipeline"],
                        help="Which phase of the pipeline to run: train_map, train_inducing, etc.")
    parser.add_argument("--dataset", type=str, default="sine",
                        help="Which dataset to use: 'xor' or 'sine'")
    parser.add_argument("--nsamples", metavar='N', type=int, default=128,
                        help="Number of data samples to generate.")
    parser.add_argument("--noise", type=float, default=0.5,
                        help="Noise level for synthetic data.")
    parser.add_argument("--numh", type=int, default=32,
                        help="Number of hidden units.")
    parser.add_argument("--numl", type=int, default=2,
                        help="Number of hidden layers.")
    parser.add_argument("--numc", type=int, default=2,
                        help="Number of output classes (for classification).")
    parser.add_argument("--seed_data", type=int, default=42,
                        help="Random seed for data generation.")
    parser.add_argument("--seed_model", type=int, default=1337,
                        help="Random seed for model init.")
    parser.add_argument("--seed_induc", type=int, default=314159265,
                        help="Random seed for inducing points.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Prior precision or noise precision, depends on usage.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for MAP training.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--epochs_map", type=int, default=2000,
                        help="Number of epochs for MAP training.")
    parser.add_argument("--m_induc", metavar='m', type=int, default=16,
                        help="Number of inducing points.")
    parser.add_argument("--num_steps_induc", type=int, default=500,
                        help="Number of steps for inducing point optimization.")
    parser.add_argument("--ckpt_map", type=str, default="map/map_checkpoint.pkl",
                        help="Filename for loading/saving the MAP model checkpoint.")
    parser.add_argument("--ckpt_induc", type=str, default="ind/induc_checkpoint.pkl",
                        help="Filename for loading/saving the inducing points checkpoint.")
    args = parser.parse_args()
    
    # print selected options
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)
        
    # Step A: Create or load data
    rng_data = jax.random.PRNGKey(args.seed_data)
    x, y = create_dataset(args.dataset, args.n_samples, rng_data, args.noise)
    
    # 90/10 train/test split
    trainsplit = int(0.9 * args.n_samples)
    xtrain, ytrain = x[:trainsplit], y[:trainsplit]
    xtest,  ytest  = x[trainsplit:], y[trainsplit:]

    # todo: save the dataset?
    
    # Step B: Visualize data and Build model according to dataset
    if args.dataset == 'sine':
        # PLOT
        scatterp(xtrain,ytrain,color=Colors.paleblue, label='Train data')
        scatterp(xtest,ytest,color=Colors.yellow, zorder=2, label='Test data')
        t = jnp.linspace(-4,3,250)
        f = sine_wave_fun
        plot_latent_fun = lambda t: linep(t,f(t),zorder=3,color=Colors.darkorange,label='$f$ (latent fun)') \
            if not jnp.any(jnp.isnan(f(t[0]))) else ...
        plot_latent_fun(t)
        plt.legend()
        plt.show()
        
        model = SimpleRegressor(numh=args.num_h, numl=args.num_l)
    else:
        model = SimpleClassifier(numh=args.num_h, numl=args.num_l, numc=args.num_c)

    # Initialize model with a random input
    rng_model = jax.random.PRNGKey(args.seed_model)
    rng_model, inp_rng, init_rng = jax.random.split(rng_model, 3)
    dummy_inp = jax.random.normal(inp_rng, shape=(args.num_h, 1))  # or shape=(1, input_dim)
    params = model.init(init_rng, dummy_inp)

    print("== Model Summary ==")
    print_summary(params)
    
    
    optimizer_map = optax.adam(learning_rate=args.lr)
    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer_map
    )

    # Step C: (Optionally) Train the model to get MAP solution
    if args.mode in ["train_map", "full_pipeline"]:
        train_dataset = JAXDataset(xtrain, ytrain)
        test_dataset  = JAXDataset(xtest,  ytest)
        train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, args.batch_size)
        
        map_model_state = train_model(
            model_state,
            train_loader,
            test_loader,
            train_step=train_map_step,
            prior_std=args.alpha**(-0.5),
            num_epochs=args.epochs_map
        )
        # Save the MAP checkpoint
        save_checkpoint(map_model_state, args.checkpoint_map)
    
    else:
        # If not training, we assume we load from checkpoint
        map_model_state = load_checkpoint(args.checkpoint_map)


    # Step D: (Optionally) Inducing Points
    if args.mode in ["train_inducing", "full_pipeline"]:
        # todo Possibly load map_model_state:
        # map_model_state = load_checkpoint(args.checkpoint_map)  # if we haven't already

        # Initial inducing points
        rng_inducing = jax.random.PRNGKey(args.seed_inducing)
        xinit = jax.random.uniform(rng_inducing, shape=(args.m_inducing,)) * 7.0 - 4.0

        # Optimize inducing points
        full_dataset = (xtrain, ytrain)
        xinduc = train_inducing_points(
            map_model_state, xinit,
            full_dataset=full_dataset,
            rng=rng_inducing,
            num_mc_samples=10,
            alpha=args.alpha,
            num_steps=args.num_steps_induc
        )

        # Save the inducing points
        save_checkpoint(xinduc, args.checkpoint_induc)
    
    else:
        # If not training them, we might load from a checkpoint:
        xinduc = load_checkpoint(args.checkpoint_induc)

    # Step E: (Optional) Evaluate / Plot results
    # -----------------------------------------------------------------------
    # TODO Here we could
    #  1) Compare the predictive distribution with full, subsampled, and optimized
    #  2) Plot the data, the model predictions, etc.
    #  3) Save any figures or metrics.
    
    xlin = jnp.linspace(-4, 3, 100)[:,None]
    postpreddist_full = predict_lla(map_model_state, xlin, 
                                         xtrain,
                                         prior_std=1.0)
    postpreddist_rand = predict_lla(map_model_state, xlin, 
                                         xtrain_subs,
                                         prior_std=1.0, full_set_size=xtrain.shape[0])
    postpreddist_optimized = predict_lla(map_model_state, xlin, 
                                         xinduc,
                                         prior_std=1.0, full_set_size=xtrain.shape[0])
    
    # ! Plot of optimized inducing points vs others
    fig, ax = plt.subplots(figsize=(8,5))
    plot_cinterval(xlin.squeeze(), postpreddist_full.mean(), postpreddist_full.stddev(), 
                text="full", color='orange',zorder=5)
    plot_cinterval(xlin.squeeze(), postpreddist_rand.mean(), postpreddist_rand.stddev(), 
                text="ind. init", color='red', zorder=3)
    plot_cinterval(xlin.squeeze(), postpreddist_optimized.mean(), postpreddist_optimized.stddev(), 
                text="ind. optimized", color='green',zorder=4)
    scatterp(xtest,ytest,color=Colors.yellow, zorder=2, label='Test data')
    plot_inducing_points_1D(ax, xinduc, color='green', offsetp=0.00, zorder=3, label=None)
    plot_inducing_points_1D(ax, xinit, offsetp=0.00, zorder=3, label=None)
    plt.legend(loc='lower right')
    plt.savefig("fig/pdist-full_vs_init_vs_optimized.pdf")
    
    print("Done.")



if __name__ == "__main__":
    main()




# #%%
# """MODEL INITIALIZATION"""
# numh = 32
# numl = 2
# numc = 2
# model = SimpleRegressor(numh=numh, numl=numl)
# # model = SimpleClassifier(numh=numh, numl=numl, numc=numc)
# rng = jax.random.key(seed=1337)
# rng, inp_rng, init_rng = jax.random.split(rng, 3)

# inp = jax.random.normal(inp_rng, shape=(numh,1))
# params = model.init(init_rng, inp)
# print_summary(params)


# """SYNTHETIC DATA GENERATION"""
# n = 128
# key = jax.random.key(seed=42)
# noise = 0.5

# # todo options for which dataset
# x,y = xor_dataset(n, key, noise) # classification

# x,y = sine_wave_dataset(n, key, noise, split_in_middle=True) # regression
# f = sine_wave_fun # latent function for regression

# trainsplit = int(0.9*n)
# xtrain, ytrain = x[:trainsplit], y[:trainsplit]
# xtest, ytest = x[trainsplit:], y[trainsplit:]
# train_dataset = JAXDataset(xtrain, ytrain)
# test_dataset = JAXDataset(xtest, ytest)

# # todo saving of dataset here maybe? 

# # dataset visualization
# def plot_data():
#     scatterp(xtrain,ytrain,color=Colors.paleblue, label='Train data')
#     scatterp(xtest,ytest,color=Colors.yellow, zorder=2, label='Test data')
# plot_data()
# t = jnp.linspace(-4,3,250)
# plot_latent_fun = lambda t: linep(t,f(t),zorder=3,color=Colors.darkorange,label='$f$ (latent fun)') \
#     if not jnp.any(jnp.isnan(f(t[0]))) else ...
# plot_latent_fun(t)
# plt.legend()
# plt.show()



# """
# MAP TRAINING
# """
# batch_size = 32
# train_loader,test_loader = get_dataloaders(train_dataset, test_dataset, batch_size)
# alpha = 1.0
# lr = 1e-3
# optimizer = optax.adam(learning_rate=lr)
# model_state = train_state.TrainState.create(apply_fn=model.apply, 
#                                             params=params, 
#                                             tx=optimizer)
# map_model_state = train_model(model_state, 
#                                   train_loader, 
#                                   test_loader,
#                                   train_step=train_map_step,
#                                   prior_std=alpha**(-1/2),
#                                   num_epochs=2_000)
# # todo: save MAP checkpoint here

# """
# INDUCING POINTS
# Take a subset of m training data points instead of the full set.
# """

# # todo: load MAP checkpoint here

# m = 16
# subsample_key = jax.random.PRNGKey(seed=314159265)
# #%%
# xlin = jnp.linspace(-4.5,3.5,100)[:,None]
# ylin_map = model.apply(map_model_state.params, xlin)

# fig, ax = plt.subplots(figsize=(8,5))

# # xinit = jnp.linspace(xtest.min(), xtest.max(), num=m)
# # xinit = jax.random.normal(jax.random.PRNGKey(4), shape=(m,))* 2
# xinit = jax.random.uniform(jax.random.PRNGKey(4), shape=(m,))*7 - 4
# perm = jax.random.permutation(subsample_key, xtrain.shape[0])
# subsample = perm[:m]
# xtrain_subs = xinit
# predictive_dist_subsampled = predict_lla(map_model_state, xlin, 
#                                          xtrain_subs,
#                                          prior_std=1.0, full_set_size=xtrain.shape[0])
# predictive_dist =            predict_lla(map_model_state, xlin, 
#                                          xtrain,
#                                          prior_std=1.0)
# plot_cinterval(xlin.squeeze(), predictive_dist.mean(), predictive_dist.stddev(), 
#                text="full", color='orange',zorder=4)
# plot_cinterval(xlin.squeeze(), predictive_dist_subsampled.mean(), predictive_dist_subsampled.stddev(), 
#                text="subsampled", color='red', zorder=3)
# scatterp(xtest,ytest,color=Colors.yellow, zorder=2, label='Test data')
# # scatterp(xtrain_subs, ytrain_subs, color='red', zorder=4, label='Inducing data points')
# plot_inducing_points_1D(ax, xtrain_subs, offsetp=-0.05, zorder=3, label=None) # ! plot last to ensure correct ylim!
# plt.legend(loc='lower right')
# plt.show()

# #%%
# """
# OPTIMIZE INDUCING POINTS
# """
# # TODO optional loading of inducing points checkpoint here

# key, train_rng = jax.random.split(key, 2)
# stepsize = 1e-2
# xoptimizer = optax.adam(stepsize)
# full_dataset = (xtrain,ytrain)
# xinduc = train_inducing_points(map_model_state, xinit, full_dataset=full_dataset, rng=train_rng, num_mc_samples=10, alpha=alpha, num_steps=500)

# # TODO optional saving of inducing points checkpoint here

# #%%

# predictive_dist_optimized = predict_lla(map_model_state, xlin, 
#                                          xinduc,
#                                          prior_std=1.0, full_set_size=xtrain.shape[0])

# # Plot of optimized inducing points vs others
# fig, ax = plt.subplots(figsize=(8,5))
# plot_cinterval(xlin.squeeze(), predictive_dist.mean(), predictive_dist.stddev(), 
#                text="full", color='orange',zorder=5)
# plot_cinterval(xlin.squeeze(), predictive_dist_subsampled.mean(), predictive_dist_subsampled.stddev(), 
#                text="ind. init", color='red', zorder=3)
# plot_cinterval(xlin.squeeze(), predictive_dist_optimized.mean(), predictive_dist_optimized.stddev(), 
#                text="ind. optimized", color='green',zorder=4)
# scatterp(xtest,ytest,color=Colors.yellow, zorder=2, label='Test data')
# plot_inducing_points_1D(ax, xinduc, color='green', offsetp=0.00, zorder=3, label=None)
# plot_inducing_points_1D(ax, xinit, offsetp=0.00, zorder=3, label=None)
# plt.legend(loc='lower right')
# plt.show()