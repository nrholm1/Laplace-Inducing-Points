import os
from typing import Callable
import numpy as np
import optax
import pytest
import jax
import jax.numpy as jnp
from flax import struct
from flax.training import train_state

from src.toymodels import SimpleClassifier, SimpleRegressor
from src.utils import load_yaml, load_checkpoint, load_array_checkpoint
from src.toydata import JAXDataset, get_dataloaders

# jax.config.update("jax_enable_x64", True)

@pytest.fixture
def regression_1d_data():
    """
    Returns a small 1D regression dataset (X, y).
    For instance: y ~ 2x + noise
    """
    X = jnp.array([[-1.0], [0.0], [1.1], [3.5]])
    y = 2.0 * X + 0.1 * jax.random.normal(jax.random.PRNGKey(42), shape=X.shape)
    return X, y


@pytest.fixture
def small_model_state(regression_1d_data):
    """
    Create a very small 'model' state for a 1-layer linear model.

    Only parameters "W" and "b" are learnable.
    The apply_fn returns a fixed logvar of 0 (when requested) so that
    the likelihood is equivalent to a mean squared error loss.
    
    With fixed variance, the Gauss-Newton approximation should match
    the full Hessian (computed via jax.hessian) for this linear model.
    """
    def apply_fn(params, x, return_logvar=True):
        W = params['params']["W"]  # scalar
        b = params['params']["b"]  # scalar
        mu = W * x + b
        if return_logvar:
            return mu, params['logvar']['logvar']
        else:
            return mu

    key = jax.random.PRNGKey(0)
    key_w, key_b, key_logvar = jax.random.split(key, 3)
    W_init = jax.random.normal(key_w, ()) * 0.1
    b_init = jax.random.normal(key_b, ()) * 0.1
    logvar_init = jax.random.uniform(key_logvar, ()) * 0.1

    params = {
        'params': {
        "W": W_init,
        "b": b_init,
    },
        'logvar': {
        "logvar": logvar_init,
    }}

    @struct.dataclass
    class TrainState:
        params: any
        apply_fn: Callable = struct.field(pytree_node=False)
    
    return TrainState(params, apply_fn)


@pytest.fixture
def toyregressor_state():
    model_cfg = load_yaml("config/toyregressor_sine.yml")
    model_type = "regressor"
    num_h = model_cfg["num_h"]
    num_l = model_cfg["num_l"]
    
    
    model = SimpleRegressor(numh=num_h, numl=num_l)
    rng_model = jax.random.PRNGKey(model_cfg["rng_seed"])
    dummy_inp = jax.random.normal(rng_model, shape=(num_h, 1))
    variables = model.init(rng_model, dummy_inp)

    optimizer_map = optax.adam(1e-3)
    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables,
        tx=optimizer_map
    )
    map_model_state = load_checkpoint(
            ckpt_dir="checkpoint/map/",
            prefix="map_sine",
            target=model_state
        )
    return map_model_state


@pytest.fixture
def sine_data():
    # Load data
    datafile = f"data/sine.npz"
    if not os.path.exists(datafile):
        raise FileNotFoundError(f"Data file not found: {datafile}")
    data_npz = np.load(datafile)
    x = jax.device_put(data_npz["x"])
    y = jax.device_put(data_npz["y"])
    n_samples = x.shape[0]
    print(f"[INFO] Loaded dataset from {datafile} with {n_samples} samples.")
    
    trainsplit = int(0.9 * n_samples)
    xtrain, ytrain = x[:trainsplit], y[:trainsplit]
    xtest,  ytest  = x[trainsplit:], y[trainsplit:]
    train_dataset = JAXDataset(xtrain, ytrain)
    test_dataset  = JAXDataset(xtest,  ytest)
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size=min(16, len(test_dataset)))
    
    return train_loader, test_loader


@pytest.fixture
def classification_2d_data():
    """
    Generates a simple 2D classification dataset with two classes.
    
    Returns:
        X: A jnp.array of shape (200, 2) containing the features.
        y: A jnp.array of shape (200,) containing the class labels (0 or 1).
    """
    key = jax.random.PRNGKey(0)
    n_points = 100  # Number of points per class

    # Class 0: centered at (-1, 0) with some noise.
    key, subkey = jax.random.split(key)
    class0 = jax.random.normal(subkey, (n_points, 2)) * 0.5 + jnp.array([-1.0, 0.0])

    # Class 1: centered at (1, 0) with some noise.
    key, subkey = jax.random.split(key)
    class1 = jax.random.normal(subkey, (n_points, 2)) * 0.5 + jnp.array([1.0, 0.0])

    # Concatenate the classes into one dataset.
    X = jnp.concatenate([class0, class1], axis=0)
    # Create labels: 0 for class0, 1 for class1.
    y = jnp.concatenate([jnp.zeros(n_points, dtype=jnp.int32), jnp.ones(n_points, dtype=jnp.int32)], axis=0)
    
    return X, y


@pytest.fixture
def classifier_state():
    # Load the YAML configuration file.
    config_path = "config/toyclassifier_xor.yml"
    config = load_yaml(config_path)
    
    # Extract configuration parameters.
    model_type = config.get("model_type", "classifier")  # defaults to classifier
    num_h = config["num_h"]
    num_l = config["num_l"]
    num_c = config.get("num_c", 2)
    model_seed = config["rng_seed"]

    # Instantiate the classifier.
    model = SimpleClassifier(numh=num_h, numl=num_l, numc=num_c)

    # Initialize the model parameters.
    # dummy_inp = jax.random.normal(rng_inp, shape=(num_h, num_c))
    dummy_inp = jnp.ones((1, 2))
    variables = model.init(jax.random.PRNGKey(model_seed), dummy_inp)

    optimizer_map = optax.adam(1e-3)
    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables,
        tx=optimizer_map
    )
    model_state = load_checkpoint(
            ckpt_dir="checkpoint/map/",
            prefix="map_xor",
            target=model_state
        )

    # class State:
    #     pass

    # state = State()
    # state.params = params
    # state.apply_fn = model.apply
    return model_state


@pytest.fixture
def matrix_test_suite():
    """
    Create PSD matrices with varying spectrum magnitude for verifying numerical stability.
    PSD makes it less random since there otherwise might be some sign issues.
    Also, PSD will always be the case for GGN, by design.
    """
    # trace = 6
    M1 = jnp.diag(jnp.array([1.,2.,3.]))
    
    # trace = 10
    M2 = jnp.array([ 
        [  1., 4,  50],
        [-30,  4., 16],
        [ 12,  6,  5.],
    ])
    M2 = M2@M2.T
    
    M3 = jax.random.normal(key=jax.random.PRNGKey(seed=45895), shape=(3000,3000))
    M3 = M3@M3.T
    
    return M1,M2,M3