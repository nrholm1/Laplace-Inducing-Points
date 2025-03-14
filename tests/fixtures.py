from typing import Callable
import pytest
import jax
import jax.numpy as jnp
from flax import struct

from src.toymodels import SimpleClassifier
from src.utils import load_yaml


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
            return mu, params['params']['logvar']
        else:
            return mu

    key = jax.random.PRNGKey(0)
    key_w, key_b, key_logvar = jax.random.split(key, 3)
    W_init = jax.random.normal(key_w, ()) * 0.1
    b_init = jax.random.normal(key_b, ()) * 0.1
    logvar_init = jax.random.uniform(key_logvar, ()) * 0.1

    params = {'params': {
        "W": W_init,
        "b": b_init,
        "logvar": logvar_init,
    }}

    @struct.dataclass
    class TrainState:
        params: any
        apply_fn: Callable = struct.field(pytree_node=False)
    
    return TrainState(params, apply_fn)


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
    config_path = "config/tests/toyclassifier.yml"
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
    rng_model = jax.random.PRNGKey(model_seed)
    dummy_inp = jax.random.normal(rng_model, shape=(num_h, num_c))
    params = model.init(rng_model, dummy_inp)

    # Create a state object with the parameters and the model's apply function.
    # You can use a simple object or a dataclass; here we use a simple object.
    class State:
        pass

    state = State()
    state.params = params
    state.apply_fn = model.apply
    return state