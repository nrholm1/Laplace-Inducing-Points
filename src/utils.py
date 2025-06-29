import os
import yaml
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import jax.flatten_util

is_pd = lambda M: jnp.all(jnp.linalg.eigvals(M) >= 1e-9)


def flatten_nn_params(params):
    # ! maybe inefficient?
    nn_params = {
        k: v for k, v in params.items() if k not in ['logvar', 'batch_stats']
    }
    return jax.flatten_util.ravel_pytree(nn_params)
    

def save_array_checkpoint(array, ckpt_dir, name, step):
    """
    Save a JAX array (e.g. for inducing points) as a .npy file:
      ckpt_dir/name_step.npy
    """
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    filename = os.path.join(ckpt_dir, f"{name}_{step}.npy")
    np.save(filename, np.array(array))  # convert to NumPy
    print(f"[checkpoint] Saved array checkpoint at step {step} in {filename}")


def load_array_checkpoint(ckpt_dir, name, step):
    """
    Load a .npy file and return as a JAX array (device_put).
      ckpt_dir/name_step.npy
    """
    ckpt_dir = os.path.abspath(ckpt_dir)
    filename = os.path.join(ckpt_dir, f"{name}_{step}.npy")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} not found")
    array = np.load(filename)
    print(f"[checkpoint] Loaded array checkpoint from {filename}")
    return jax.device_put(array)


def save_checkpoint(train_state, ckpt_dir, prefix, step):
    """
    Save a Flax TrainState to ckpt_dir with given prefix, e.g.:
      ckpt_dir/prefix_<step>
    """
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir,
        target=train_state,
        step=step,
        prefix=prefix + "_",
        overwrite=True
    )
    print(f"[checkpoint] Saved model checkpoint at step {step} in {ckpt_dir} (prefix={prefix})")


def load_checkpoint(ckpt_dir, prefix, target=None):
    """
    Load a Flax TrainState from ckpt_dir with the given prefix.
    If multiple steps are present, loads the latest by default.
    """
    ckpt_dir = os.path.abspath(ckpt_dir)
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=target,
        prefix=prefix + "_",
    )
    print(f"[checkpoint] Loaded model checkpoint from {ckpt_dir} (prefix={prefix})")
    return restored_state


def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


count_model_params = lambda params: sum(x.size for x in jax.tree_util.tree_leaves(params))

def print_summary(params):
    num_model_params = count_model_params(params)    
    print(f"Param count     (D) : {num_model_params}")
    print(f"Cov. mat. size (D^2): {num_model_params**2:.3e}")
    
    
def print_options(args):
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)