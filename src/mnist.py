import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from seaborn import set_style

import jax
import jax.numpy as jnp 
import flax.linen as nn
from flax.training import train_state
import optax
from tqdm import tqdm

from src.toydata import JAXDataset, jax_collate_fn, get_dataloaders
from src.utils import save_checkpoint

# Set random seeds for reproducibility.
torch.manual_seed(0)
np.random.seed(0)

# Set plotting style.
set_style('darkgrid')


def load_mnist_numpy(train=True):
    """
    Loads MNIST using torchvision, converts images to numpy arrays,
    and reshapes each image to (28,28,1).
    (MNIST from torchvision is already normalized to [0,1].)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # returns a tensor of shape (1, 28, 28)
    ])
    
    dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    images, labels = [], []
    
    for img, label in dataset:
        # Reshape from (1, 28, 28) to (28, 28, 1)
        images.append(img.reshape(28,28,1))
        labels.append(label)
        
    images = np.stack(images, axis=0)  # Shape: (N, 28, 28, 1)
    labels = np.array(labels)          # Shape: (N,)
    return images, labels


class CNN(nn.Module):
    """A simple CNN model using linen."""
    @nn.compact
    def __call__(self, x):
        # Input shape: (batch, 28, 28, 1)
        # Use 'SAME' padding so that the output spatial dimensions remain.
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten; should result in (batch, 3136)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


# todo make this MAP training instead!

def loss_fn(params, batch, state):
    """Compute loss and logits for a batch."""
    X, y = batch
    logits = state.apply_fn(params, X)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=y.squeeze().astype(jnp.int32)
    ).mean()
    return loss, logits


@jax.jit
def train_step(state, batch):
    """Perform a single training step."""
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params, batch, state)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, logits


@jax.jit
def eval_step(state, batch):
    """Evaluate the model on a batch."""
    X, y = batch
    logits = state.apply_fn(state.params, X)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=y.squeeze().astype(jnp.int32)
    ).mean()
    return state, loss, logits


def create_train_state(rng, model, learning_rate, momentum):
    """Initialize TrainState with model parameters and an optimizer."""
    dummy_inp = jnp.ones((1, 28, 28, 1), jnp.float32)
    params = model.init(rng, dummy_inp)
    tx = optax.adamw(learning_rate=learning_rate, b1=momentum)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


if __name__ == '__main__':
    train_steps = 1200
    eval_every = 200
    batch_size = 32

    # Load MNIST.
    train_images, train_labels = load_mnist_numpy(train=True)
    test_images, test_labels = load_mnist_numpy(train=False)
    train_dataset = JAXDataset(train_images, train_labels)
    test_dataset = JAXDataset(test_images, test_labels)
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size, collate_fn=jax_collate_fn)
    
    # Instantiate the model.
    model = CNN()
    # Create the TrainState.
    state = create_train_state(jax.random.PRNGKey(0), model, learning_rate=0.005, momentum=0.9)

    test_acc = 0.
    pbar = tqdm(enumerate(train_loader), total=train_steps)
    for step, batch in pbar:
        state, loss, logits = train_step(state, batch)
        if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
            # Evaluate on the test set.
            for test_batch in test_loader:
                state, test_loss, test_logits = eval_step(state, test_batch)
                test_acc += (test_logits.argmax(axis=1) == test_batch[1].squeeze()).mean()
            test_acc /= len(test_loader)
            pbar.set_description(f"Step {step}, Loss: {loss:.4f}, Test acc.: {test_acc:.4f}")
    
    # Save a checkpoint.
    save_checkpoint(state, ckpt_dir="./checkpoint/map", prefix="mnist", step=step)
    # pdb.set_trace()