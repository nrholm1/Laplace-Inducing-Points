import jax
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import jax.numpy as jnp
import numpy as np

from src.data import JAXDataset, NumpyDataset, get_dataloaders as _get_dataloaders, jax_collate_fn, numpy_collate_fn

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
        images.append(img.reshape(28, 28, 1))
        labels.append(label)
        
    images = np.stack(images, axis=0)  # Shape: (N, 28, 28, 1)
    labels = np.array(labels)          # Shape: (N,)
    return images, labels


def load_fmnist_numpy(train=True):
    """
    Loads Fashion-MNIST using torchvision, converts images to numpy arrays,
    and reshapes each image to (28,28,1).
    (FashionMNIST from torchvision is already normalized to [0,1].)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # returns a tensor of shape (1, 28, 28)
    ])
    
    dataset = datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)
    images, labels = [], []
    
    for img, label in dataset:
        # Reshape from (1, 28, 28) to (28, 28, 1)
        images.append(img.reshape(28, 28, 1))
        labels.append(label)
        
    images = np.stack(images, axis=0)  # Shape: (N, 28, 28, 1)
    labels = np.array(labels)          # Shape: (N,)
    return images, labels


def get_dataloaders(name, batch_size, num_workers=0):
    if name == 'mnist':
        xtrain, ytrain = load_mnist_numpy(train=True)
        xtest, ytest   = load_mnist_numpy(train=False)

    elif name == 'fmnist':
        xtrain, ytrain = load_fmnist_numpy(train=True)
        xtest, ytest   = load_fmnist_numpy(train=False)

    else:
        raise ValueError(f"Unknown dataset name '{name}'")

    # Use NumpyDataset + numpy_collate_fn by default
    train_dataset = NumpyDataset(xtrain, ytrain)
    test_dataset  = NumpyDataset(xtest, ytest)
    train_loader, test_loader = _get_dataloaders(
        train_dataset, test_dataset,
        batch_size,
        collate_fn=numpy_collate_fn
    )
    
    print(f"[INFO] Loaded dataset '{name}'.")
    return train_loader, test_loader
