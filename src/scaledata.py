import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from src.data import JAXDataset, get_dataloaders as _get_dataloaders, jax_collate_fn


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



def get_dataloaders(name, batch_size):
    if name == 'mnist':
        xtrain, ytrain = load_mnist_numpy(train=True)
        xtest, ytest = load_mnist_numpy(train=False)
        train_dataset = JAXDataset(xtrain, ytrain)
        test_dataset = JAXDataset(xtest, ytest)
        train_loader, test_loader = _get_dataloaders(train_dataset, test_dataset, batch_size, collate_fn=jax_collate_fn)
    elif name == ...: # todo add more models
        ...
    
    print(f"[INFO] Loaded dataset '{name}'.")
    
    return train_loader, test_loader