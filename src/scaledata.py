import jax
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import jax.numpy as jnp
import numpy as np

from src.data import (
    JAXDataset, NumpyDataset,
    get_dataloaders as _get_dataloaders,
    jax_collate_fn, numpy_collate_fn,
)

def load_mnist_numpy(train=True):
    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(root='./data', train=train,
                        download=True, transform=transform)
    imgs, labs = [], []
    for img, lab in ds:
        imgs.append(img.reshape(28, 28, 1))
        labs.append(lab)
    return np.stack(imgs, axis=0), np.array(labs, dtype=np.int32)

def load_fmnist_numpy(train=True):
    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.FashionMNIST(root='./data', train=train,
                              download=True, transform=transform)
    imgs, labs = [], []
    for img, lab in ds:
        imgs.append(img.reshape(28, 28, 1))
        labs.append(lab)
    return np.stack(imgs, axis=0), np.array(labs, dtype=np.int32)


def _cifar10_transform(train=True, aug=True):
    if not train or not aug:
        pipeline = [
            transforms.ToTensor(),
            # transforms.Normalize((0.4914,0.4822,0.4465),
            #             (0.2023,0.1994,0.2010)),
        ]
    else:
        pipeline = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914,0.4822,0.4465),
            #             (0.2023,0.1994,0.2010)),
        ]
    return transforms.Compose(pipeline)

def _load_cifar10_numpy(train=True, aug=None):
    """Return (N,32,32,3) float32 [0,1], (N,) int32."""
    # tfm = transforms.Compose([transforms.ToTensor()])  # C,H,W float32
    if aug is None: aug = train
    ds  = datasets.CIFAR10(root='./data', train=train,
                           download=True, 
                           transform=_cifar10_transform(train, aug=aug))
    imgs, labs = [], []
    for img, lab in ds:
        # (3,32,32)  -> (32,32,3)
        imgs.append(np.transpose(img.numpy(), (1, 2, 0)))
        labs.append(lab)
    return np.stack(imgs).astype(np.float32), np.array(labs, np.int32)

def get_dataloaders(name, batch_size, num_workers=0, aug=True):
    """
    Returns three loaders: train, test, val.
    For MNIST/FMNIST we split the *train* set into 90% train / 10% val.
    The original test set is used as 'test'.
    """
    if name == 'mnist':
        x_all, y_all = load_mnist_numpy(train=True)
        x_test, y_test = load_mnist_numpy(train=False)

    elif name == 'fmnist':
        x_all, y_all   = load_fmnist_numpy(train=True)
        x_test, y_test = load_fmnist_numpy(train=False)
        
    elif name == 'cifar10':
        x_all, y_all   = _load_cifar10_numpy(train=True, aug=aug)
        x_test, y_test = _load_cifar10_numpy(train=False)

    else:
        raise ValueError(f"Unknown dataset name '{name}'")

    # ---- split train→train/val (last 2.5% val) ----
    n_total   = x_all.shape[0]
    n_val     = int(0.02 * n_total)
    n_train   = n_total - n_val

    x_train = x_all[:n_train]
    y_train = y_all[:n_train]
    x_val   = x_all[n_train:]
    y_val   = y_all[n_train:]

    # ---- wrap in datasets & loaders ----
    train_ds = NumpyDataset(x_train, y_train)
    test_ds  = NumpyDataset(x_test,  y_test)
    val_ds   = NumpyDataset(x_val,   y_val)

    train_loader, test_loader, val_loader = _get_dataloaders(
        train_ds, test_ds, val_ds,
        batch_size,
        collate_fn=numpy_collate_fn,
    )

    print(f"[INFO] Loaded '{name}'  •  "
          f"train={len(x_train)}  val={len(x_val)}  test={len(x_test)}")
    return train_loader, test_loader, val_loader
