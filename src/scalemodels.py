import pdb
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax.core.frozen_dict import FrozenDict, freeze

from src.toymodels import SimpleClassifier


class MiniCNN(nn.Module):
    """Tiny CNN for MNIST/FMNIST (also runs on CIFAR-10).
    """
    c1: int = 8
    c2: int = 16
    num_classes: int = 10
    use_bias: bool = True
    checkpoint: bool = False  # set True to remat convs

    @nn.compact
    def __call__(self, x, *, train=False):
        # Accept (B,H,W,1) or (H,W,1); (B,28,28,1) for MNIST, (B,32,32,3) for CIFAR-10
        if x.ndim == 3:   # missing batch dim
            x = x[None, ...]
        x = x.astype(jnp.float32)

        Conv = nn.Conv
        if self.checkpoint:
            Conv = nn.remat(nn.Conv)

        # Downsample aggressively with stride-2 convs
        x = Conv(self.c1, kernel_size=(3,3), strides=(2,2), padding="SAME", use_bias=self.use_bias)(x)  # 28→14 or 32→16
        x = nn.relu(x)
        x = Conv(self.c2, kernel_size=(3,3), strides=(2,2), padding="SAME", use_bias=self.use_bias)(x)  # 14→7 or 16→8
        x = nn.relu(x)

        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))   # (B, c2)

        # Linear head
        logits = nn.Dense(self.num_classes, use_bias=True)(x)
        return logits




class ConvPoolBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.features, kernel_size=(5, 5),
                    strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x

class DenseRelu(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        return x

class LeNet5(nn.Module):
    """LeNet-5 for MNIST/Fashion-MNIST with rematerialization on heavy blocks."""
    @nn.compact
    def __call__(self, x,  *args, **kwargs):
        # Ensure shape (batch, 28, 28, 1)
        if x.ndim == 3:  # missing batch dim
            x = x[None, ...]

        # Pad to 32×32 to reproduce original LeNet geometry
        x = jnp.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), mode="constant")

        CBlock = nn.remat(ConvPoolBlock)
        DBlock = nn.remat(DenseRelu)
        # CBlock = ConvPoolBlock
        # DBlock = DenseRelu

        # C1 → pool
        x = CBlock(features=6)(x)
        # C3 → pool
        x = CBlock(features=16)(x)

        # Flatten: 16 × 5 × 5 = 400
        x = x.reshape((x.shape[0], -1))

        # F5 → ReLU
        x = DBlock(features=120)(x)
        # F6 → ReLU
        x = DBlock(features=84)(x)

        # Output: 10 logits (cheap; no checkpoint)
        x = nn.Dense(features=10)(x)
        return x




class LargeClassifier(nn.Module):
    input_shape: tuple # shape of a single input
    numh: list # list of number of hidden units per layer
    numl: int # number of layers
    numc: int # number of classes
    
    @nn.compact
    def __call__(self, X,  *args, **kwargs):
        if X.shape == self.input_shape:
            X = X.reshape(-1)
        else:
            X = X.reshape(X.shape[0], -1)
        for j in range(self.numl):
            X = nn.tanh(nn.Dense(features=self.numh[j])(X))
        logits = nn.Dense(features=self.numc)(X)
        return logits


class BasicBlock(nn.Module):
    """A single residual block with two 3x3 convolutions."""
    channels: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, *, train: bool):
        residual = x

        # First conv + BN + ReLU
        x = nn.Conv(
            features=self.channels,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Second conv + BN
        x = nn.Conv(
            features=self.channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        # If the spatial resolution or channel dims changed, downsample the residual
        if residual.shape != x.shape:
            residual = nn.Conv(
                features=self.channels,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False
            )(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        x = nn.relu(x + residual)
        return x


class ResNet1M(nn.Module):
    """
    ResNet-like architecture with ~1M parameters.
    3 x 3 BasicBlocks each, with channel widths [32, 64, 128].
    """
    num_classes: int

    @nn.compact
    def __call__(self, x, *, train: bool):
        # If input is grayscale (C=1), replicate to 3 channels
        if x.ndim == 3:  # missing batch dim
            x = x[None, ...]
        if x.shape[-1] == 1:
            x = jnp.tile(x, (1, 1, 1, 3))

        # conv (3×3, stride=1, 32 channels)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # 3 × BasicBlock at 32 channels, no downsampling
        for _ in range(3):
            x = BasicBlock(channels=32, stride=1)(x, train=train)

        # first block downsamples, then 2 blocks at 64 channels
        x = BasicBlock(channels=64, stride=2)(x, train=train)
        for _ in range(2):
            x = BasicBlock(channels=64, stride=1)(x, train=train)

        # first block downsamples, then 2 blocks at 128 channels
        x = BasicBlock(channels=128, stride=2)(x, train=train)
        for _ in range(2):
            x = BasicBlock(channels=128, stride=1)(x, train=train)
        
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(features=self.num_classes)(x)
        return x


EMPTY_STATS = freeze({}) # used by models without BN
class TrainState(train_state.TrainState):
    batch_stats: FrozenDict = EMPTY_STATS
    alpha = None # optional optimized alpha


def get_model(model_cfg):
    model_name = model_cfg['name']
    
    if model_name == "LeNet5":
        return LeNet5()
    elif model_name == "MiniCNN":
        model = MiniCNN(c1=model_cfg.get("c1", 8), 
                        c2=model_cfg.get("c2", 16), 
                        num_classes=10, 
                        checkpoint=model_cfg.get("checkpoint", False)
        )
        return model
    elif model_name == "large_classifier":
        input_shape = tuple(model_cfg["input_shape"])
        num_h = model_cfg["num_h"]
        num_l = model_cfg["num_l"]
        num_c = model_cfg.get("num_c")
        return LargeClassifier(input_shape=input_shape, numh=num_h, numl=num_l, numc=num_c)
    elif model_name == "classifier":
        num_h = model_cfg["num_h"]
        num_l = model_cfg["num_l"]
        num_c = model_cfg.get("num_c")
        return SimpleClassifier(numh=num_h, numl=num_l, numc=num_c)
    elif model_name == "ResNet1":
        num_c = model_cfg.get("num_c")
        return ResNet1M(num_classes=num_c)
    else:
        raise ValueError(f"Unknown model name: {model_name}")