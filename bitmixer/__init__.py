from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


def binarize(x: torch.Tensor) -> torch.Tensor:
    return F.tanh(2 * x)


class BitMixerBlock(nn.Module):
    """BitMixer block.

    Example
    -------
    >>> module = BitMixerBlock(embedding_dimension=256, sequence_length=1024)
    >>> x = torch.randn((1, 1024, 256))
    >>> x = module(x)  # Shape: (1, 1024, 256).
    """

    def __init__(
        self, 
        *, 
        embedding_dimension: int, 
        sequence_length: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        sequence_length : int
            The sequence length.
        """

        super().__init__()

        self.linear_1 = nn.Linear(
            in_features=sequence_length,
            out_features=embedding_dimension * 3,
        )

        self.linear_2 = nn.Linear(
            in_features=embedding_dimension * 3,
            out_features=sequence_length,
        )

        self.linear_3 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension * 3,
        )

        self.linear_4 = nn.Linear(
            in_features=embedding_dimension * 3,
            out_features=embedding_dimension,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module."""

        # Inter-token mixing.

        z = rearrange(x, 'b l e -> b e l')
        z = z @ binarize(self.linear_1.weight.T) + binarize(self.linear_1.bias)
        z = binarize(z)
        z = z @ binarize(self.linear_2.weight.T) + binarize(self.linear_2.bias)
        z = binarize(z)

        # Intra-token mixing.

        u = binarize(rearrange(z, 'b e l -> b l e') + x)
        z = u @ binarize(self.linear_3.weight.T) + binarize(self.linear_3.bias)
        z = binarize(z)
        z = z @ binarize(self.linear_4.weight.T) + binarize(self.linear_4.bias)
        z = binarize(z + u)

        return z


@dataclass(frozen=True)
class BitMixerConfiguration:
    embedding_dimension: int
    sequence_length: int
    layers: int


class BitMixer(nn.Module):
    """BitMixer.

    Example
    -------
    >>> configuration = BitMixerConfiguration(
    ...     embedding_dimension=256,
    ...     sequence_length=1024,
    ...     layers=16,
    ... )
    >>> module = BitMixer(configuration=configuration)
    >>> x = torch.randn((1, 1024, 256))
    >>> x = module(x)  # Shape: (1, 1024, 256).
    """

    def __init__(self, *, configuration: BitMixerConfiguration) -> None:
        """Initialize the module.

        Parameters
        ----------
        configuration : BitMixerConfiguration
            The module configuration.
        """

        super().__init__()

        self.layers = nn.ModuleList([
            BitMixerBlock(
                embedding_dimension=configuration.embedding_dimension,
                sequence_length=configuration.sequence_length,
            ) for _ in range(configuration.layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module."""

        for layer in self.layers:
            x = layer(x)
        
        return x
