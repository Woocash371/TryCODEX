"""Model definitions for CFD surrogate learning on graphs."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class SimpleCFDGNN(nn.Module):
    """A minimal Graph Convolutional Network for CFD pressure prediction."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: "torch_geometric.data.Data") -> torch.Tensor:  # type: ignore[name-defined]
        """Forward pass mapping node features to pressure predictions."""

        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        return x


__all__ = ["SimpleCFDGNN"]
