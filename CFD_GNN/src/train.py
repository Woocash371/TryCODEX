"""Training script for the CFD GNN surrogate model."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import nn

from model_gnn import SimpleCFDGNN

LOGGER = logging.getLogger(__name__)


def train(
    data_path: Path,
    output_path: Path,
    epochs: int = 500,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    dropout: float = 0.0,
    device: str | torch.device = "cpu",
) -> None:
    """Train ``SimpleCFDGNN`` on the dataset stored at ``data_path``."""

    data = torch.load(data_path)
    data = data.to(device)

    model = SimpleCFDGNN(input_dim=data.x.size(1), hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred = model(data).view(-1)
        loss = criterion(pred, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                mse = loss.item()
                mae = torch.mean(torch.abs(pred - data.y)).item()
            LOGGER.info("Epoch %d/%d - MSE: %.6f  MAE: %.6f", epoch, epochs, mse, mae)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    LOGGER.info("Training complete. Model saved to %s", output_path)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the CFD GNN surrogate model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("CFD_GNN/data/data_graph.pt"),
        help="Path to the PyTorch Geometric Data object produced by convert_mesh_to_graph.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("CFD_GNN/models/gnn_model.pt"),
        help="Where to store the trained model weights.",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension size for the GNN.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability applied after each layer.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device specifier (cpu, cuda, cuda:0, ...).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity of console logging.",
    )

    parsed = parser.parse_args(args=args)
    logging.basicConfig(level=getattr(logging, parsed.log_level), format="%(levelname)s: %(message)s")

    if not parsed.data.exists():
        parser.error(f"Dataset {parsed.data} not found. Please run convert_mesh_to_graph.py first.")

    train(
        data_path=parsed.data,
        output_path=parsed.output,
        epochs=parsed.epochs,
        lr=parsed.lr,
        hidden_dim=parsed.hidden_dim,
        dropout=parsed.dropout,
        device=torch.device(parsed.device),
    )


if __name__ == "__main__":
    main()
