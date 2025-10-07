"""Visualisation utilities for inspecting GNN predictions."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from model_gnn import SimpleCFDGNN

LOGGER = logging.getLogger(__name__)


def plot_predictions(
    data_path: Path,
    model_path: Path,
    output_path: Path | None = None,
    device: str | torch.device = "cpu",
) -> None:
    """Load a trained model and produce a scatter plot of predicted pressure."""

    if not data_path.exists():
        raise FileNotFoundError(f"Graph dataset {data_path} does not exist. Run convert_mesh_to_graph.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint {model_path} not found. Train the model via train.py.")

    device = torch.device(device)
    data = torch.load(data_path).to(device)

    model = SimpleCFDGNN(input_dim=data.x.size(1)).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        predictions = model(data).view(-1).cpu().numpy()

    coords = data.pos.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=predictions, cmap="viridis", s=40)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Predicted pressure field")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(scatter, ax=ax, label="Pressure")
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        LOGGER.info("Saved visualisation to %s", output_path)
    else:
        plt.show()

    plt.close(fig)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Visualise pressure predictions from the trained GNN.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("CFD_GNN/data/data_graph.pt"),
        help="Path to the saved PyTorch Geometric Data object.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("CFD_GNN/models/gnn_model.pt"),
        help="Checkpoint produced by train.py containing trained weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for saving the scatter plot instead of displaying it.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device specifier (cpu, cuda, ...).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity of console logging.",
    )

    parsed = parser.parse_args(args=args)
    logging.basicConfig(level=getattr(logging, parsed.log_level), format="%(levelname)s: %(message)s")

    plot_predictions(
        data_path=parsed.data,
        model_path=parsed.model,
        output_path=parsed.output,
        device=parsed.device,
    )


if __name__ == "__main__":
    main()
