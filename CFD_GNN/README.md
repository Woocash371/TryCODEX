# CFD_GNN

Minimal example of a CFD-to-GNN pipeline that converts OpenFOAM simulations into
graph data and trains a simple graph neural network (GNN) surrogate to predict
pressure from velocity and spatial coordinates.

## Project structure

```
CFD_GNN/
├── data/
│   └── case_0001.vtu          # Sample mesh exported via foamToVTK (placeholder included)
├── src/
│   ├── convert_mesh_to_graph.py
│   ├── model_gnn.py
│   ├── train.py
│   └── visualize.py
├── requirements.txt
└── README.md
```

The provided `case_0001.vtu` file is a tiny illustrative mesh. Replace it with
your own export from OpenFOAM by running `foamToVTK` inside the simulation case
and copying the resulting `.vtu` file into `data/`.

If the supplied VTU file is missing velocity (`U`) or pressure (`p`) fields, or
if the file is not found, the conversion script automatically synthesises a
smooth example field so the rest of the pipeline continues to function.

## Getting started

1. Install dependencies (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```
2. Convert the VTU mesh into a PyTorch Geometric graph:
   ```bash
   python src/convert_mesh_to_graph.py --mesh data/case_0001.vtu --output data/data_graph.pt
   ```
3. Train the GNN surrogate:
   ```bash
   python src/train.py --data data/data_graph.pt --output models/gnn_model.pt
   ```
4. Visualise the predicted pressure field:
   ```bash
   python src/visualize.py --data data/data_graph.pt --model models/gnn_model.pt --output figures/prediction.png
   ```

All scripts accept additional options (e.g. log level, device, dropout) that can
be inspected with the `--help` flag.

## Extending to 3D

Pass `--include-z` to `convert_mesh_to_graph.py` to include the z-coordinate and
`U_z` component in the node features. The remainder of the pipeline will
automatically adapt to the new feature dimensionality.

## Data provenance

The workflow assumes CFD results exported from OpenFOAM using:

```bash
foamToVTK
```

Place the generated `.vtu` files into the `data/` directory and update the
`--mesh` argument accordingly.
