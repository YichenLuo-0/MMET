# MMET: Multi-Input and Multi-Scale Efficient Transformer

[![Paper (IJCAI 2025)](https://img.shields.io/badge/Paper-IJCAI%202025-green)](https://github.com/YichenLuo-0/MMET)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code for MMET: A Multi-Input and Multi-Scale Transformer for Efficient PDEs Solving, presented at the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025).

## Introduction
A novel Transformer-based framework for solving partial differential equations (PDEs) efficiently in **multi-input, multi-scale, and large-scale scenarios**.

## Core Contributions

- **Multi-Input Gated Condition Embedding (GCE)**  
  Efficiently encodes PDE inputs of varying types and dimensions (e.g., boundary/initial conditions, operators, geometry).
  
- **Multi-Scale Support via Encoder-Decoder Architecture**  
  Query arbitrary resolutions without retraining, supporting zero-shot generalization on unseen mesh grids.

- **Hilbert Curve Re-serialization + Patch Embedding**  
  Spatially coherent patching of mesh inputs reduces attention complexity and boosts performance on large-scale geometry.

## Installation

```bash
git clone https://github.com/YichenLuo-0/MMET.git
cd MMET
pip install -r requirements.txt
```

Recommended environment:
- Python ≥ 3.8
- PyTorch ≥ 1.11
- GPU with ≥ 24GB memory for full-scale experiments

## Getting Started

### Train MMET on a PDE benchmark:

```bash
python train.py --config configs/heat2d.yaml
```

### Inference with pre-trained model:

```bash
python inference.py --model-checkpoint checkpoints/heat2d.pt --resolution 100x40
```

### Visualize predictions:

```bash
python plot_results.py --input data/test_case.pkl
```

## Datasets

Supported PDE benchmark datasets:

| Dataset        | Type                        | Highlights                                  |
|----------------|-----------------------------|---------------------------------------------|
| Poisson        | 2D physics-informed          | Classic PINN setup, analytical solution     |
| Shape-Net Car  | 3D aerodynamics              | Multi-scale meshes, complex geometry        |
| Darcy Flow     | 2D porous media              | PDEBench benchmark, variable diffusivity    |
| Heat2D         | 2D thermodynamics            | Multi-input, multi-geometry, dynamic BC     |
| Beam2D         | 2D solid mechanics           | Elasticity with dynamic boundary loads      |
| HeatSink2D     | Complex heat conduction      | No ground truth, physics-only supervision   |

More details and download links in [`docs/datasets.md`](docs/datasets.md).

## Citation

If you find this repository helpful, please consider citing:

```bibtex
@inproceedings{luo2025mmet,
  title={MMET: A Multi-Input and Multi-Scale Transformer for Efficient PDEs Solving},
  author={Yichen Luo and Jia Wang and Dapeng Lan and Yu Liu and Zhibo Pang},
  booktitle={Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2025}
}
```

## Contact

For questions or collaboration, feel free to contact:
- **Yichen Luo**: [yichenlu@kth.se](mailto:yichenlu@kth.se)
- **Jia Wang**: [Jia.Wang02@xjtlu.edu.cn](mailto:Jia.Wang02@xjtlu.edu.cn)
- **Zhibo Pang**: [zhibo@kth.se](mailto:yichenlu@kth.se)
