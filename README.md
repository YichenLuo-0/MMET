# MMET: Multi-Input and Multi-Scale Efficient Transformer

[![Paper (IJCAI 2025)](https://img.shields.io/badge/Paper-IJCAI%202025-green)](https://github.com/YichenLuo-0/MMET)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code for **MMET: A Multi-Input and Multi-Scale Transformer for Efficient PDEs Solving**, presented at *the 34th
International Joint Conference on Artificial Intelligence (IJCAI 2025)*.

## Introduction

**MMET (Multi-Input and Multi-Scale Efficient Transformer)** is a Transformer-based framework tailored for solving
partial differential equations (PDEs) in complex scientific and engineering domains. Traditional neural PDE solvers
often struggle with generalization across varying geometries, boundary/initial conditions, and resolution scales. MMET
addresses these limitations through a unified and scalable architecture.

<img src="fig/introduction.png" alt="Introduction" width="500"/>

Unlike existing methods that require model retraining for each new instance, MMET introduces a flexible encoder-decoder
structure that separates mesh and query representations. This design enables:

- **Zero-shot generalization** across unseen geometries and boundary conditions,
- **Multi-scale querying** without retraining,
- And **reduced computational overhead** when handling large-scale unstructured meshes.

At the core of MMET are three key innovations:

- A **Gated Condition Embedding (GCE)** layer for dynamically encoding inputs with different types and dimensions;
- A **Hilbert-curve-based patch embedding** strategy to preserve spatial locality and reduce attention complexity;
- And a **multi-scale decoder** that directly supports adaptive resolution inference.

These components enable MMET to outperform state-of-the-art baselines across diverse physics benchmarks, including
elasticity, fluid mechanics, thermodynamics, and porous media flow.

<img src="fig/architecture.png" alt="Architecture" width="800"/>

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

## Directory Layout

```plaintext
MMET/
├── datasets/                  # Dataset definitions and utilities
│   ├── README.md              # Dataset overview and download instructions
│   ├── __init__.py            # Package initialization
│   ├── datasets.py            # Dataset loading and processing utilities
│   ├── datasets_pinn.py       # Physics-informed dataset utilities
│   ├── beam2d/                # Beam2D dataset implementation
│   │   ├── beam2d.py          # Beam2D dataset class
│   │   └── data_generation.py # Data generation script for Beam2D
│   ├── darcy_flow/            # Darcy Flow dataset implementation
│   │   └── darcy_flow.py      # Darcy Flow dataset class
│   ├── heat2d/                # Heat2D dataset implementation
│   │   └── heat2d.py          # Heat2D dataset class
│   ├── heat_sink2d/           # HeatSink2D dataset implementation
│   │   └── heat_sink2d.py     # HeatSink2D dataset class
│   ├── poisson/               # Poisson dataset implementation
│   │   └── poisson.py         # Poisson dataset class
│   └── shape_net_car/         # Shape-Net Car dataset implementation
│       └── shape_net_car.py   # Shape-Net Car dataset class
├── model/                     # MMET model implementation
│   ├── __init__.py            # Package initialization
│   ├── mmet.py                # MMET model definition
│   ├── activation_func.py     # Custom WaveAct activation functions
│   ├── embedding.py           # Gated Condition Embedding (GCE) layer
│   ├── hilbert.py             # Hilbert curve utilities for patch embedding
│   ├── patching.py            # Patch embedding utilities
│   └── serialization.py       # Model serialization utilities
├── train.py                   # Training script for MMET
├── train_pinn.py              # Training script for physics-informed PINNs
├── inference.py               # Inference script for MMET
├── requirements.txt           # Python package dependencies
├── README.md                  # Project overview and instructions
├── LICENSE                    # License information
└── fig/                       # Figures for documentation
```

## Getting Started

### Train MMET on a PDE benchmark:

We provide training examples of MMET on the Darcy Flow dataset. You can easily adapt the code to train on other datasets
by modifying the dataloader and model parameters in the `train.py` file. To test this, run the following command:

```bash
python train.py --epochs 2000 --batch_size 4 --lr 1e-3
```

For physics-driven datasets, we provide training scripts in the `train_pinn.py` file. You can run it with:

```bash
python train_pinn.py --epochs 2000 --lr 1e-1
```

### Inference with pre-trained model:

To perform inference using a pre-trained MMET model, you can use the provided `inference.py` script by running the
following command:

```bash
python inference.py
```

## Datasets

Supported PDE benchmark datasets:

| Dataset       | Type                    | Training Method | Highlights                                |
|---------------|-------------------------|-----------------|-------------------------------------------|
| Poisson       | 2D physics-informed     | Physics-driven  | Classic PINN setup, analytical solution   |
| Shape-Net Car | 3D aerodynamics         | Data-driven     | Multi-scale meshes, complex geometry      |
| Darcy Flow    | 2D porous media         | Data-driven     | PDEBench benchmark, variable diffusivity  |
| Heat2D        | 2D thermodynamics       | Data-driven     | Multi-input, multi-geometry, dynamic BC   |
| Beam2D        | 2D solid mechanics      | Data-driven     | Elasticity with dynamic boundary loads    |
| HeatSink2D    | Complex heat conduction | Physics-driven  | No ground truth, physics-only supervision |

More details and download links in [`datasets/README.md`](datasets/README.md).

## Citation

If you find this paper or repository helpful, please consider citing:

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

## Acknowledgements

This repository is built upon the following open-source projects:

- [PDEBench dataset](https://github.com/pdebench/PDEBench)
- [GNOT](https://github.com/thu-ml/GNOT)
- [PinnsFormer](https://github.com/AdityaLab/pinnsformer)
- [Transolver](https://github.com/thuml/Transolver)