# Datasets for MMET

This folder contains the datasets used to benchmark and train the **MMET** framework, as introduced in the Appendix in
our ArXiv version paper:

> **MMET: A Multi-Input and Multi-Scale Transformer for Efficient PDEs Solving**  
> [[ArXiv]](https://github.com/YichenLuo-0/MMET)

These datasets span various physical domains and are designed to test MMET’s capabilities in handling:

- Multi-input PDE problems (e.g., multiple boundary types)
- Multi-scale query resolutions
- Complex geometries and real-world simulations

## Physics-Driven Datasets

### Poisson

- **Domain**: 2D rectangular
- **Governing Eq.**: $-\\nabla^2 u = f(x, y)$
- **BCs**: Dirichlet (u = 0)
- **Ground truth**: Analytical solution
- **Train/Test**: [50×50] / [100×100] grid

### HeatSink2d

- **Domain**: 2D Thermodynamic heat sink
- **Governing Eq.**: Steady-state heat conduction
- **BCs**: Dirichlet + Neumann
- **Ground truth**: FEM from Ansys Workbench
- **Note**: Test data = 10 unique FEM samples (to be released)

## Data-Driven Datasets

### Darcy Flow

- **Domain**: 2D porous media
- **Task**: Map diffusion coefficients to pressure field
- **Use case**: Operator learning for fluid simulation
- **Train/Test**:
    - Train: 9,900 samples × 1,000 points
    - Test: 100 samples × 1,000 points
- **Source**: [PDEBench Dataset](https://github.com/pdebench/PDEBench)

### Shape-Net Car

- **Domain**: 3D aerodynamic car shapes
- **Task**: Predict surface pressure $p$ and velocity $v$ from car geometry
- **Train/Test**:
    - Train: 789 samples × 3,268 points
    - Test: 100 samples × 3,268 points
- **Note**: Only 3,268 car surface points are used for predict pressure. Code for training on the full mesh will be
  released later.
- **Source**: [ShapeNet CFD Data](http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip)

### Heat2d

- **Domain**: 2D heat conduction
- **Task**: Predict temperature field $T$ under varying conditions and multi-scale resolution
- **Train/Test**:
    - Train: 1,000 samples (the number of points varies for each data)
    - Test: 100 samples
- **Note**: The original dataset has two versions, one with 5,500 samples and another reduced version with 1,100
  samples. The original authors only open-sourced the reduced version, which we use in our experiments.
- **Source**: [GNOT Dataset (Google Drive)](https://drive.google.com/drive/folders/1kicZyL1t4z6a7B-6DJEOxIrX877gjBC0)

### Beam2d

- **Domain**: Elastic beam under bending moment
- **Task**: Predict displacement and stress fields under varying boundary conditions
- **Geometry**: FEM mesh (5,404 nodes) from Ansys
- **Train/Test**:
    - Train: 1,000 samples × 1,000 points
    - Test: 100 samples × 5,000 points
- **Outputs**: Displacement ($u, v$), Stress ($\\sigma_x, \\sigma_y, \\tau_{xy}$)
- **Special**: Resolution of query points differs across training/testing
- **Note**: In our paper,we uses von Mises stress to evaluate the stress field, but the dataset provides stress
  components ($\\sigma_x, \\sigma_y, \\tau_{xy}$) for more detailed analysis.
- **Source**: will be released later