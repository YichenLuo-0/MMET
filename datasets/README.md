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
- **Equation**: $-\\nabla^2 u = f(x, y)$
- **BCs**: Dirichlet (u = 0)
- **Ground truth**: Analytical
- **Train/Test**: [50×50] / [100×100] grid
- **Use case**: Physics-informed pretraining baseline

### HeatSink2d

- **Domain**: Thermodynamic heat sink
- **Governing Eq.**: Steady-state heat conduction
- **Features**: Dynamic geometry, bottom BC shape, no labels (physics-only training)
- **BCs**: Dirichlet + Neumann
- **Format**: PDE loss only
- **Ground truth**: FEM from Ansys
- **Note**: Test data = 10 unique FEM samples (to be released)

## Data-Driven Datasets

### Darcy Flow

- **Domain**: 2D porous media
- **Task**: Map diffusion coefficients to pressure field
- **Format**: Spatially varying coefficients and targets
- **Use case**: Operator learning for fluid simulation
- **Train/Test**: 10,000 samples × 1,000 points
- **Source**: [PDEBench](https://github.com/pdebench/PDEBench)

### Shape-Net Car

- **Domain**: 3D aerodynamic car shapes
- **Task**: Predict surface pressure $p$
- **Geometry**: Complex mesh, 32,186 points per sample
- **Train/Test split**: 789 / 100
- **Note**: Only 3,268 car surface points are used for prediction
- **Source**: [ShapeNet CFD Data](http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip)

### Heat2d

- **Domain**: 2D heat conduction
- **Features**: Multi-scale resolution, dynamic BCs
- **Versions**:
    - Full: 5,500 samples
    - Reduced: 1,100 samples
- **Format**: PDE + labeled supervision
- **Source**: [GNOT Dataset (Google Drive)](https://drive.google.com/drive/folders/1kicZyL1t4z6a7B-6DJEOxIrX877gjBC0)

### Beam2d

- **Domain**: Elastic beam under bending moment
- **Outputs**: Displacement ($u, v$), Stress ($\\sigma_x, \\sigma_y, \\tau_{xy}$)
- **Geometry**: FEM mesh (5,404 nodes) from Ansys
- **Train/Test**:
    - Train: 1,000 samples × 1,000 points
    - Test: 100 samples × 5,000 points
- **Special**: Resolution of query points differs across training/testing
- **Note**: Uses von Mises stress for evaluation



