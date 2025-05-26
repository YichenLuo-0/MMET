def get_dataset(dataset_name):
    if dataset_name == "Darcy Flow":
        from .darcy_flow.darcy_flow import DarcyDataset
        dataset_train = DarcyDataset(
            file_path="datasets/darcy_flow/darcy_flow.h5",
            train=True
        )
        dataset_test = DarcyDataset(
            file_path="datasets/darcy_flow/darcy_flow.h5",
            train=False
        )

    elif dataset_name == "Shape-Net Car":
        from .shape_net_car.shape_net_car import CarDataset
        dataset_train = CarDataset(
            data_path="./datasets/shape_net_car/datas_train.pkl",
            point_path="./datasets/shape_net_car/points_train.pkl"
        )
        dataset_test = CarDataset(
            data_path="./datasets/shape_net_car/datas_test.pkl",
            point_path="./datasets/shape_net_car/points_test.pkl"
        )

    elif dataset_name == "Heat2d":
        from .heat2d.heat2d import Heat2dDataset
        dataset_train = Heat2dDataset(
            file_path="datasets/heat2d/heat2d_1100_train.pkl"
        )
        dataset_test = Heat2dDataset(
            file_path="datasets/heat2d/heat2d_1100_train.pkl"
        )

    elif dataset_name == "Beam2d":
        from .beam2d.beam2d import Beam2dDataset
        dataset_train = Beam2dDataset(
            fea_path="datasets/beam2d/beam2d.rst",
            train=True
        )
        dataset_test = Beam2dDataset(
            fea_path="datasets/beam2d/beam2d.rst",
            train=False
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset_train, dataset_test
