from torch.utils.data import DataLoader


def load_path(
    dataset_type,
    path,
    device,
    normalization_data_dir=None,
    load_all=True,
    batch_size=100,
    normalize_parameters=True,
):
    """
    Loads a dataset into a DataLoader.

    Args:
        dataset_type (class):
            - the dataset class to load. Must match the signature of `datasets.ZarrDataset`

        path (string):
            - full path where the traces and parameters are saved

        device (torch.device):
            - the device to load the data onto

        normalization_data_dir (string, optional):
            - path where the normalization data is saved
            - if None, will use `path`

        load_all (bool, optional):
            - whether to load everything into memory at once

        batch_size (int, optional):
            - the batch size to use

        normalize_parameters (bool):
            - whether to normalize the parameters

    Returns:
        ds (dataset_type):
            - the dataset object

        loader (DataLoader):
            - the DataLoader object
    """
    ds = dataset_type(
        path,
        normalization_data_dir=normalization_data_dir,
        load_all=load_all,
        normalize_parameters=normalize_parameters,
    )
    if load_all:
        ds.traces = ds.traces.to(device)
        ds.traces_max = ds.traces_max.to(device)
        ds.traces_min = ds.traces_min.to(device)
        ds.parameters = ds.parameters.to(device)
        if normalize_parameters:
            ds.parameters_max = ds.parameters_max.to(device)
            ds.parameters_min = ds.parameters_min.to(device)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return ds, loader
