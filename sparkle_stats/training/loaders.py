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
