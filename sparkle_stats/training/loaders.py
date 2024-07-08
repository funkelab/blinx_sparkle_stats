from torch.utils.data import DataLoader


def load_path(
    dataset_type,
    path,
    device,
    normalization_data_dir=None,
    load_all=True,
    batch_size=100,
):
    ds = dataset_type(
        path,
        normalization_data_dir=normalization_data_dir,
        load_all=load_all,
    )
    ds.traces = ds.traces.to(device)
    ds.traces_mean = ds.traces_mean.to(device)
    ds.parameters = ds.parameters.to(device)
    ds.parameters_means = ds.parameters_means.to(device)
    ds.parameters_std_devs = ds.parameters_std_devs.to(device)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return ds, loader
