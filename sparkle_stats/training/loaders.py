from torch.utils.data import DataLoader


def load_path(
    dataset_type,
    path,
    device,
    trace_mean=None,
    parameter_means=None,
    parameter_std_devs=None,
    load_all=True,
    batch_size=100,
):
    ds = dataset_type(
        path,
        trace_mean=trace_mean,
        parameter_means=parameter_means,
        parameter_std_devs=parameter_std_devs,
        load_all=load_all,
    )
    ds.traces = ds.traces.to(device)
    ds.parameters = ds.parameters.to(device)
    ds.parameter_means = ds.parameter_means.to(device)
    ds.parameter_std_devs = ds.parameter_std_devs.to(device)
    ds.trace_mean = ds.trace_mean.to(device)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return ds, loader
