from torch.utils.data import DataLoader


def load_path(
    dataset_type,
    path,
    device,
    means=None,
    std_devs=None,
    load_all=True,
    batch_size=100,
):
    ds = dataset_type(path, means=means, std_devs=std_devs, load_all=load_all)
    ds.traces = ds.traces.to(device)
    ds.parameters = ds.parameters.to(device)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return ds, loader
