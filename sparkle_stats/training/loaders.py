from torch.utils.data import DataLoader


def load_path(
    dataset_type,
    path,
    device,
    load_all=True,
    batch_size=100,
):
    ds = dataset_type(path, load_all=load_all)
    ds.traces = ds.traces.to(device)
    ds.parameters = ds.parameters.to(device)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return ds, loader
