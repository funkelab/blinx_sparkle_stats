from sparkle_stats.datasets.zarr_dataset import ZarrDataset


class ZarrStateOnlyDataset(ZarrDataset):
    def __getitem__(self, item):
        trace, parameters = super().__getitem__(item)
        trace = trace[1, :].unsqueeze(0)
        parameters = parameters[:, [5, 6]]
        return trace, parameters

    @property
    def output_classes(self):
        return 2
