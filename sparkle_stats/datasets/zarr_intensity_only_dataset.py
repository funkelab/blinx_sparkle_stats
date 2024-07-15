from sparkle_stats.datasets.zarr_dataset import ZarrDataset


class ZarrIntensityOnlyDataset(ZarrDataset):
    def __getitem__(self, item):
        trace, parameters = super().__getitem__(item)
        trace = trace[0, :].unsqueeze(0)
        return trace, parameters

    @property
    def output_classes(self):
        return 7
