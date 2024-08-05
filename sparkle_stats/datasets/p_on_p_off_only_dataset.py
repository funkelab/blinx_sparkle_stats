from sparkle_stats.datasets.zarr_intensity_only_dataset import ZarrIntensityOnlyDataset


class PonPoffOnlyDataset(ZarrIntensityOnlyDataset):
    """Pairs intensity traces with only p_on and p_off."""

    def __getitem__(self, item):
        t, p = super().__getitem__(item)
        # only show p_on and p_off
        p = p[:, [5, 6]]
        print(p.shape)
        return t, p

    @property
    def output_classes(self):
        return 2
