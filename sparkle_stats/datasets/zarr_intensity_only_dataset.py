from sparkle_stats.datasets.zarr_dataset import ZarrDataset


class ZarrIntensityOnlyDataset(ZarrDataset):
    """Returns only intensities for traces (no z states)."""

    def __getitem__(self, item):
        """Get a single trace.

        Returns:
            - trace (torch.Tensor): (1xT) tensor of intensities
            - parameters (torch.Tensor): (7,) tensor of parameters
        """
        trace, parameters = super().__getitem__(item)
        trace = trace[0, :].unsqueeze(0)
        return trace, parameters

    @property
    def output_classes(self):
        return 7
