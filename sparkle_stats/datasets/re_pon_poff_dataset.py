from .zarr_intensity_only_dataset import ZarrIntensityOnlyDataset


class RePonPoffDataset(ZarrIntensityOnlyDataset):
    def __getitem__(self, item):
        trace, parameters = super().__getitem__(item)
        parameters = parameters[:, [0, -2, -1]]
        return trace, parameters

    @property
    def output_classes(self):
        return 3
