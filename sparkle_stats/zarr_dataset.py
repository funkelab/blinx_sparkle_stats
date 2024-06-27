import os

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class ZarrDataset(Dataset):
    """A dataset stored on the filesystem as a trace zarr and a parameter zarr."""

    def __init__(self, data_dir, load_all=False):
        """
        Args:
            data_dir (string):
                - path where the traces and parameters are saved
            load_all (bool, optional):
                - whether to load everything into memory at once
        """

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory {data_dir} not found")
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"Found file at {data_dir} expected directory")

        self.traces_path = os.path.join(data_dir, "traces")
        self.parameters_path = os.path.join(data_dir, "parameters")

        if not os.path.exists(self.traces_path):
            raise FileNotFoundError(f"Can't find traces at {self.traces_path}")
        if not os.path.exists(self.parameters_path):
            raise FileNotFoundError(f"Can't find parameters at {self.parameters_path}")

        self.traces = zarr.open(self.traces_path, mode="r")
        self.parameters = zarr.open(self.parameters_path, mode="r")

        self.load_all = load_all
        if load_all:
            # load into memory
            self.traces = np.array(self.traces)
            self.parameters = np.array(self.parameters)
            # convert to torch
            self.traces = torch.from_numpy(self.traces.astype(np.float32))
            self.parameters = torch.from_numpy(self.parameters.astype(np.float32))

        if self.traces.ndim != 3:
            raise ValueError(f"Expected 3d zarr array, found {self.traces.ndim}d")
        if self.parameters.ndim != 2:
            raise ValueError(f"Expected 2d zarr array, found {self.parameters.ndim}d")

        self.trace_count = self.traces.shape[0]
        self.trace_length = self.traces.shape[1]

        parameters_length = self.parameters.shape[0]
        if parameters_length != self.trace_count:
            raise ValueError(
                f"Found {parameters_length} parameters but {self.trace_count} traces"
            )

    def __len__(self):
        return self.trace_count

    def __getitem__(self, item):
        # is currently NxTxC
        # reshape into CxT
        trace = self.traces[item, :, :].reshape(2, -1)
        parameters = self.parameters[item, :].reshape(-1)

        if not self.load_all:
            trace = torch.from_numpy(trace.astype(np.float32))
            parameters = torch.from_numpy(parameters.astype(np.float32))

        return trace, parameters

    @property
    def output_classes(self):
        return 7


class ZarrIntensityOnlyDataset(ZarrDataset):
    def __getitem__(self, item):
        trace, parameters = super().__getitem__(item)
        trace = trace[0, :].unsqueeze(0)
        return trace, parameters

    @property
    def output_classes(self):
        return 7


class ZarrStateOnlyDataset(ZarrDataset):
    def __getitem__(self, item):
        trace, parameters = super().__getitem__(item)
        trace = trace[1, :].unsqueeze(0)
        parameters = parameters[[5, 6]]
        return trace, parameters

    @property
    def output_classes(self):
        return 2
