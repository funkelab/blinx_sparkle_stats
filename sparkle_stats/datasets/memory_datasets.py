import os

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class MemoryTraceDataset(Dataset):
    """Loads an entire zarr into memory.

    Use only if there's enough RAM on the machine to load everything."""

    def __init__(self, data_dir):
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory {data_dir} not found")
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"Found file at {data_dir} expected directory")

        traces_path = os.path.join(data_dir, "traces")
        parameters_path = os.path.join(data_dir, "parameters")
        if not os.path.exists(traces_path):
            raise FileNotFoundError(f"Can't find traces at {traces_path}")
        if not os.path.exists(parameters_path):
            raise FileNotFoundError(f"Can't find parameters at {parameters_path}")

        self.traces = np.array(zarr.open(traces_path, mode="r"))
        self.parameters = np.array(zarr.open(parameters_path, mode="r"))

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
        return (
            torch.from_numpy(self.traces[item, :, :].reshape(2, -1).astype(np.float32)),
            torch.from_numpy(self.parameters[item, :].reshape(-1).astype(np.float32)),
        )


class MemoryIntensityOnlyDataset(MemoryTraceDataset):
    """Loads only intensities from a zarr into memory.

    Use only if there's enough RAM on the machine to load everything."""

    def __getitem__(self, item):
        trace, parameters = super().__getitem__(item)
        # is currently CxT
        # reshape into 1xT
        trace = trace[0, :].unsqueeze(0)
        return trace, parameters


class MemoryStateOnlyDataset(MemoryTraceDataset):
    """Loads only states from a zarr into memory.

    Use only if there's enough RAM on the machine to load everything."""

    def __getitem__(self, item):
        trace, parameters = super().__getitem__(item)
        trace = trace[1, :].unsqueeze(0)
        return trace, parameters
