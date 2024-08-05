import os

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from sparkle_stats.generate_dataset import get_file_paths_from_base


class ZarrDataset(Dataset):
    """A dataset stored on the filesystem as zarr and numpy arrays.

    The dataset is expected to have the following directories:
        - `traces/`: (BxTx2) traces zarr array
        - `parameters/`: (Bx7) parameters zarr array
        - `y/`: (Bx1) y zarr array

    If `normalized_parameters=True`, it is also expected to have the following files:
        - `traces_max.npy`: numpy array with a single element, the maximum intensity value across all traces in the dataset
        - `traces_min.npy`: numpy array with a single element, the minimum intensity value across all traces in the dataset
        - `parameters_max.npy`: numpy array with a single element, the maximum parameter value across all parameters in the dataset
        - `parameters_min.npy`: numpy array with a single element, the minimum parameter value across all parameters in the dataset

    If the dataset is generated with `sparkle_stats.generate_dataset`, the above structure will be created.
    """

    def __init__(
        self,
        data_dir,
        normalization_data_dir=None,
        normalize_parameters=True,
        load_all=False,
    ):
        """
        Args:
            data_dir (string):
                - path where the traces and parameters are saved
            normalization_data_dir (string, optional):
                - path where the normalization data is saved
                - if None, will use `data_dir`
            normalize_parameters (bool):
                - whether to normalize the parameters
            load_all (bool, optional):
                - whether to load everything into memory at once
        """

        _directory_exists(data_dir)

        if normalization_data_dir is None:
            normalization_data_dir = data_dir
        _directory_exists(normalization_data_dir)

        traces_path, _, _, parameters_path, _, _, _ = get_file_paths_from_base(data_dir)

        (
            _,
            traces_max_path,
            traces_min_path,
            _,
            parameters_max_path,
            parameters_min_path,
            _,
        ) = get_file_paths_from_base(normalization_data_dir)

        _path_exists(traces_path, "traces")
        _path_exists(traces_min_path, "traces min")
        _path_exists(traces_max_path, "traces max")
        _path_exists(parameters_path, "parameters")

        self.normalize_parameters = normalize_parameters
        if self.normalize_parameters:
            _path_exists(parameters_min_path, "parameters min")
            _path_exists(parameters_max_path, "parameters max")

        traces = zarr.open(traces_path, mode="r")
        parameters = zarr.open(parameters_path, mode="r")

        self.load_all = load_all
        if self.load_all:
            self.traces = torch.from_numpy(np.array(traces).astype(np.float32))
            self.parameters = torch.from_numpy(np.array(parameters).astype(np.float32))
        else:
            self.traces = traces
            self.parameters = parameters

        self.trace_count = self.traces.shape[0]
        self.trace_length = self.traces.shape[1]

        parameters_length = self.parameters.shape[0]
        if parameters_length != self.trace_count:
            raise ValueError(
                f"Found {parameters_length} parameters but {self.trace_count} traces"
            )

        self.traces_max = torch.from_numpy(np.load(traces_max_path).astype(np.float32))
        self.traces_min = torch.from_numpy(np.load(traces_min_path).astype(np.float32))
        if self.normalize_parameters:
            self.parameters_max = torch.from_numpy(
                np.load(parameters_max_path).astype(np.float32)
            )
            self.parameters_min = torch.from_numpy(
                np.load(parameters_min_path).astype(np.float32)
            )

    def __len__(self):
        return self.trace_count

    def __getitem__(self, item):
        """Get a single trace.

        Returns:
            - trace (torch.Tensor): (2xT) tensor of intensity and z states
            - parameters (torch.Tensor): (7,) tensor of parameters
        """
        # is currently NxTxC
        # reshape into CxT
        raw_trace = self.traces[item, :, :].T
        raw_parameters = self.parameters[item, :].reshape(-1)

        if not self.load_all:
            tensor_trace = torch.from_numpy(raw_trace.astype(np.float32))
            tensor_parameters = torch.from_numpy(raw_parameters.astype(np.float32))
        else:
            # will already be tensors from the constructor
            # clone to avoid overwriting the backing data
            tensor_trace = raw_trace.clone()
            tensor_parameters = raw_parameters.clone()

        trace_intensity = (tensor_trace[0] - self.traces_min) / (
            self.traces_max - self.traces_min
        ) * 2 - 1
        trace_z = tensor_trace[1]
        trace = torch.vstack((trace_intensity, trace_z))

        if self.normalize_parameters:
            parameters = (tensor_parameters - self.parameters_min) / (
                self.parameters_max - self.parameters_min
            ) * 2 - 1
        else:
            parameters = tensor_parameters.unsqueeze(0)

        return trace, parameters

    @property
    def output_classes(self):
        """Number of parameters __getitem__ will return"""
        return 7


def _directory_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory {path} not found")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Found file at {path} expected directory")


def _path_exists(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Can't find {name} at {path}")
