import os

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from sparkle_stats.generate_dataset import get_file_paths_from_base


class ZarrDataset(Dataset):
    """A dataset stored on the filesystem as a trace zarr and a parameter zarr."""

    def __init__(
        self,
        data_dir,
        normalization_data_dir=None,
        load_all=False,
    ):
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

        if normalization_data_dir is None:
            normalization_data_dir = data_dir

        if not os.path.exists(normalization_data_dir):
            raise FileNotFoundError(f"Directory {data_dir} not found")
        if not os.path.isdir(normalization_data_dir):
            raise NotADirectoryError(f"Found file at {data_dir} expected directory")

        traces_path, _, parameters_path, _, _, _ = get_file_paths_from_base(data_dir)

        _, traces_mean_path, _, parameters_means_path, parameters_std_devs_path, _ = (
            get_file_paths_from_base(normalization_data_dir)
        )

        if not os.path.exists(traces_path):
            raise FileNotFoundError(f"Can't find traces at {traces_path}")
        if not os.path.exists(parameters_path):
            raise FileNotFoundError(f"Can't find parameters at {parameters_path}")
        if not os.path.exists(traces_mean_path):
            raise FileNotFoundError(f"Can't find trace mean at {traces_mean_path}")
        if not os.path.exists(parameters_means_path):
            raise FileNotFoundError(
                f"Can't find parameters means at {parameters_means_path}"
            )
        if not os.path.exists(parameters_std_devs_path):
            raise FileNotFoundError(
                f"Can't find parameters std devs at {parameters_std_devs_path}"
            )

        self.traces = zarr.open(traces_path, mode="r")
        self.traces_mean = np.load(traces_mean_path)
        self.parameters = zarr.open(parameters_path, mode="r")
        self.parameters_means = np.load(parameters_means_path)
        self.parameters_std_devs = np.load(parameters_std_devs_path)

        self.traces_mean = torch.from_numpy(
            np.array(self.traces_mean).astype(np.float32)
        )
        self.parameters_means = torch.from_numpy(
            np.array(self.parameters_means).astype(np.float32)
        )
        self.parameters_std_devs = torch.from_numpy(
            np.array(self.parameters_std_devs).astype(np.float32)
        )

        # torch data loader changes the data if there is an attribute named "load_all" set to true on this class.
        # obfuscating the name to __load_all fixes the problem?
        # can't reproduce
        self.__load_all = load_all
        if self.__load_all:
            # load into memory
            self.traces = np.array(self.traces).astype(np.float32)
            self.parameters = np.array(self.parameters).astype(np.float32)
            # convert to torch
            self.traces = torch.from_numpy(self.traces)
            self.parameters = torch.from_numpy(self.parameters)

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
        trace = self.traces[item, :, :].T
        parameters = self.parameters[item, :].reshape(-1)

        if not self.__load_all:
            trace = torch.from_numpy(trace.astype(np.float32))
            parameters = torch.from_numpy(parameters.astype(np.float32))

        trace[0] = trace[0] / self.traces_mean
        parameters = (parameters - self.parameters_means) / self.parameters_std_devs

        return trace, parameters

    @property
    def output_classes(self):
        return 7
