import os

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from sparkle_stats.sample_parameters import PARAMETER_COUNT


class ZarrDataset(Dataset):
    """A dataset stored on the filesystem as a trace zarr and a parameter zarr."""

    def __init__(
        self,
        data_dir,
        trace_mean=None,
        parameter_means=None,
        parameter_std_devs=None,
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

        self._load_parameter_std_devs(load_all, parameter_std_devs)
        self._load_parameter_means(load_all, parameter_means)
        self._load_trace_mean(load_all, trace_mean)

    # noinspection DuplicatedCode
    def _load_parameter_std_devs(self, load_all, parameter_std_devs):
        if parameter_std_devs is not None and parameter_std_devs.shape != (
            1,
            PARAMETER_COUNT,
        ):
            raise ValueError(
                f"Expected standard deviations to be of shape (1 x {PARAMETER_COUNT}), found {parameter_std_devs.shape}"
            )
        elif parameter_std_devs is not None:
            self.parameter_std_devs = parameter_std_devs
        elif parameter_std_devs is None and load_all:
            self.parameter_std_devs = torch.std(self.parameters, axis=0).reshape(1, -1)
        else:
            existing_parameter_std_devs_path = os.path.join(
                self.parameters_path, "std_devs"
            )
            if parameter_std_devs is None and os.path.exists(
                existing_parameter_std_devs_path
            ):
                self.parameter_std_devs = torch.load(
                    existing_parameter_std_devs_path, map_location=torch.device("cpu")
                )
                if self.parameter_std_devs.shape != (1, PARAMETER_COUNT):
                    raise ValueError(
                        f"Found existing std_devs at {existing_parameter_std_devs_path} with shape {self.parameter_std_devs.shape}, expected shape (1 x {PARAMETER_COUNT})"
                    )
            elif parameter_std_devs is None:
                self.parameter_std_devs = np.mean(self.parameters, axis=0).reshape(
                    1, -1
                )
                torch.save(self.parameter_std_devs, existing_parameter_std_devs_path)

    # noinspection DuplicatedCode
    def _load_parameter_means(self, load_all, parameter_means):
        if parameter_means is not None and parameter_means.shape != (
            1,
            PARAMETER_COUNT,
        ):
            raise ValueError(
                f"Expected means to be of shape (1 x {PARAMETER_COUNT}), found {parameter_means.shape}"
            )
        elif parameter_means is not None:
            self.parameter_means = parameter_means
        elif parameter_means is None and load_all:
            self.parameter_means = torch.mean(self.parameters, axis=0).reshape(1, -1)
        else:
            existing_parameter_means_path = os.path.join(self.parameters_path, "means")
            if parameter_means is None and os.path.exists(
                existing_parameter_means_path
            ):
                self.parameter_means = torch.load(
                    existing_parameter_means_path, map_location=torch.device("cpu")
                )
                if self.parameter_means.shape != (1, PARAMETER_COUNT):
                    raise ValueError(
                        f"Found existing means at {existing_parameter_means_path} with shape {self.parameter_means.shape}, expected shape (1 x {PARAMETER_COUNT})"
                    )
            elif parameter_means is None:
                self.parameter_means = np.mean(self.parameters, axis=0).reshape(1, -1)
                torch.save(self.parameter_means, existing_parameter_means_path)

    # noinspection DuplicatedCode
    def _load_trace_mean(self, load_all, trace_mean):
        if trace_mean is not None and trace_mean.shape != 1:
            raise ValueError(
                f"Expected means to be of shape (1), found {trace_mean.shape}"
            )
        elif trace_mean is not None:
            self.trace_mean = trace_mean
        elif trace_mean is None and load_all:
            self.trace_mean = torch.mean(self.traces[:, :, 0].reshape(1, -1))
        else:
            existing_trace_means_path = os.path.join(self.traces_path, "means")
            if trace_mean is None and os.path.exists(existing_trace_means_path):
                self.trace_mean = torch.load(
                    existing_trace_means_path, map_location=torch.device("cpu")
                )
                if self.trace_mean.shape != (1, PARAMETER_COUNT):
                    raise ValueError(
                        f"Found existing means at {existing_trace_means_path} with shape {self.trace_mean.shape}, expected shape (1)"
                    )
            elif trace_mean is None:
                self.trace_mean = np.mean(self.traces, axis=0).reshape(1, -1)
                torch.save(self.trace_mean, existing_trace_means_path)

    def __len__(self):
        return self.trace_count

    def __getitem__(self, item):
        # is currently NxTxC
        # reshape into CxT
        trace = self.traces[item, :, :].T
        parameters = self.parameters[item, :].reshape(-1)

        if not self.load_all:
            trace = torch.from_numpy(trace.astype(np.float32))
            parameters = torch.from_numpy(parameters.astype(np.float32))

        trace = trace / self.trace_mean
        parameters = (parameters - self.parameter_means) / self.parameter_std_devs

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
