import logging
import os

import jax.numpy as jnp
import numpy as np
import zarr

from sparkle_stats.generate_traces import vmap_generate_traces
from sparkle_stats.sample_parameters import PARAMETER_COUNT, sample_parameters

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def generate_zarr_dataset(
    data_dir,
    y_list,
    traces_per_y,
    num_frames,
    hyper_parameters,
    seed=None,
):
    """Generates a dataset stored as a zarr.

    Args:
        data_dir (string):
            - directory on the file system to save the zarr
        y_list (iterable):
            - all y's to create traces for
        traces_per_y (int):
            - number of traces to create for each y
        num_frames (int):
            - length of a single trace
        hyper_parameters (:class:`HyperParameters`):
            - hyper parameters used for generating traces
        seed (int, optional):
            - random seed for the jax psudo rendom number generator
    """

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Found file at {data_dir} expected directory")

    # the zarr is stored as an NxTx2 array, packing both the traces and states together
    traces_path = os.path.join(data_dir, "traces")
    parameters_path = os.path.join(data_dir, "parameters")
    zarr_traces = zarr.open(
        traces_path,
        mode="w",
        shape=(len(y_list) * traces_per_y, num_frames, 2),
    )
    zarr_parameters = zarr.open(
        parameters_path,
        mode="w",
        shape=(len(y_list) * traces_per_y, PARAMETER_COUNT),
    )

    for idx, y in enumerate(y_list):
        logger.debug(f"starting generating parameters for y={y}")
        parameters = sample_parameters(
            num_params=traces_per_y,
            seed=seed,
        )
        logger.debug(f"finished generating parameters for y={y}")

        # zarr eventually calls array.astype(..., order="K") on the jax array
        # jax arrays don't implement the order kwarg even though its numpy standard
        # have to convert the arrays into numpy arrays so it doesn't error out
        # relevant lines: zarr.core.Array: _process_for_setitem#2287
        parameters = np.array(parameters)
        zarr_parameters[idx * traces_per_y : (idx + 1) * traces_per_y, :] = parameters

        logger.debug(f"wrote parameters for y={y}")

        logger.debug(f"starting generating traces for y={y}")
        traces, states = vmap_generate_traces(
            y, parameters, num_frames, hyper_parameters, seed=None
        )
        logger.debug(f"finished generating traces for y={y}")

        traces = np.array(traces)
        states = np.array(states)

        zarr_traces[idx * traces_per_y : (idx + 1) * traces_per_y, :, 0] = traces
        logger.debug(f"wrote traces for y={y}")
        zarr_traces[idx * traces_per_y : (idx + 1) * traces_per_y, :, 1] = states
        logger.debug(f"wrote states for y={y}")

        logger.info(f"finished for y={y}")


def generate_memory_dataset(
    y_list,
    traces_per_y,
    num_frames,
    hyper_parameters,
    seed=None,
):
    """Generates a dataset in memory.

    Args:
        y_list (iterable):
            - all y's to create traces for
        traces_per_y (int):
            - number of traces to create for each y
        num_frames (int):
            - length of a single trace
        hyper_parameters (:class:`HyperParameters`):
            - hyper parameters used for generating traces
        seed (int, optional):
            - random seed for the jax psudo rendom number generator

    Returns:
        trace (array):
            - an array containing traces with intensity values for each frame

        states (array):
            - array the same shape as trace, containing the number of 'on' emitters in each frame
    """
    all_traces = []
    all_states = []
    all_parameters = []
    for y in y_list:
        parameters = sample_parameters(
            num_params=traces_per_y,
            seed=seed,
        )
        all_parameters.append(parameters)

        y_traces, y_states = vmap_generate_traces(
            y, parameters, num_frames, hyper_parameters, seed=None
        )
        all_traces.append(y_traces)
        all_states.append(y_states)

    all_traces = jnp.vstack(all_traces)
    all_states = jnp.vstack(all_states)
    all_parameters = jnp.vstack(all_parameters)

    traces_and_states = jnp.concat(
        (jnp.expand_dims(all_traces, axis=2), jnp.expand_dims(all_states, axis=2)),
        axis=2,
    )

    return traces_and_states, all_parameters
