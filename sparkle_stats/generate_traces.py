import time

import blinx
import blinx.trace_model
import jax
import jax.numpy as jnp
from jax import random

from sparkle_stats.parameters_util import parameters_array_to_object


def vmap_generate_traces(y, parameters, num_frames, hyper_parameters, seed=None):
    """Create several simulated intensity traces, each with different parameters.

    Args:
        y (int):
            - the total number of fluorescent emitters

        parameters (array): - n x PARAMETER_COUNT array of parameters with each row corresponding to values used by Parameters

        num_frames (int):
            - the number of observations to simulate

        hyper_parameters (:class:`HyperParameters`):
            - hyperparameters with `delta_t` set for the time between frames in the traces

        seed (int, optional):
            - random seed for the jax pseudo random number generator

    Returns:
        trace (array):
            - an n x num_frames array containing traces with intensity values for each frame

        states (array):
            - array the same shape as trace, containing the number of 'on' emitters in each frame
    """
    if seed is None:
        seed = time.time_ns()
    key = random.PRNGKey(seed)

    num_traces = parameters.shape[0]

    subkeys = random.split(key, num_traces)
    seeds = subkeys[:, 0]
    mapped = jax.vmap(
        _generate_trace_from_packed_params,
        in_axes=(None, 0, None, None, 0),
    )
    trace, zs = mapped(y, parameters, num_frames, hyper_parameters, seeds)
    return jnp.squeeze(trace), jnp.squeeze(zs)


def _generate_trace_from_packed_params(
    y, parameters, num_frames, hyper_parameters, seed=None
):
    """Generate a trace from parameters packed into an array.

    Args:
        parameters (array):
            1 x PARAMETER_COUNT array of parameters passed to the Parameters constructor.
    """
    parameters_obj = parameters_array_to_object(parameters)
    return blinx.trace_model.generate_trace(
        y, parameters_obj, num_frames, hyper_parameters, seed
    )
