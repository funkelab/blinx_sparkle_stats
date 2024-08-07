import jax
from blinx import Parameters
from jax import numpy as jnp

from sparkle_stats.sample_parameters import PARAMETER_COUNT


def parameter_list_to_array(parameters):
    """Convert a list of parameters into an array that vmap_generate_traces accepts.

    Args:
        parameters (list):
            A list of Parameters objects

    Returns:
        parameters (array):
            Array of shape (len(parameters), PARAMETER_COUNT) that can be passed to vmap_generate_traces
    """
    return jnp.array(jax.tree.leaves(parameters)).reshape((-1, PARAMETER_COUNT))


def parameters_array_to_object(parameters):
    """Convert a flattened parameters array into a Parameters object.

    Args:
        parameters (array):
            1 x PARAMETER_COUNT array of parameters

    Returns:
        parameters (:class:`Parameters`)
    """
    r_e, r_bg, mu_ro, sigma_ro, gain, p_on, p_off = tuple(parameters)
    return Parameters(r_e, r_bg, mu_ro, sigma_ro, gain, p_on, p_off)


def parameters_matrix_to_object(parameters):
    """
    Convert a Nx7 parameters matrix into a Parameters object.

    Args:
        parameters (array):
            N x 7 array of parameters

    Returns:
        parameters (:class:`Parameters`)
    """
    # noinspection PyTypeChecker
    return Parameters(
        r_e=jnp.expand_dims(parameters[:, 0], axis=1),
        r_bg=jnp.expand_dims(parameters[:, 1], axis=1),
        mu_ro=jnp.expand_dims(parameters[:, 2], axis=1),
        sigma_ro=jnp.expand_dims(parameters[:, 3], axis=1),
        gain=jnp.expand_dims(parameters[:, 4], axis=1),
        p_on=jnp.expand_dims(parameters[:, 5], axis=1),
        p_off=jnp.expand_dims(parameters[:, 6], axis=1),
    )
