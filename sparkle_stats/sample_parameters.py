import time

import jax.numpy as jnp
from jax import random

PARAMETER_COUNT = 7


def sample_parameters(
    num_params,
    seed=None,
):
    """Generate several sets of parameters

    Args:
        num_params (int):
            - the number of parameters to generate

        seed (int, optional):
            - random seed for the jax pseudo random number generator

    Returns:
        parameters (array):
            - a num_params x PARAMETER_COUNT array that can be passed to vmap_generate_traces
    """

    if seed is None:
        seed = time.time_ns()

    key = random.PRNGKey(seed)
    subkeys = random.split(key, 7)

    r_e = sample_r_e(num_params, subkeys[0])
    r_bg = sample_r_bg(num_params, subkeys[1])
    mu_ro = sample_mu_ro(num_params, subkeys[2])
    sigma_ro = sample_sigma_ro(num_params, subkeys[3])
    gain = sample_gain(num_params, subkeys[4])
    p_on = sample_p_on(num_params, subkeys[5])
    p_off = sample_p_off(num_params, subkeys[6])

    parameters = jnp.hstack((r_e, r_bg, mu_ro, sigma_ro, gain, p_on, p_off))
    parameters = parameters.reshape(-1, PARAMETER_COUNT)
    return parameters


def sample_r_e(num_params, key):
    return random.uniform(key, shape=(num_params, 1), minval=2, maxval=4)


def sample_r_bg(num_params, key):
    return random.uniform(key, shape=(num_params, 1), minval=2, maxval=8.5)


def sample_mu_ro(num_params, key):
    return random.normal(key, shape=(num_params, 1)) * 304 + 5000


def sample_sigma_ro(num_params, key):
    return random.normal(key, shape=(num_params, 1)) * 152 + 750


def sample_gain(num_params, key):
    return random.normal(key, shape=(num_params, 1)) * 0.122 + 2.2


def sample_p_on(num_params, key):
    return random.uniform(key, shape=(num_params, 1), minval=0.001, maxval=0.2)


def sample_p_off(num_params, key):
    return random.uniform(key, shape=(num_params, 1), minval=0.001, maxval=0.2)
