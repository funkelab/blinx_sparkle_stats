import time
import jax.numpy as jnp

PARAMETER_COUNT = 7


def sample_parameters(
    num_params,
    hyper_parameters,

    seed=None,
):
    """Generate several sets of parameters

    Args:
        num_params (int):
            - the number of parameters to generate

        hyper_parameters (:class:`HyperParameters`):
            - HyperParameters used for generating each trace

        seed (int, optional):
            - random seed for the jax pseudo random number generator

    Returns:
        parameters (array):
            - a num_params x PARAMETER_COUNT array that can be passed to vmap_generate_traces
    """

    if seed is None:
        seed = time.time_ns()

    r_e = sample_r_e(num_params, hyper_parameters, seed)
    r_bg = sample_r_bg(num_params, hyper_parameters, seed)
    mu_ro = sample_mu_ro(num_params, hyper_parameters, seed)
    sigma_ro = sample_sigma_ro(num_params, hyper_parameters, seed)
    gain = sample_gain(num_params, hyper_parameters, seed)
    p_on = sample_p_on(num_params, hyper_parameters, seed)
    p_off = sample_p_off(num_params, hyper_parameters, seed)

    parameters = jnp.hstack((r_e, r_bg, mu_ro, sigma_ro, gain, p_on, p_off))
    parameters = parameters.reshape(-1, PARAMETER_COUNT)
    return parameters


# noinspection PyUnusedLocal
def sample_r_e(num_params, hyper_parameters, seed):
    return jnp.repeat(5, num_params).reshape(-1, 1)


# noinspection PyUnusedLocal
def sample_r_bg(num_params, hyper_parameters, seed):
    return jnp.repeat(5, num_params).reshape(-1, 1)


# noinspection PyUnusedLocal
def sample_mu_ro(num_params, hyper_parameters, seed):
    return jnp.repeat(2000, num_params).reshape(-1, 1)


# noinspection PyUnusedLocal
def sample_sigma_ro(num_params, hyper_parameters, seed):
    return jnp.repeat(700, num_params).reshape(-1, 1)


# noinspection PyUnusedLocal
def sample_gain(num_params, hyper_parameters, seed):
    return jnp.repeat(2, num_params).reshape(-1, 1)


# noinspection PyUnusedLocal
def sample_p_on(num_params, hyper_parameters, seed):
    return jnp.repeat(0.01, num_params).reshape(-1, 1)


# noinspection PyUnusedLocal
def sample_p_off(num_params, hyper_parameters, seed):
    return jnp.repeat(0.01, num_params).reshape(-1, 1)
