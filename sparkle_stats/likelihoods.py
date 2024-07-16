import jax
import jax.numpy as jnp
from blinx.trace_model import get_trace_log_likelihood

from sparkle_stats.parameters_util import parameters_array_to_object


def vmap_get_trace_log_likelihoods(
    traces,
    y,
    parameters,
    hyper_parameters,
):
    """
    Get the log_likelihood of N traces and parameters

    Args:
        traces (array of shape (N, T):
            N sequences of T intensity observations

        y (int):
            the total number of fluorescent emitters

        parameters (array of shape (N, 7):
            N sets of parameters

        hyper_parameters (:class:`HyperParameters`, optional):
            The hyperparameters used for the maximum log_likelihood estimation

    Returns:
        log_likelihoods (array of shape (N,)):
            log likelihood for each of the N traces and parameters
    """

    mapped = jax.vmap(
        _get_trace_log_likelihood_from_packed_params,
        in_axes=(0, None, 0, None),
    )
    log_likelihoods = mapped(traces, y, parameters, hyper_parameters)

    return log_likelihoods


def _get_trace_log_likelihood_from_packed_params(
    trace,
    y,
    parameters,
    hyper_parameters,
):
    parameters_obj = parameters_array_to_object(parameters)
    return get_trace_log_likelihood(trace, y, parameters_obj, hyper_parameters)


def get_y_log_likelihoods(
    y_values,
    traces,
    parameters,
    hyper_parameters,
):
    """
    Get the log_likelihood of N traces and parameters for multiple y values

    Args:
        y_values (array of shape (Y,)):
            array of y values to try for each set of traces and parameters

        traces (array of shape (N, T):
            N sequences of T intensity observations

        parameters (array of shape (N, 7):
            N sets of parameters

        hyper_parameters (:class:`HyperParameters`, optional):
            The hyperparameters used for the maximum log_likelihood estimation

    Returns:
        y_log_likelihoods (array of shape (Y, N)):
            log likelihood of each Y for each of the N traces and parameters
    """

    y_log_likelihoods = []
    for y in y_values:
        log_likelihoods = vmap_get_trace_log_likelihoods(
            traces, y, parameters, hyper_parameters
        )
        y_log_likelihoods.append(log_likelihoods)
    y_log_likelihoods = jnp.stack(y_log_likelihoods)
    return y_log_likelihoods


def select_best_y_log_likelihoods(y_values, y_log_likelihoods):
    """
    Select the most likely y from multiple log likelihoods per y

    Args:
        y_values (array of shape (Y,)):
            array of y values for each set of traces and parameters

        y_log_likelihoods (array of shape (Y, N)):
            log likelihood of each y for each of the N traces and parameters

    Returns:
        max_y (array of shape (N,)):
            the y with the highest likelihood for each set of traces and parameters

        max_log_likelihoods (array of shape (N,)):
            the highest likelihood for each set of traces and parameters
    """

    max_log_likelihoods = jnp.max(y_log_likelihoods, axis=0)
    max_y = y_values[jnp.argmax(y_log_likelihoods, axis=0)]
    return max_y, max_log_likelihoods
