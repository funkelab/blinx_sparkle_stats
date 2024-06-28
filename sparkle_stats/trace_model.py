import jax
from blinx.trace_model import get_trace_log_likelihood

__all__ = [
    "vmap_get_trace_log_likelihood",
]

from sparkle_stats.parameters_util import parameters_array_to_object


def vmap_get_trace_log_likelihood(traces, y, parameters, hyper_parameters):
    """
    Get the log_likelihood of a sets of parameters for multiple traces.

    Args:
        traces (tensor of shape n x t:

            several sequences of intensity observations

        y (int):

            the total number of fluorescent emitters to test for

        parameters (array):

            - n x PARAMETER_COUNT array of parameters with each row corresponding to the parameters to try

        hyper_parameters (:class:`HyperParameters`, optional):

            The hyper-parameters used for the maximum likelihood estimation.

    Returns:

        log_likelihood (array of shape (n)):

            log_likelihood of observing the traces given the parameters

    """
    mapped = jax.vmap(
        _get_trace_log_likelihood_from_packed_params,
        in_axes=(0, None, 0, None),
    )
    output = mapped(traces, y, parameters, hyper_parameters)
    return output.reshape(-1, 1)


def _get_trace_log_likelihood_from_packed_params(
    traces, y, parameters, hypter_parameters
):
    """Gets a trace's log likelihood from parameters packed into an array.

    Args:
        parameters (array):
            1 x PARAMETER_COUNT array of parameters passed to the Parameters constructor.
    """
    print(parameters.shape)
    parameters_obj = parameters_array_to_object(parameters)
    return get_trace_log_likelihood(
        traces,
        y,
        parameters_obj,
        hypter_parameters,
    )
