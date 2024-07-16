import jax.numpy as jnp
from blinx import Parameters
from sparkle_stats.parameters_util import parameters_matrix_to_object


def test_parameters_matrix_to_object():
    parameters_matrix = jnp.arange(3 * 7).reshape(3, 7)

    parameters_object = parameters_matrix_to_object(parameters_matrix)

    assert isinstance(parameters_object, Parameters)
    assert parameters_object.r_e.shape == (3, 1)
    assert parameters_object.r_bg.shape == (3, 1)
    assert parameters_object.mu_ro.shape == (3, 1)
    assert parameters_object.sigma_ro.shape == (3, 1)
    assert parameters_object.gain.shape == (3, 1)
    assert parameters_object.p_on.shape == (3, 1)
    assert parameters_object.p_off.shape == (3, 1)
