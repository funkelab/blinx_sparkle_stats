import jax.numpy as jnp
from blinx import HyperParameters
from sparkle_stats.generate_dataset import generate_memory_dataset
from sparkle_stats.likelihoods import (
    get_y_log_likelihoods,
    vmap_get_trace_log_likelihoods,
)


def test_vmap_get_trace_log_likelihoods():
    y_list = [6, 7]
    traces_per_y = 10
    num_frames = 4000
    hyper_parameters = HyperParameters()
    seed = 1

    traces, parameters, all_ys = generate_memory_dataset(
        y_list, traces_per_y, num_frames, hyper_parameters, seed=seed
    )
    traces = traces[:, :, 0]
    hyper_parameters.max_x = traces.max()

    log_likelihoods = vmap_get_trace_log_likelihoods(
        traces,
        6,
        parameters,
        hyper_parameters,
    )

    assert log_likelihoods.shape == (traces.shape[0],)
    assert jnp.isfinite(log_likelihoods).all()


def test_get_y_log_likelihoods():
    y_list = [6, 7]
    traces_per_y = 10
    num_frames = 4000
    hyper_parameters = HyperParameters()
    seed = 1

    traces, parameters, all_ys = generate_memory_dataset(
        y_list, traces_per_y, num_frames, hyper_parameters, seed=seed
    )
    traces = traces[:, :, 0]
    hyper_parameters.max_x = traces.max()

    y_values = jnp.array([5, 6, 7, 8]).reshape(-1)

    y_log_likelihoods = get_y_log_likelihoods(
        y_values,
        traces,
        parameters,
        hyper_parameters,
    )

    assert y_log_likelihoods.shape == (y_values.shape[0], traces.shape[0])
    assert jnp.isfinite(y_log_likelihoods).all()
