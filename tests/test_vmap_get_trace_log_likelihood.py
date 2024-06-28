from blinx import HyperParameters
from sparkle_stats.generate_dataset import generate_memory_dataset
from sparkle_stats.trace_model import vmap_get_trace_log_likelihood


def test_vmap_get_trace_log_likelihood():
    y_list = [6, 7]
    traces_per_y = 10
    num_frames = 10
    hyper_parameters = HyperParameters(
        max_x=10,
        num_x_bins=10,
        num_outliers=2,
    )
    seed = 1

    traces, parameters, _ = generate_memory_dataset(
        y_list, traces_per_y, num_frames, hyper_parameters, seed
    )

    output = vmap_get_trace_log_likelihood(traces, 6, parameters, hyper_parameters)
    assert output.shape == (len(y_list) * traces_per_y, 1)
