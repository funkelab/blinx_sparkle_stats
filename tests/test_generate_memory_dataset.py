from blinx import HyperParameters
from sparkle_stats.generate_dataset import generate_memory_dataset
from sparkle_stats.sample_parameters import PARAMETER_COUNT


def test_generate_memory_dataset():
    y_list = [6, 7]
    traces_per_y = 10
    num_frames = 10
    hyper_parameters = HyperParameters()
    seed = 1

    traces, parameters, all_ys = generate_memory_dataset(
        y_list, traces_per_y, num_frames, hyper_parameters, seed
    )

    assert traces.shape == (len(y_list) * traces_per_y, num_frames, 2)
    assert parameters.shape == (len(y_list) * traces_per_y, PARAMETER_COUNT)
    assert all_ys.shape == (len(y_list) * traces_per_y)
