from sparkle_stats.sample_parameters import sample_parameters, PARAMETER_COUNT
from blinx import HyperParameters


def test_sample_parameters():
    num_params = 10
    hyper_parameters = HyperParameters()

    seed = 1

    parameters = sample_parameters(num_params, hyper_parameters, seed)

    assert parameters.shape == (num_params, PARAMETER_COUNT)
