from sparkle_stats.sample_parameters import PARAMETER_COUNT, sample_parameters


def test_sample_parameters():
    num_params = 10
    seed = 1

    parameters = sample_parameters(num_params, seed)

    assert parameters.shape == (num_params, PARAMETER_COUNT)
