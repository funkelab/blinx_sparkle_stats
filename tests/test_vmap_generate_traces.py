from blinx import HyperParameters
from blinx.parameters import Parameters
from sparkle_stats.generate_traces import parameter_list_to_array, vmap_generate_traces


def test_vmap_generate_traces_with_different_params():
    num_frames = 10
    hyper_parameters = HyperParameters()
    seed = 1

    parameters = [
        Parameters(
            r_e=5,
            r_bg=5,
            mu_ro=2000,
            sigma_ro=700,
            gain=2,
            p_on=0.01,
            p_off=0.01,
        ),
        Parameters(
            r_e=4,
            r_bg=4,
            mu_ro=3000,
            sigma_ro=600,
            gain=2.5,
            p_on=0.03,
            p_off=0.02,
        ),
        Parameters(
            r_e=5,
            r_bg=4,
            mu_ro=2500,
            sigma_ro=750,
            gain=2,
            p_on=0.05,
            p_off=0.01,
        ),
    ]

    parameters_arr = parameter_list_to_array(parameters)

    sim_trace, sim_zs = vmap_generate_traces(
        y=4,
        parameters=parameters_arr,
        num_frames=num_frames,
        hyper_parameters=hyper_parameters,
        seed=seed,
    )

    assert sim_trace.shape == (len(parameters), num_frames)
    assert sim_zs.shape == (len(parameters), num_frames)
