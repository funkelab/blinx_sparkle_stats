import logging
import os

from blinx import HyperParameters
from jax import numpy as jnp
from jax import random
from sparkle_stats.generate_dataset import generate_zarr_dataset
from sparkle_stats.sample_parameters import sample_p_off, sample_p_on, sample_parameters

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

data_dir = "/nrs/funke/projects/blinx/full_datasets/playground_val_dataset_v7"
assert not os.path.exists(data_dir), f"Can't overwrite {data_dir}"

y_list = range(5, 11)
traces_per_y = 10
num_frames = 4_000
hyper_parameters = HyperParameters()
traces_per_chunk = 60
seed = 4


def custom_sample(num_params, seed):
    original = sample_parameters(num_params, seed)
    keys = random.split(random.PRNGKey(seed + 1), 2)
    pon_again = sample_p_on(num_params, keys[0])
    poff_again = sample_p_off(num_params, keys[1])
    pon_original = original[:, -2].reshape(pon_again.shape)
    poff_original = original[:, -1].reshape(poff_again.shape)
    pon = jnp.min(jnp.hstack((pon_original, pon_again)), axis=1).reshape(
        pon_original.shape
    )
    poff = jnp.min(jnp.hstack((poff_original, poff_again)), axis=1).reshape(
        poff_original.shape
    )
    return jnp.hstack((original[:, :-2], pon, poff))


generate_zarr_dataset(
    data_dir,
    y_list,
    traces_per_y,
    num_frames,
    hyper_parameters,
    traces_per_chunk=traces_per_chunk,
    seed=seed,
    sample_func=custom_sample,
)
