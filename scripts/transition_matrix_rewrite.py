"""
Rewrite of the transition matrix to allow vmapping over multiple y values.

Key change is the convolutions of t_on and t_off in a way that allows for constant
array sizes. Instead of convolving the last and first rows, all possible
combinations of rows are convolved together and then masked away to select the right
ones. This increases the time complexity from O(n) to O(n^2). Not benchmarked, so
unknown if it offers any speedup.
"""
# TODO: maybe roll the t_on and t_off matricies so we can actually use the O(n) method?

# %%
from functools import partial

import blinx.trace_model as tm
import jax
import jax.numpy as jnp


# %%
# scipy.special.conb doesn't work with jax tracing
# see https://stackoverflow.com/a/74761360
def comb(n, k):
    return jnp.exp(
        jax.scipy.special.gammaln(n + 1)
        - jax.scipy.special.gammaln(k + 1)
        - jax.scipy.special.gammaln(n - k + 1)
    )


def create_comb_matrix(y, max_y):
    i_indices = jnp.arange(0, max_y + 1)
    j_indices = jnp.arange(0, max_y + 1)

    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def comb_i_j(i, j):
        return comb(i, j)

    arr = comb_i_j(i_indices, j_indices)
    return arr


def create_comb_matrix_slanted(y, max_y):
    i_indices = jnp.arange(0, max_y + 1)
    j_indices = jnp.arange(0, 2 * max_y + 1)

    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def comb_i_j(i, j):
        j = j - (max_y - i)
        return comb(i, j)

    arr = comb_i_j(i_indices, j_indices)
    return arr


def create_prob_matrix(y, p, max_y):
    i_indices = jnp.arange(0, max_y + 1)
    j_indices = jnp.arange(0, max_y + 1)

    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def prob_i_j(i, j):
        a = jnp.clip(j, a_min=0)
        b = jnp.clip(i - j, a_min=0)
        return p**a * (1.0 - p) ** b

    arr = prob_i_j(i_indices, j_indices)
    return arr


def create_prob_matrix_slanted(y, p, max_y):
    i_indices = jnp.arange(0, max_y + 1)
    j_indices = jnp.arange(0, 2 * max_y + 1)

    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def prob_i_j(i, j):
        j = j - (max_y - i)
        a = jnp.clip(j, a_min=0.0)
        b = jnp.clip(i - j, a_min=0.0)
        return p**a * (1.0 - p) ** b

    arr = prob_i_j(i_indices, j_indices)
    return arr


def correlate_matrix(t_on, t_off, y, max_y):
    i_indices = jnp.arange(0, max_y + 1)
    j_indices = jnp.arange(0, max_y + 1)

    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def correlate_i_j(i, j):
        return jnp.correlate(t_on[i, :], t_off[j, :], mode="valid")

    arr = correlate_i_j(i_indices, j_indices)

    # select the diagonal we want
    # the indexes conviently add up to y
    mask = (j_indices[None, :] + i_indices[:, None]) == y
    # anything that is selected will stay, anything outside will become 0
    arr = mask[:, :, None] * arr
    # each row and column only have one convolution, so sum either way to reduce to 2D
    # might have to change the roll depending on the axis
    arr = jnp.sum(arr, axis=0)
    # the columns we want are at the end, so bring them to the front
    arr = jnp.roll(arr, y + 1, axis=1)

    return arr


@partial(jax.vmap, in_axes=(0, 0, 0, None))
def create_transition_matrix(y, p_on, p_off, max_y):
    comb_matrix = create_comb_matrix(y, max_y)
    comb_matrix_slanted = create_comb_matrix_slanted(y, max_y)

    prob_matrix_on = create_prob_matrix_slanted(y, p_on, max_y)
    prob_matrix_off = create_prob_matrix(y, p_off, max_y)

    t_on_matrix = comb_matrix_slanted * prob_matrix_on
    t_off_matrix = comb_matrix * prob_matrix_off

    correlated = correlate_matrix(t_on_matrix, t_off_matrix, y, max_y)

    return correlated


# %%
y_list = [1, 2, 3, 4, 5]
p_on_list = [0.01, 0.4, 0.003, 0.04, 0.05]
p_off_list = [0.1, 0.02, 0.3, 0.002, 0.5]
max_y = 6

vm_correlateds = create_transition_matrix(
    jnp.array(y_list), jnp.array(p_on_list), jnp.array(p_off_list), max_y
)

for idx, (y, p_on, p_off) in enumerate(zip(y_list, p_on_list, p_off_list)):
    tm_correlated = tm.create_transition_matrix(y, p_on, p_off)
    vm_correlated = vm_correlateds[idx]
    assert jnp.isclose(
        vm_correlated[: y + 1, : y + 1],
        tm_correlated,
    ).all(), vm_correlated[: y + 1, : y + 1]
    assert jnp.isclose(
        vm_correlated.at[: y + 1, : y + 1].set(0.0),
        jnp.zeros((max_y + 1, max_y + 1)),
    ).all(), vm_correlated.at[: y + 1, : y + 1].set(0.0)
    print(f"correct for idx = {idx}")


# %%
