import jax
import numpy as np
import torch
from sparkle_stats.training.loss import likelihood_loss


def test_likelihood_loss():
    labels = torch.arange(5 * 7).reshape(5, 7)
    predictions = torch.arange(5 * 7 * 2).reshape(5, 7 * 2)

    loss = likelihood_loss(predictions, labels)
    assert loss.shape == (1,)


def test_likelihood_loss_against_jax():
    labels = torch.tensor([[0.0]])
    predictions = torch.tensor([[0.0, 1.0]])

    loss = likelihood_loss(predictions, labels)
    assert loss.shape == (1,)

    jax_loss = jax.scipy.stats.norm.logpdf(0.0, 0.0, 1.0)

    loss = loss.detach().numpy()
    jax_loss = np.array(jax_loss)
    assert np.isclose(loss, jax_loss).all()
