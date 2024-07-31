import torch


def likelihood_loss(predictions, labels, alpha=1, epsilon=1e-2):
    # labels: B, 2*classes
    # predictions: B, classes
    assert (
        len(predictions.shape) == 2
    ), f"Expected predictions to be of shape (B, 2 * classes), found shape {predictions.shape}"
    assert (
        predictions.shape[1] % 2 == 0
    ), f"Expected predictions to be of shape (B, (2 * classes)), found shape {predictions.shape}"

    num_classes = predictions.shape[1] // 2
    assert (
        len(labels.shape) == 2
    ), f"Expected labels to be of shape (B, {num_classes}), found shape {labels.shape}"
    assert (
        labels.shape[1] == num_classes
    ), f"Expected labels to be of shape (B, {num_classes}), found shape {labels.shape}"

    # should turn
    #  [1 2 3 4 5 6]
    #
    # into
    # [[1, 2]
    #  [3, 4]
    #  [5, 6]]

    # predictions = predictions.reshape(-1, num_classes, 2)

    means = predictions[:, :num_classes]
    variances = predictions[:, num_classes:] + epsilon
    variances = alpha * variances + (1 - alpha) * 2

    dist = torch.distributions.Normal(means, variances)
    probs = dist.log_prob(labels)
    return probs.sum(axis=1).mean()
