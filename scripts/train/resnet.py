import datetime
import os

import torch
import wandb
from sparkle_stats.datasets import ZarrIntensityOnlyDataset
from sparkle_stats.models import ResNet1D
from sparkle_stats.training.loaders import load_path
from sparkle_stats.training.loop import train_with_alpha
from sparkle_stats.training.loss import likelihood_loss

local = False

device = torch.device("cuda:0" if not local and torch.cuda.is_available() else "cpu")
print(f"running on {device}")

# %%
lr = 1e-5
epochs = 500
# add notes to differentiate different ResNet variations in W&B
notes = ""
model_type = "ResNet1D"
train_dataset = "/nrs/funke/projects/blinx/full_datasets/train_dataset_v7"
val_dataset = "/nrs/funke/projects/blinx/full_datasets/val_dataset_v7"
parameters = "all"
save_id = "__".join(
    (os.path.basename(__file__), datetime.datetime.now().strftime("%Y-%m-%d__%H:%M:%S"))
)
save_path = f"/groups/funke/home/{os.getlogin()}/checkpoints/blinx/{save_id}"

assert not os.path.exists(save_path), f"won't overwrite checkpoints {save_path}"
os.mkdir(save_path)

if not local:
    wandb.init(
        project="blinx_sparkle_stats",
        config={
            "learning rate": lr,
            "epochs": epochs,
            "model type": model_type,
            "train dataset": train_dataset,
            "validation dataset": val_dataset,
            "parameters": parameters,
            "save id": save_id,
            "save path": save_path,
            "notes": notes,
        },
        name=save_id,
    )

# %%
train_ds, train_loader = load_path(
    ZarrIntensityOnlyDataset,
    train_dataset,
    device,
    load_all=True,
    batch_size=100,
    normalize_parameters=True,
)

val_ds, val_loader = load_path(
    ZarrIntensityOnlyDataset,
    val_dataset,
    device,
    normalization_data_dir=train_dataset,
    load_all=True,
    batch_size=100,
    normalize_parameters=True,
)

assert (
    train_ds.trace_length == val_ds.trace_length
), "Expected train and val to have same trace length"

# %%
model = ResNet1D(output_classes=train_ds.output_classes * 2, start_channels=32)
# models saved with DataParallel need to jump through some extra hoops when loading and predicting
if torch.cuda.device_count() > 1:
    print(f"using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model.to(device)

# %%
loss_fn = likelihood_loss
# performing gradient ascent here so we want to maximize the likelihood
optimizer = torch.optim.Adam(model.parameters(), lr=lr, maximize=True)


def alpha_sample_func(epoch):
    """
    Args:
        epoch (int):
            - index of the current epoch
    Returns:
        alpha (float, 0 <= alpha <= 1):
            - alpha value to use in the loss function. Variance inputted for the loss is `alpha * variance + (1 - alpha) * 2`
    """
    # the ResNet with clipped gradients can learn the variance on its own without any handholding
    return 1


# %%

train_with_alpha(
    epochs=epochs,
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    save_path=save_path,
    alpha_sample_func=alpha_sample_func,
    # need to set this along size maximize=True for optimizer
    maximize=True,
    # ResNet with likelihood needs clipped gradients
    # todo: see if changing optimizer learning rate helps
    clip_gradient=True,
)
