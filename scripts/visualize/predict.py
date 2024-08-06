# %%
import datetime
import os

import numpy as np
import torch
from get_args import get_args
from get_model import get_model
from sparkle_stats.datasets import ZarrIntensityOnlyDataset
from torch.utils.data import DataLoader

device = torch.device("cpu")


# %%
args = get_args()
print(args)

# %%
ds_path = args["ds_path"]
normalization_ds_path = args["normalization_ds_path"]
model_path = args["model_path"]
architecture = args["architecture"]
ds_name = args["ds_name"]

# %%
parameter_indexes = list(range(7))
print(parameter_indexes)

model_ds_time_fig_directory = os.path.join(
    "/nrs/funke/projects/blinx/predictions/",
    model_path.split("/")[-2],
    ds_name,
    datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S"),
).replace(":", "_")

os.makedirs(model_ds_time_fig_directory, exist_ok=True)

ds = ZarrIntensityOnlyDataset(
    ds_path,
    load_all=True,
    normalization_data_dir=normalization_ds_path,
    normalize_parameters=True,
)

ds.parameters = ds.parameters.to(device)
ds.traces = ds.traces.to(device)

model = get_model(architecture, ds, 14)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))


loader = DataLoader(ds, batch_size=100, shuffle=False)
outputs = []
for data, _ in loader:
    output = model(data)
    outputs.append(output)

outputs = torch.vstack(outputs).detach().numpy()

np.save(model_ds_time_fig_directory)
