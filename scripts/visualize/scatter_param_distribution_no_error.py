# %%
import datetime
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import zarr
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
    "/nrs/funke/projects/blinx/images/",
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

parameters = {
    "all": ds.parameters.detach().numpy(),
}
predictions = {}

loader = DataLoader(ds, batch_size=100, shuffle=False)
outputs = []
for data, _ in loader:
    output = model(data)
    outputs.append(output)

outputs = torch.vstack(outputs).detach().numpy()

if len(outputs.shape) == 3:
    outputs = outputs.squeeze(1)

predictions["normalized"] = outputs
mins = np.load(os.path.join(normalization_ds_path, "parameters_min.npy"))
maxs = np.load(os.path.join(normalization_ds_path, "parameters_max.npy"))

mins = mins[:, parameter_indexes]
maxs = maxs[:, parameter_indexes]

parameters["normalized"] = (parameters["all"] - mins) / (maxs - mins) * 2 - 1

all_y = np.array(zarr.open(os.path.join(ds_path, "y"), "r"))

# %%
plt.style.use("dark_background")
default_colors = list(map(mcolors.to_hex, plt.get_cmap("Dark2").colors))
# default_colors = list(
#     map(mcolors.to_hex, plt.rcParams["axes.prop_cycle"].by_key()["color"])
# )
print(default_colors)

plt.rcParams["axes.labelsize"] = 17
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.titlepad"] = 20
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["savefig.format"] = "png"
plt.rcParams["savefig.dpi"] = 1200
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.25
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = "\n".join(
    [
        r"\usepackage{amsmath}",
    ]
)

width = 13 + 1 / 3
height = 7.5


def configure_plot(fig, ax):
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: r"\text{" + str(x) + "}")
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: r"\text{" + str(x) + "}")
    )

    width = 13 + 1 / 3
    height = 7.5
    fig.set_figheight(height)
    fig.set_figwidth(width)

    bbox = ax.get_position()
    fig.legend(
        loc="center",
        ncol=2,
        bbox_to_anchor=(bbox.x0 + 0.5 * bbox.width, bbox.y0 - 0.12),
        bbox_transform=fig.transFigure,
    )


bins = 4

names = [
    r"\textbf{Emitter Intensity (\boldmath{$r_{e}$})}",
    r"\textbf{Background Intensity (\boldmath{$r_{e}$})}",
    r"\boldmath{\mu_{\rho}}",
    r"\boldmath{\sigma_{\rho}}",
    r"\textbf{gain}",
    r"\textbf{Emitter On Probability (\boldmath{$p_\textbf{on}$})}",
    r"\textbf{Emitter Off Probability (\boldmath{$p_\textbf{off}$})}",
]

# %%
indexes = [0, 5, 6]


edges = np.linspace(all_y.min(), all_y.max(), bins + 1)
indices = np.digitize(all_y, edges)

fig, axs = plt.subplots(1, len(indexes), figsize=(5.5 * len(indexes), 5.5))
plt.subplots_adjust(wspace=0.2)
# change background to be default_colors[0]


for ax_idx, param_idx in enumerate(indexes):
    params = parameters["normalized"][:, param_idx]
    preds = predictions["normalized"][:, [param_idx, param_idx + 7]]
    preds, variances = preds[:, 0], preds[:, 1]
    binned_params = [params[indices == i] for i in range(1, bins + 1)]
    binned_preds = [preds[indices == i] for i in range(1, bins + 1)]
    binned_variances = [variances[indices == i] for i in range(1, bins + 1)]

    for bin_idx, (x, y, variance) in enumerate(
        zip(binned_params, binned_preds, binned_variances)
    ):
        ax = axs[ax_idx]
        ax.scatter(
            x=x,
            y=y,
            s=0.3,
            color=default_colors[0],
        )

        ax.plot(
            np.unique(x),
            np.poly1d(np.polyfit(x, x, 1))(np.unique(x)),
            color="lightgray",
        )

        ax.set_xlim([params.min(), params.max()])
        ax.set_ylim([preds.min(), preds.max()])
        if bin_idx == 0:
            ax.set_ylabel(
                r"\textbf{Predicted}",
                fontsize=15,
                labelpad=5,
                fontweight="bold",
            )
            ax.set_xlabel(
                r"\textbf{Actual}",
                fontsize=15,
                labelpad=5,
                fontweight="bold",
            )

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.tick_params(axis="both", length=6, width=1.5, labelsize=15)
        for spine in ax.spines.values():
            spine.set_linewidth(1)


for idx, param_idx in enumerate(indexes):
    axs[idx].set_title(
        names[param_idx],
        fontsize=20,
        fontweight="bold",
        pad=20,
    )


plt.savefig(
    os.path.join(
        model_ds_time_fig_directory,
        "scatter_param_distrubtion_no_error_by_re_pon_poff.png",
    ),
    bbox_inches="tight",
    pad_inches=0.25,
    format="png",
    dpi=600,
)

plt.show()

# %%
