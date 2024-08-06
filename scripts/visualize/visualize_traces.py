# %%
import datetime
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import zarr

# %%
model_ds_time_fig_directory = os.path.join(
    "/nrs/funke/projects/blinx/images/parameters",
    datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S"),
).replace(":", "_")

os.makedirs(model_ds_time_fig_directory, exist_ok=True)

# %%
z = zarr.open("/nrs/funke/projects/blinx/full_datasets/val_dataset_v7/traces", "r")
y = zarr.open("/nrs/funke/projects/blinx/full_datasets/val_dataset_v7/y", "r")
# select 6 random items from z
# so like [i, :, :]
selected_idx = np.random.random_integers(low=0, high=z.shape[0] - 1, size=6)
selected_items = [z[i, :, 0] for i in selected_idx]
selected_y = [y[i] for i in selected_idx]

# %%
plt.style.use("dark_background")
# default_colors = list(map(mcolors.to_hex, plt.get_cmap("Set1").colors))
default_colors = list(
    map(mcolors.to_hex, plt.rcParams["axes.prop_cycle"].by_key()["color"])
)
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


# %%
fig, axs = plt.subplots(3, 2, figsize=(width, height))
fig.text(
    0.5,
    0.05,
    r"\textbf{Time}",
    ha="center",
    va="center",
    fontsize=15,
)

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.2, wspace=0.05)

for i, (item, y) in enumerate(zip(selected_items, selected_y)):
    col = i % 2
    row = i // 2
    ax = axs[row, col]
    # setting text for fig creates massive margin
    if col == 0 and row == 1:
        ax.set_ylabel(r"\textbf{Intensity}", fontsize=15)
    # text box in top right corner
    ax.text(
        1,
        1,
        r"\boldmath{$n = " + str(int(y)) + r"$}",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="black", edgecolor="white", boxstyle="square"),
        fontsize=10,
    )
    # ax.set_title(r"\boldmath{$n = "+str(int(y))+r"$}", fontsize=15, pad=10)
    ax.plot(item, color=default_colors[5])
    ax.set_xlim(0, 250)
    ax.set_ylim(item.min(), item.max())
    ax.set_xticklabels([])
    ax.set_yticklabels([])


plt.savefig(
    os.path.join(model_ds_time_fig_directory, "visualize_traces.png"),
    bbox_inches="tight",
    pad_inches=0.25,
    format="png",
    dpi=600,
)
plt.show()

# %%
