# %%
import datetime
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from blinx import HyperParameters, Parameters
from blinx.trace_model import generate_trace

device = torch.device("cpu")

# %%
time_fig_directory = os.path.join(
    "/nrs/funke/projects/blinx/images/",
    "parameters",
    datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S"),
).replace(":", "_")
os.makedirs(time_fig_directory, exist_ok=True)

# %%
plt.rcParams["axes.labelsize"] = 17
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.titlepad"] = 20
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["legend.fontsize"] = 18
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
hyper_parameters = HyperParameters()
parameters = {}
traces = {}

# %%
plt.style.use("dark_background")
default_colors = list(map(mcolors.to_hex, plt.get_cmap("Dark2").colors))
# default_colors = list(
#     map(mcolors.to_hex, plt.rcParams["axes.prop_cycle"].by_key()["color"])
# )
print(default_colors)

# %%
parameters["low_r_e"] = Parameters(
    r_e=2,
    r_bg=6,
    mu_ro=2000,
    sigma_ro=750,
    gain=2.2,
    p_on=0.1,
    p_off=0.1,
)
parameters["high_r_e"] = Parameters(
    r_e=4,
    r_bg=6,
    mu_ro=2000,
    sigma_ro=750,
    gain=2.2,
    p_on=0.1,
    p_off=0.1,
)

num_frames = 250

traces["low_r_e"] = np.array(
    generate_trace(1, parameters["low_r_e"], num_frames, hyper_parameters, seed=0)[0][0]
)
traces["high_r_e"] = np.array(
    generate_trace(1, parameters["high_r_e"], num_frames, hyper_parameters, seed=1)[0][
        0
    ]
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(width, height))

ax.plot(traces["high_r_e"], label=r"\boldmath{$r_{e} = 4$}", color=default_colors[0])
ax.plot(traces["low_r_e"], label=r"\boldmath{$r_{e} = 2$}", color=default_colors[1])

ax.set_xlabel(r"\textbf{Time}", labelpad=7)
ax.set_ylabel(r"\textbf{Intensity}", labelpad=12)

ax.set_xlim([0, num_frames])

ax.set_title(r"\textbf{Emitter Intensity (\boldmath{$r_{e}$})}")

configure_plot(fig, ax)

plt.savefig(
    os.path.join(time_fig_directory, "low_high_r_e.png"),
)
# plt.show()

# %%
parameters["low_r_bg"] = Parameters(
    r_e=3,
    r_bg=5,
    mu_ro=2000,
    sigma_ro=750,
    gain=2.2,
    p_on=0.1,
    p_off=0.1,
)
parameters["high_r_bg"] = Parameters(
    r_e=3,
    r_bg=7,
    mu_ro=2000,
    sigma_ro=750,
    gain=2.2,
    p_on=0.1,
    p_off=0.1,
)

num_frames = 250

traces["low_r_bg"] = np.array(
    generate_trace(2, parameters["low_r_bg"], num_frames, hyper_parameters, seed=1)[0][
        0
    ]
)
traces["high_r_bg"] = np.array(
    generate_trace(2, parameters["high_r_bg"], num_frames, hyper_parameters, seed=1)[0][
        0
    ]
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(width, height))

ax.plot(traces["high_r_bg"], label=r"\boldmath{$r_{bg} = 7$", color=default_colors[0])
ax.plot(traces["low_r_bg"], label=r"\boldmath{$r_{bg} = 5$", color=default_colors[1])

ax.set_xlabel(r"\textbf{Time}", labelpad=7)
ax.set_ylabel(r"\textbf{Intensity}", labelpad=12)

ax.set_xlim([0, num_frames])
ax.set_title(r"\textbf{Background Intensity (\boldmath{$r_{bg}$})}")

configure_plot(fig, ax)

plt.savefig(
    os.path.join(time_fig_directory, "low_high_r_bg.png"),
)
# plt.show()

# %%
parameters["low_p_on"] = Parameters(
    r_e=3,
    r_bg=6,
    mu_ro=2000,
    sigma_ro=750,
    gain=2.2,
    p_on=0.01,
    p_off=0.1,
)
parameters["low_p_off"] = Parameters(
    r_e=3,
    r_bg=6,
    mu_ro=2000,
    sigma_ro=750,
    gain=2.2,
    p_on=0.1,
    p_off=0.01,
)

num_frames = 250

traces["low_p_on"] = np.array(
    generate_trace(1, parameters["low_p_on"], num_frames, hyper_parameters, seed=7)[0][
        0
    ]
)
traces["low_p_off"] = np.array(
    generate_trace(1, parameters["low_p_off"], num_frames, hyper_parameters, seed=4)[0][
        0
    ]
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(width, height))

ax.plot(
    traces["low_p_off"],
    label=r"\boldmath{$p_\textbf{on} = 0.1$}, \boldmath{$p_\textbf{off} = 0.01$}",
    color=default_colors[0],
)
ax.plot(
    traces["low_p_on"],
    label=r"\boldmath{$p_\textbf{on} = 0.01$}, \boldmath{$p_\textbf{off} = 0.1$}",
    color=default_colors[1],
)

ax.set_xlabel(r"\textbf{Time}", labelpad=7)
ax.set_ylabel(r"\textbf{Intensity}", labelpad=12)

ax.set_xlim([0, num_frames])

ax.set_title(
    r"\textbf{Emitter On and Off Probabilities (\boldmath{$p_\textbf{on}$} \textbf{and} \boldmath{$p_\textbf{off}$})}",
)

configure_plot(fig, ax)

plt.savefig(
    os.path.join(time_fig_directory, "low_high_p_on_p_off.png"),
)
# plt.show()

# %%
