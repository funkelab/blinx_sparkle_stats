# Visualize Models

- Generate the visualizations for the models
- Most visualizations require a full LaTeX install to be on PATH
- Most visualizations also include an `indexes` list, containing the indexes of the parameters to visualize
  - Initialized to the indicies for the default parameters, but can be changed to select different ones
  - See table below for the index values

| Index | Parameter      |
| ----- | -------------- |
| 0     | $r_e$          |
| 1     | $r_\text{bg}$  |
| 2     | $\mu_\rho$     |
| 3     | $\sigma_\rho$  |
| 4     | $\text{gain}$  |
| 5     | $p_\text{on}$  |
| 6     | $p_\text{off}$ |

## `get_x`

- These files are helper functions to get the proper paths for datasets and the model. They also enable using the actual visualization scripts as python CLIs (`python script.py -h`).
- These rely on a lot of the conventions used when creating/saving datasets with the other scripts, and can break if they aren't followed
- If any of the helper modules break, it's probably easier to just replace their calls in the visualization scripts with hardcoded paths.
- Most visualizations enter this by calling `get_args`, returning a dictionary of paths
  - See table below for the dict keys


| `args` key              | Description                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| `ds_path`               | Full path to the dataset to predict on                                                   |
| `normalization_ds_path` | Full path to the dataset containing normalization factors                                |
| `model_path`            | Full path to the model checkpoints                                                       |
| `architecture`          | ALL CAPS shorthand for the model architecture (see `get_architecture` for possibilities) |
| `ds_name`               | Not used; name of the dataset                                                            |

## `scatter_param_distrubtion_no_error`

- Creates scatter plots (Predicted vs Actual value) for each parameter without error bars representing the variance

## `scatter_param_distribution_no_error_by_n`

- Creates scatter plots (Predicted vs Actual value) for each parameter without error bars representing the variance, and binned trace emitter count

## `visualize_traces`

- Creates several example trace plots
- Is not a python CLI and does not need a model
- The dataset path is hardcoded into the script