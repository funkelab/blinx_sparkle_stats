# Train Models

- Training and data loading utilities are stored in `sparkle_stats.train`
- The scripts all log to Weights & Biases, so you will need to set that up before running
    - The training functions in `sparkle_stats.train` log to W&B and will error if you don't set them up


## ResNet

- This version of ResNet predicts both a mean and variance and uses maximum likelihood as a loss
	- In order to predict two variables, output classes are set to 14. The model doesn't care what each "class" represents. The loss function is what creates that distinction.
    - Reshaping into `(-1, 2, 7)` will use the second dimension to group means (`[:, 0, :]`) and variances (`[:, 1, :]`)
<!-- TODO: fix output shape -->
| Index   | Parameter               |
| ------- | ----------------------- |
| [0, 0]  | $r_e$ mean              |
| [0, 1]  | $r_e$ variance          |
| [0, 2]  | $r_\text{bg}$ mean      |
| [0, 3]  | $r_\text{bg}$ variance  |
| [0, 4]  | $\mu_\rho$ mean         |
| [0, 5]  | $\mu_\rho$ variance     |
| [0, 6]  | $\sigma_\rho$ mean      |
| [0, 7]  | $\sigma_\rho$ variance  |
| [0, 8]  | $\text{gain}$ mean      |
| [0, 9]  | $\text{gain}$ variance  |
| [0, 10] | $p_\text{on}$ mean      |
| [0, 11] | $p_\text{on}$ variance  |
| [0, 12] | $p_\text{off}$ mean     |
| [0, 13] | $p_\text{off}$ variance |

