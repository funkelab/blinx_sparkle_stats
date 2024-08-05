# Create Datasets

- All datasets have chunk many traces together to avoid excessive I/O
    - Better to load large chunks than to waste a lot of time reading from disk when trying to load thousands of traces
- Traces are stored in shape `(4_000, 2)`
    - Second dimension is "channel". 0th index is the actual intensity values and 1st is the z-state
    - Possible to train models on the z-states only (models would predict just $p_\text{on}$ and $p_\text{off}$)
- This iteration of data generation favors lower $p_\text{on}$ and $p_\text{off}$ by selecting the min of 2 uniform distributions

## `train`

- 1,000,000 traces
- 1 <= N <= 40
- 4,000 frames/trace
- ~25 GB
- `(10_000, 4_000, 2)` chunk size
- Used to train large models

## `val`

- 40,000 traces
- 1 <= N <= 40
- 4,000 frames/trace
- ~1 GB
- `(4_000, 4_000, 2)` chunk size
- Used to train large models
- Also used to run visualizations

## `playground_train`

- 600 traces
- 5 <= N <= 10
- 100 traces/N
- 4,000 frames/trace
- ~15 MB
- `(600, 4_000, 2)` chunk size
- Used to train small models and ensure pipelines work
- Also used to test visualizations locally

## `playground_val`

- 60 traces
- 5 <= N <= 10
- 10 traces/N
- 4,000 frames/trace
- ~1.5 MB
- `(60, 4_000, 2)` chunk size
- Used to train small models and ensure pipeliens work