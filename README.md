# Reproducing Our Results

## Install Dependencies

To reproduce our experiments, run the following on NEMO cluster:

```shell
# Install the dependencies.
$ pip install -r requirements.txt
```
> [!TIP]
> If `singularity` is available, you can install the dependencies via `sudo singularity build mfhpo-simulator.sif mfhpo-simulator.def` as well.

> [!NOTE]
> The simulator module is available at [mfhpo-simulator](https://github.com/nabenabe0928/mfhpo-simulator).

> [!IMPORTANT]
> As SMAC3 is not compatible with JAHS-Bench-201 and LCBench, we need to prepare for another enviroment for SMAC3 by `$ pip install -r requirements-for-smac.txt`.

## Run Our Experiments

If you are using a cluster with the MOAB architecture, all the experiments are simply run by:

```shell
$ ./scripts/submit.sh
```

> [!NOTE]
> All the results will be stored in `mfhpo-simulator-info/` directory created by our wrapper.

If you are not using the MOAB architecture, each experiment can be run by the following commands as well although you need to iterate over all the conditions:

```shell
# We used seed from 0 to 29.
seed=0
# We used n_workers in {1,2,4,8}.
n_workers=4
# We have {lc,hpobench,hpolib,jahs,branin,hartmann}.
bench_cond="--bench_name hpolib --dataset_id 0"  # An example.
# tmp_dir is typically necessary for cluster computation to reduce the filesystem burden.
# If you are using your local computer, this variable can be simply omitted.
tmp_dir=$TMPDIR

# Benchmark conditions.
# bench_cond="--bench_name lc --dataset_id 0"  # dataset_id must be in {0, 1, ..., 33}.
# bench_cond="--bench_name hpobench --dataset_id 0"  # dataset_id must be in {0, 1, ..., 7}.
# bench_cond="--bench_name hpolib --dataset_id 0"  # dataset_id must be in {0, 1, 2, 3}.
# bench_cond="--bench_name jahs --dataset_id 0"  # dataset_id must be in {0, 1, 2}.
# bench_cond="--bench_name branin"  # Branin does not have any additional args.
# bench_cond="--bench_name hartmann --dim 3"  # Hartmann has an arg of `--dim` and `--dim` must be either 3 or 6.

# An Experiment for BOHB.
$ python -m src.bohb --n_workers $n_workers --seed $seed --tmp_dir $tmp_dir $bench_cond

# An Experiment for DEHB.
$ python -m src.dehb --n_workers $n_workers --seed $seed --tmp_dir $tmp_dir $bench_cond

# An Experiment for SMAC3.
# SMAC3 requires another enviroment.
$ python -m src.smac --n_workers $n_workers --seed $seed --tmp_dir $tmp_dir $bench_cond

# An Experiment for Random Search.
$ python -m src.random --n_workers $n_workers --seed $seed --tmp_dir $tmp_dir $bench_cond

# An Experiment for TPE.
$ python -m src.tpe --n_workers $n_workers --seed $seed --tmp_dir $tmp_dir $bench_cond

# An Experiment for HyperBand.
$ python -m src.hyperband --n_workers $n_workers --seed $seed --tmp_dir $tmp_dir $bench_cond

# An Experiment for NePS.
$ ./src/neps.sh --n_workers $n_workers --seed $seed --tmp_dir $tmp_dir $bench_cond

# Small Experiments.
# The Experiments for Figures 3 and 4.
$ python -m validation.test_order_match

# The Experiments for Figure 5.
$ python -m validation.test_line_match

# Experiments not in the paper.
$ python -m validation.test_cumtime_match
```

## Visualize Our Results

### Figures and Tables in the Paper

These visualizations require all the experiments by each optimizer.
- [Table of Overall Runtime Reduction (Table 1)](validation/real_world_runtime_reduction.py)
- [Critical Difference Diagrams (Figure 6)](validation/rank_test.py)

```shell
# Table of Overall Runtime Reduction (Table 1).
$ python -m validation.real_world_runtime_reduction

# Critical Difference Diagrams (Figure 6).
$ python -m validation.rank_test
```

These visualizations require small experiments mentioned above.
- [Visualization of the Order Match (Figure 3)](validation/viz_order_match.ipynb)
- [Trajectory Match of the Cumulative Runtime (Figure 4)](validation/viz_cumtime_match_large.ipynb)
- [Trajectory Match of Performance over Time (Figure 5)](validation/viz_runtime_reduction.py)

```shell
# Trajectory Match of Performance over Time (Figure 5).
$ python -m validation.viz_runtime_reduction
```

### The Figures and Tables Not in the Paper.

These visualizations require all the experiments by each optimizer.
- [Visualization of Average Rank over Cumulative Runtime](validation/viz_avg_rank.py)
- [Table of Runtime Reduction for Each Benchmark](validation/create_latex_code_for_perf_over_time.py)

```shell
# Visualization of Average Rank over Cumulative Runtime.
$ python -m validation.viz_avg_rank

# Table of Runtime Reduction for Each Benchmark.
$ python -m validation.create_latex_code_for_perf_over_time
```

These visualizations require small experiments mentioned above.
- [Visualization of the Cumulative Runtime Match with Very Small Examples](validation/viz_cumtime_match.ipynb)
- [Visualization of Performance Over Runtime for Each Benchmark](validation/viz_perf_over_time.py)
- [Visualization of the Dataset Size Trajectory Match](validation/viz_size_traj_match.ipynb)

```shell
# Visualization of Performance Over Runtime for Each Benchmark.
$ python -m validation.viz_perf_over_time
```
