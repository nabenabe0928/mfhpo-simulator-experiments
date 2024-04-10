# Reproducing Our Results

To reproduce our experiments, run the following on NEMO cluster:

```shell
# Install the dependencies.
$ pip install -r requirements.txt

# Run all the experiments.
$ ./scripts/submit.sh
```
> [!NOTE]
> The simulator module is available at [mfhpo-simulator](https://github.com/nabenabe0928/mfhpo-simulator).

> [!IMPORTANT]
> As SMAC3 is not compatible with JAHS-Bench-201 and LCBench, we need to prepare for another enviroment for SMAC3 by `$ pip install -r requirements-for-smac.txt`.

If you are not using the MOAB architecture, each experiment can be run by the following commands as well:

```shell
# We used seed from 0 to 29.
seed=0
# We used n_workers in {1,2,4,8}.
n_workers=4
# We have {lc,hpobench,hpolib,jahs,branin,hartmann}.
bench_cond="--bench_name hpolib --dataset_id 0"  # An example.

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
```
