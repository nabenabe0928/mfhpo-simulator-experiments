from argparse import ArgumentParser

from benchmark_apis.hpo.lcbench import LCBenchSurrogate


parser = ArgumentParser()
parser.add_argument("--tmp_dir", type=str, default=None)
args = parser.parse_args()

LCBenchSurrogate.set_local_config(root_dir=args.tmp_dir)
