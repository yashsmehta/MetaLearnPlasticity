import argparse
import math

def str_to_bool(value):
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-input_dim", type=int, default=5)
    ap.add_argument("-output_dim", type=int, default=1)
    ap.add_argument("-hidden_layers", type=int, default=0)
    ap.add_argument("-hidden_neurons", type=int, default=-1)
    ap.add_argument(
        "-non_linear", type=str_to_bool, nargs="?", const=True, default=False
    )
    ap.add_argument("-plasticity_rule", type=str, default="oja")
    ap.add_argument("-meta_epochs", type=int, default=5)
    ap.add_argument("-num_trajec", type=int, default=10)
    ap.add_argument("-len_trajec", type=int, default=5)
    ap.add_argument("-type", type=str, default="weight")
    ap.add_argument("-num_meta_params", type=int, default=2)
    ap.add_argument(
        "-log_expdata", type=str_to_bool, nargs="?", const=True, default=False
    )
    ap.add_argument("-output_file", type=str, default="tester")
    ap.add_argument("-jobid", type=int, default=0)
    args = ap.parse_args()

    return (
        args.input_dim, 
        args.output_dim, 
        args.hidden_layers, 
        args.hidden_neurons, 
        args.non_linear,
        args.plasticity_rule,
        args.meta_epochs,
        args.num_trajec,
        args.len_trajec,
        args.type,
        args.num_meta_params,
        args.log_expdata,
        args.output_file,
        args.jobid,
    )

def A_index_to_powers(index):
    i = (index % (3 ** 2)) % 3
    j = math.floor((index % (3 ** 2)) / 3)
    k = math.floor(index / (3 ** 2))
    return i,j,k

def powers_to_A_index(i, j, k):
    index = (3 ** 0) * i + (3 ** 1) * j + (3 ** 2) * k
    return index