import argparse

def str_to_bool(value):
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", type=int, default=5)
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("-non_linearity", type=str, default="linear")
    ap.add_argument("-plasticity_rule", type=str, default="oja")
    ap.add_argument("-meta_epochs", type=int, default=5)
    ap.add_argument("-num_trajec", type=int, default=10)
    ap.add_argument("-len_trajec", type=int, default=10)
    ap.add_argument("-type", type=str, default="weights")
    ap.add_argument(
        "-log_expdata", type=str_to_bool, nargs="?", const=True, default=False
    )
    ap.add_argument("-jobid", type=int, default=0)
    args = ap.parse_args()

    return (
        args.m, 
        args.n, 
        args.non_linearity,
        args.plasticity_rule,
        args.meta_epochs,
        args.num_trajec,
        args.len_trajec,
        args.type,
        args.log_expdata,
        args.jobid,
    )