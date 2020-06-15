# coding=utf-8
import time
from pathlib import Path
from typing import List, Dict, Union, Tuple

import numpy as np
from matplotlib import rcParams

rcParams["font.family"] = "monospace"
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import argparse
# plt.xkcd()



## lib functions


def nice_string_output(
    names: List[str], values: List[str], extra_spacing: int = 0,
):
    max_values = len(max(values, key=len))
    max_names = len(max(names, key=len))
    string = ""
    for name, value in zip(names, values):
        string += "{0:s} {1:>{spacing}} \n".format(
            name,
            value,
            spacing=extra_spacing + max_values + max_names - len(name),
        )
    return string[:-2]



def run(args):

    dt = args["dt"]
    n_samples = args["samples"]
    tmax = args["tmax"]

    desc_str = nice_string_output(
        names=["dt","tmax"],
        values=[f"{val:.3f}" for val in (dt,)],
    )
    print(f"Simulating XXX  for {n_samples}, with\n")
    print(desc_str)



def parse_arguments() -> Dict[str, Union[int, float, str, Path]]:
    parser = argparse.ArgumentParser(description="Chapter X  exercise XX code")
    parser.add_argument("-N", "--samples", type=int, default=1024)

    parser.add_argument("-t","--tmax", type=float, default=20.)
    parser.add_argument("--dt", type=float, default=0.1)

    parser.add_argument("-o", "--outdir", type=str, default="./figs")
    parser.add_argument(
        "--filetype", type=str, choices=["png", "pdf"], default="pdf"
    )

    args = parser.parse_args()
    argdict = vars(args)  # returns a dict, easier to deal with
    po = Path(argdict["outdir"])
    if not po.exists():
        po.mkdir()
    print("Set output dir to: " + str(po.absolute()))
    argdict["outdir"] = po

    # Set output name here
    timestr = timestr = time.strftime("%Y%m%d_%H%M%S")
    out_str = f"ch6_e3a_" + timestr

    argdict["out_name"] = out_str

    return argdict




if __name__ == "__main__":
    args = parse_arguments()
    run(args)
