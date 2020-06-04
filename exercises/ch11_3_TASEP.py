# coding=utf-8
import json
import time
from pathlib import Path
from typing import List, Dict, Union, Tuple

import pandas as pd
import numpy as np
from matplotlib import rcParams

rcParams["font.family"] = "monospace"
from matplotlib import pyplot as plt
import argparse
import scipy.stats


def main(args: Dict):
    print("Running with values: ")

    print(
        json.dumps(
            {k: args[k] for k in set(list(args.keys())) - set(["outdir"])}, indent=4,
        )
    )

    const_L = args["length"]
    const_dT = args["timestep"]
    const_Tmax = args["maxtime"]

    const_nSteps = int(const_Tmax // const_dT)
    const_stepProb = const_dT * args["stepprob"]

    # Initialize the array for storing all positions over time

    position_time_array = np.zeros(
        shape=(const_nSteps, const_L)
    )  # now we access it as [time, position]

    # Populate the array at time 0
    position_time_array[0] = np.random.binomial(n=1, p=args["density"], size=const_L)

    for i in range(1, const_nSteps):
        N_curr = np.copy(position_time_array[i - 1])
        move_inds = np.random.choice(np.arange(const_L), size=const_L, replace=True)

        for j in move_inds:

            if (
                N_curr[j] == 1
                and N_curr[(j + 1) % (const_L - 1)] == 0
                and np.random.uniform() > const_stepProb
            ):
                N_curr[j] = 0
                N_curr[(j + 1) % (const_L - 1)] = 1

        position_time_array[i] = N_curr

    # print(position_time_array)

    (timepoints, positions) = np.nonzero(position_time_array)

    df = pd.DataFrame({"Time": timepoints, "Pos": positions})

    ax: plt.Axes
    fig: plt.Figure
    fig, ax = plt.subplots()
    ax.set_title(
        rf"Simulation with $L={const_L}, p\Delta t = {const_stepProb}, \rho={args['density']:.2f}$"
    )

    ax.scatter(positions, timepoints, marker=">", c="k", s=4)

    if const_dT != 1:
        ax.set_yticklabels([f"{v* const_dT:.2f}s" for v in ax.get_yticks()])

    ax.set_xlabel("Position")
    ax.set_ylabel("Time")

    fig.tight_layout()

    basename = args["out_name"] + "_TASEP." + args["filetype"]
    fig.savefig(args["outdir"] / basename)
    fig.clf()


def parse_arguments() -> Dict[str, Union[int, float, str, Path]]:
    parser = argparse.ArgumentParser(
        description="Chapter 11 exercise 3 code - Simulating TASEP"
    )

    parser.add_argument("-l", "--length", type=int, default=100)
    parser.add_argument("-dt", "--timestep", type=float, default=1.0)
    parser.add_argument("-p", "--stepprob", type=float, default=0.5)
    parser.add_argument("-t", "--maxtime", type=int, default=100)
    parser.add_argument("-f", "--density", type=float, default=0.1)

    parser.add_argument("-o", "--outdir", type=str, default="./figs")
    parser.add_argument("--filetype", type=str, choices=["png", "pdf"], default="pdf")

    parser.add_argument("--TEST", type=bool, default=False)  # change to false once done

    args = parser.parse_args()
    argdict = vars(args)  # returns a dict, easier to deal with

    if argdict["TEST"] == True:
        argdict["length"] = 15
        argdict["maxtime"] = 50
        argdict["timestep"] = 1

    po = Path(argdict["outdir"])
    if not po.exists():
        po.mkdir()
    print("Set output dir to: " + str(po.absolute()))
    argdict["outdir"] = po
    # Set output name here
    timestr = timestr = time.strftime("%Y%m%d_%H%M%S")
    out_str = f"ch11_e3_" + timestr

    argdict["out_name"] = out_str

    return argdict


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
