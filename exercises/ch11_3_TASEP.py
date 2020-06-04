# coding=utf-8
import json
import time
from pathlib import Path
from typing import List, Dict, Union, Tuple

import pandas as pd
import numpy as np
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

rcParams["font.family"] = "monospace"
from matplotlib import pyplot as plt
import argparse
import scipy.stats


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main(args: Dict,):
    print("Running with values: ")

    print(
        json.dumps(
            {k: args[k] for k in set(list(args.keys())) - set(["outdir"])}, indent=4,
        )
    )

    const_L = args["length"]
    const_dT = args["timestep"]
    const_Tmax = args["maxtime"]
    const_density = args["density"]

    const_nSteps = int(const_Tmax // const_dT)
    const_stepProb = const_dT * args["stepprob"]

    # Initialize the array for storing all positions over time

    position_time_array = np.zeros(
        shape=(const_nSteps, const_L)
    )  # now we access it as [time, position]

    # Populate the array at time 0
    if args["initrandom"]:  # Randomly populate
        position_time_array[0] = np.random.binomial(n=1, p=const_density, size=const_L)
    else:
        position_time_array[0][: int(const_L * const_density)] = 1

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

    # Calculate flux between timesteps
    fluxmat = np.diff(position_time_array, axis=0)
    fluxmat[fluxmat < 0] = 0
    J = fluxmat.sum(axis=1)

    if args["initrandom"]:  # Randomly populated means steady state more or less at once
        Jmean = J.mean()
        Jstd = J.std()
    else:  # only use last 20% to guess SS
        Jmean = J[-int(len(J) / 5.0) :].mean()
        Jstd = J[-int(len(J) / 5.0) :].std()

    J_theoretical = const_stepProb * const_density * (1 - const_density)

    (timepoints, positions) = np.nonzero(position_time_array)

    ax_traj: plt.Axes
    fig: plt.Figure

    fig = plt.Figure(figsize=(7, 8))
    gs = GridSpec(nrows=3, ncols=1, figure=fig)
    ax_traj = fig.add_subplot(gs[:-1])
    ax_flux = fig.add_subplot(gs[-1])

    ax_flux.plot(J, alpha=0.7, lw=1)
    ax_flux.axhline(
        Jmean,
        ls="-.",
        alpha=0.5,
        color="xkcd:pastel red",
        label=rf"SS at $\operatorname{{E}}[J] = {Jmean:.2f}$",
    )
    ax_flux.axhspan(
        Jmean - Jstd / 2, Jmean + Jstd / 2, alpha=0.2, color="xkcd:pastel orange",
    )

    ax_flux.axhline(
        J_theoretical * const_L,
        ls="--",
        c="g",
        label=rf"$J_{{theoretical}} \cdot L = {J_theoretical*const_L:.2f} $",
    )

    ax_flux.set_xlabel("Time")
    ax_flux.set_ylabel(r"# particles moved ($J_{empirical}/L$) ")
    ax_flux.legend(loc="upper right")

    ax_traj.set_title(
        rf"Simulation with $L={const_L}, p\Delta t = {const_stepProb}, \rho={const_density:.2f}$"
    )

    ax_traj.scatter(positions, timepoints, marker=">", c="k", s=4)

    if const_dT != 1:
        ax_traj.set_yticklabels([f"{v* const_dT:.2f}s" for v in ax_traj.get_yticks()])
        ax_flux.set_xticklabels([f"{v* const_dT:.2f}s" for v in ax_flux.get_xticks()])

    ax_traj.set_xlabel("Position")
    ax_traj.set_ylabel("Time")

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
    parser.add_argument("-t", "--maxtime", type=int, default=1000)
    parser.add_argument("-f", "--density", type=float, default=0.1)
    parser.add_argument(
        "-i",
        "--initrandom",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Initialize randomly.",
    )

    parser.add_argument("-o", "--outdir", type=str, default="./figs")
    parser.add_argument("--filetype", type=str, choices=["png", "pdf"], default="pdf")

    parser.add_argument(
        "--TEST", type=str2bool, nargs="?", const=True, default=False, help="Test Mode."
    )

    args = parser.parse_args()
    argdict = vars(args)  # returns a dict, easier to deal with

    if argdict["TEST"] == True:
        argdict["length"] = 15
        argdict["maxtime"] = 20
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
