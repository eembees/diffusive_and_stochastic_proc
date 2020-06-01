# coding=utf-8
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

# plt.xkcd()
# set std params here because lazy
color_xa = "xkcd:pastel orange"
color_xb = "xkcd:pastel pink"
color_xc = "xkcd:forest green"


# lib functions
def nice_string_output(
    names: List[str], values: List[str], extra_spacing: int = 0,
):
    max_values = len(max(values, key=len))
    max_names = len(max(names, key=len))
    string = ""
    for name, value in zip(names, values):
        string += "{0:s} {1:>{spacing}} \n".format(
            name, value, spacing=extra_spacing + max_values + max_names - len(name),
        )
    return string[:-2]


def calculate_joint_rates_and_prob(*rates: Tuple[float]) -> Tuple[float, float]:
    """
    Returns joint rate and probability
    """
    K = np.sum(rates)[0]
    P = 1.0 / K

    return K, P


def run(args: Dict):
    TEST = args["test"]
    max_time = args["maxtime"]
    rate_production = args["alpha"]
    volume = args["volume"]
    rate_degradation = args["gamma"]

    plot_range = (0, 20)

    curr_time = 0
    curr_n_mrna = 0

    storage_mrna = []
    storage_time = []
    storage_steps = []

    while curr_time < max_time:
        curr_r_prod = rate_production * volume
        curr_r_deg = rate_degradation * curr_n_mrna

        # 1. calculate K
        K = sum((curr_r_deg, curr_r_prod))

        # 2. draw a random number a and find duration until next event bu -ln a /K
        a = np.random.uniform(0, 1, size=1)[0]

        tau = -np.log(a) / K

        # 3. Draw a new number b
        b = np.random.uniform(0, 1, size=1)[0]

        # TODO: make this argmin based for n r
        if b < curr_r_prod / K:  # produce
            curr_n_mrna += 1
            if TEST:
                print(f"Produced one MRNA! Count: {curr_n_mrna}")
        else:
            curr_n_mrna -= 1

            if TEST:
                print(f"Degraded one MRNA! Count: {curr_n_mrna}")

        curr_time += tau
        if TEST:
            print(f"Incremented time by: {tau:.2f} --> {curr_time:.2f}")

        storage_mrna.append(curr_n_mrna)
        storage_time.append(curr_time)
        storage_steps.append(tau)


    df = pd.DataFrame({"Time": storage_time, "N_MRNA": storage_mrna, "Steps":storage_steps})
    print(df.tail())
    print(f"N steps:{len(df)}")

    # plotting

    n_mean = sum(df["N_MRNA"] * df["Steps"]) / max_time


    param_str = nice_string_output(
        names=["alpha", "gamma", "volume", "max_time"],
        values=[
            f"{v:.2f}"
            for v in (args["alpha"], args["gamma"], args["volume"], args["maxtime"])
        ],
    )
    poisson_lambda = args["alpha"] * args["volume"] / args["gamma"]
    result_str = nice_string_output(
        names=[r"Poisson param", "Mean", "Var", "Fano", "N steps", "Mean Step"],
        values=[
            f"{v:.2f}"
            for v in (
                poisson_lambda,
                n_mean,
                # df["N_MRNA"].mean(),
                df["N_MRNA"].var(),
                df["N_MRNA"].var() / n_mean,
                len(df),
                df["Time"].diff().mean(),
            )
        ],
        extra_spacing=2,
    )

    print(param_str)
    print(result_str)

    ax_dist: plt.Axes
    ax_traj: plt.Axes
    fig: plt.Figure

    fig, (ax_traj, ax_dist) = plt.subplots(nrows=2, figsize=(6, 8))

    ax_traj.set_title("Time plot of Gillespie Algorithm")

    ax_traj.scatter(
        df["Time"], df["N_MRNA"], label="Simulated", c=color_xa, alpha=0.7, marker="2"
    )
    ax_traj.axhline(poisson_lambda, ls="-.", label="Analytical Mean")
    ax_traj.axhspan(
        poisson_lambda - poisson_lambda ** 0.5,
        poisson_lambda + poisson_lambda ** 0.5,
        ls="-.",
        label="Analytical STD",
        alpha=0.2,
        color=color_xc,
    )

    ax_traj.set_ylabel("Number of MRNA")
    ax_traj.set_xlabel("Time")

    ax_traj.legend(loc="upper left")

    ax_dist.hist(
        df["N_MRNA"],
        histtype="step",
        bins=max(plot_range),
        range=plot_range,
        label="MRNA Number",
        color=color_xa,
        density=True,
    )

    ax_dist.set_xlabel("Number of MRNA")
    ax_dist.set_ylabel("Density")
    ax_dist.set_title(
        rf"Poisson with $\lambda = {poisson_lambda:.3f}$ vs. Empirical Distribution"
    )

    analytical_poisson = scipy.stats.poisson(poisson_lambda)
    analytical_poisson_x = np.arange(*plot_range)
    analytical_poisson_y = analytical_poisson.pmf(analytical_poisson_x)

    ax_dist.vlines(
        analytical_poisson_x + 0.5,
        0,
        analytical_poisson_y,
        colors=color_xc,
        label="Analytical Poisson PMF",
        alpha=0.5,
    )

    ax_dist.legend()

    ax_traj.text(
        s="PARAMS:\n" + param_str, x=max_time * 0.7, y=9,
    )
    ax_dist.text(
        s="RESULTS:\n" + result_str, y=0.05, x=12,
    )

    fig.tight_layout()

    basename = args["out_name"] + "_simulation." + args["filetype"]
    fig.savefig(args["outdir"] / basename)
    fig.clf()


def parse_arguments() -> Dict[str, Union[int, float, str, Path]]:
    parser = argparse.ArgumentParser(description="Chapter 7 exercise 8 code")

    parser.add_argument("-a", "--alpha", type=float, default=3.0)
    parser.add_argument("-g", "--gamma", type=float, default=0.5)
    parser.add_argument("-v", "--volume", type=float, default=1.0)
    parser.add_argument("-t", "--maxtime", type=int, default=1000)

    parser.add_argument("-o", "--outdir", type=str, default="./figs")
    parser.add_argument("--filetype", type=str, choices=["png", "pdf"], default="pdf")

    parser.add_argument("--test", type=bool, default=False)  # change to false once done

    args = parser.parse_args()
    argdict = vars(args)  # returns a dict, easier to deal with

    if argdict["test"] == True:
        argdict["maxtime"] = 100

    po = Path(argdict["outdir"])
    if not po.exists():
        po.mkdir()
    print("Set output dir to: " + str(po.absolute()))
    argdict["outdir"] = po
    # Set output name here
    timestr = timestr = time.strftime("%Y%m%d_%H%M%S")
    out_str = f"ch7_e8_" + timestr

    argdict["out_name"] = out_str

    return argdict


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
