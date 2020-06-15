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
            name, value, spacing=extra_spacing + max_values + max_names - len(name),
        )
    return string[:-2]


def run(args):
    dt = args["dt"]
    n_samples = args["samples"]
    tmax = args["tmax"]

    mu = args["mu"]
    sigma = args["sigma"]

    # calculate stuff here
    n_steps = int(tmax / dt)

    g_ensemble = mu
    g_log = mu - sigma ** 2 / 2

    desc_str = nice_string_output(
        names=["dt", "tmax", "n_steps", "mu", "sigma", "g_ensemble", "g_log"],
        values=[
            f"{val:.3f}" for val in (dt, tmax, n_steps, mu, sigma, g_ensemble, g_log)
        ],
    )
    print(
        f"Simulating Geometric Brownian Motion as a Financial Market Model "
        f"for {n_samples}, with the Ito Interpretation\n"
    )
    print(desc_str)
    # Ito interpretation Euler Method

    S = np.zeros((n_steps, n_samples))
    S[0] = 1.0

    S_65 = np.copy(S)

    dBs = np.random.normal(
        0,
        scale=dt,
        # scale=np.sqrt(dt),
        size=(n_steps - 1, n_samples)
    )  # consider changing scale to sqrt(dt)?

    Bs = np.cumsum(dBs, axis=0)

    timepoints = np.arange(0, n_steps) * dt

    for i_t in range(n_steps - 1):
        S_t = S[i_t]
        S_tplus1 = np.copy(S_t)
        delta_B = dBs[i_t]
        delta_S = mu * S_t * dt + sigma * S_t * delta_B  # elementwise multiplication
        S_tplus1 += delta_S
        S[i_t + 1] = S_tplus1

    for i in range(n_samples):
        S_65[1:, i] = np.exp(g_ensemble * timepoints[1:] + sigma * Bs[:, i])

    # Now compare

    fig, ax = plt.subplots()
    ax: plt.Axes

    ax.set_title(
        rf"$T_{{max}}= {tmax:.1f},dt={dt:.3f},N={int(n_samples)}, \mu={mu:.2f}, \sigma={sigma:.2f}$"
    )
    ax.plot(timepoints, S, label="Direct Integration", alpha=0.5, ls="--")
    ax.plot(timepoints, S_65, label="Using Eqn. 6.5" + "\n"
                                                       rf"$g_{{ensemble}}={g_ensemble:.3f}$", alpha=0.5, ls="-.")

    ax.legend()

    fig.tight_layout()

    outname = args["outdir"] / (args["out_name"] + "." + args["filetype"])
    fig.savefig(outname)
    print("Saved to")
    print(outname)


def parse_arguments() -> Dict[str, Union[int, float, str, Path]]:
    parser = argparse.ArgumentParser(description="Chapter 6 exercise 3a code")
    parser.add_argument("-N", "--samples", type=int, default=1)
    parser.add_argument("-T", "--tmax", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)

    parser.add_argument("-u", "--mu", type=float, default=0.5, help="Rate of Return.")
    parser.add_argument(
        "-s", "--sigma", type=float, default=0.5, help="Volatility term."
    )

    parser.add_argument("--seed", type=int, default=24)

    parser.add_argument("-o", "--outdir", type=str, default="./figs")
    parser.add_argument("--filetype", type=str, choices=["png", "pdf"], default="pdf")

    args = parser.parse_args()
    argdict = vars(args)  # returns a dict, easier to deal with
    po = Path(argdict["outdir"])
    if not po.exists():
        po.mkdir()
    print("Set output dir to: " + str(po.absolute()))
    argdict["outdir"] = po
    np.random.seed(argdict['seed'])
    # Set output name here
    timestr = time.strftime("%Y%m%d_%H%M%S")
    out_str = f"ch6_e3a_" + timestr

    argdict["out_name"] = out_str

    return argdict


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
