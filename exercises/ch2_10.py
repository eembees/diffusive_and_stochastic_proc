# coding=utf-8
import time
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
from matplotlib import pyplot as plt
import argparse


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


# utility functions


def variance_harmonic(t, D, eta, k) -> np.ndarray:
    _k = eta / k
    return D * _k * (1 - np.exp(-2 * _k * t))


def F(x):
    return -1 * args["spring"] * x * args["inv_friction"]


# Main running functions


def run_a(args):
    D = args["diffusion"]
    dt1 = args["dt1"]
    dt2 = args["dt2"]
    var1 = 2 * D * dt1
    var2 = 2 * D * dt2
    std1 = np.sqrt(var1)
    std2 = np.sqrt(var2)
    first_str = (
        "a. In order to scale the normal distribution properly, multiply events with the \n"
        "square root of the variance of the delta W distribution.\n"
        "In this case, we have the following parameters:\n"
    )
    second_str = nice_string_output(
        names=["D", "dt1", "dt2", "var1", "std1", "var2", "std2"],
        values=[f"{val:.4f}" for val in (D, dt1, dt2, var1, std1, var2, std2)],
        extra_spacing=4,
    )
    print(first_str + second_str)


def run_b(args):
    t_max = args["tmax"]
    D = args["diffusion"]
    dt1 = args["dt1"]
    dt2 = args["dt2"]
    n_samples = args["samples"]

    var1 = 2 * D * dt1
    var2 = 2 * D * dt2
    std1 = np.sqrt(var1)
    std2 = np.sqrt(var2)

    print(
        "b. Simulating Brownian motion for "
        f"{n_samples} particles, with\n"
        + nice_string_output(
            names=["D", "dt1", "dt2", "var1", "std1", "var2", "std2"],
            values=[
                f"{val:.4f}" for val in (D, dt1, dt2, var1, std1, var2, std2)
            ],
            extra_spacing=4,
        )
    )

    # mean of distribution is 0, var is 2D dt
    n_1 = int(t_max / dt1)
    n_2 = int(t_max / dt2)

    ts_1 = np.linspace(0, t_max, num=n_1)
    ts_2 = np.linspace(0, t_max, num=n_2)

    var1 = 2 * D * dt1
    var2 = 2 * D * dt2

    W1 = np.random.normal(0, np.sqrt(var1), size=(n_1, n_samples))
    W2 = np.random.normal(0, np.sqrt(var2), size=(n_2, n_samples))

    # Set initial position to 0
    W1[0] = 0.0
    W2[0] = 0.0

    X1 = np.cumsum(W1, axis=0)
    X2 = np.cumsum(W2, axis=0)

    exp_variance_1 = np.var(X1, axis=1)
    exp_variance_2 = np.var(X2, axis=1)
    exp_average_1 = np.mean(X1, axis=1)
    exp_average_2 = np.mean(X2, axis=1)

    fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, figsize=(8, 5))
    #
    ax[0, 0].plot(ts_1, X1[:, :10], alpha=0.6, lw=1)
    ax[0, 0].set_title("Positions with dt = " + f"{dt1:.4f}")
    ax[0, 0].axhline(0, 0, t_max, ls="--", lw=1, color="k")

    ax[1, 0].plot(ts_2, X2[:, :10], alpha=0.6, lw=1)
    ax[1, 0].set_title("Positions with dt = " + f"{dt2:.4f}")
    ax[1, 0].axhline(0, 0, t_max, ls="--", lw=1, color="k")

    ax[0, 1].set_title("Positions with dt = " + f"{dt1:.4f}")
    ax[0, 1].plot(
        ts_1, exp_variance_1, label=r"$\operatorname{Var}[X]$", alpha=0.7
    )
    ax[0, 1].plot(
        ts_1, exp_average_1, label=r"$\operatorname{E}[X]$", alpha=0.7
    )
    ax[0, 1].plot(
        ts_1,
        2 * D * ts_1,
        label=r"$2\cdot D \cdot t$",
        ls="--",
        lw=1,
        alpha=0.5,
    )

    ax[1, 1].set_title("Positions with dt = " + f"{dt2:.4f}")
    ax[1, 1].plot(
        ts_2, exp_variance_2, label=r"$\operatorname{Var}[X]$", alpha=0.7
    )
    ax[1, 1].plot(
        ts_2, exp_average_2, label=r"$\operatorname{E}[X]$", alpha=0.7
    )
    ax[1, 1].plot(
        ts_2,
        2 * D * ts_2,
        label=r"$2\cdot D \cdot t$",
        ls="--",
        lw=1,
        alpha=0.5,
    )

    for a in ax.ravel():
        if sum(True if h else False for h in a.get_legend_handles_labels()) > 0:
            a.legend()

    basename = args["out_name"] + "_brownian." + args["filetype"]
    fig.suptitle(" Brownian Motion " + rf"$D={D:.2f}, N={n_samples}$")

    fig.savefig(args["outdir"] / basename)
    fig.clf()


def run_c(args):
    k = args["spring"]
    eta = args["friction"]

    t_max = args["tmax"]
    D = args["diffusion"]
    dt1 = args["dt1"]
    dt2 = args["dt2"]
    n_samples = args["samples"]

    var1 = 2 * D * dt1
    var2 = 2 * D * dt2
    std1 = np.sqrt(var1)
    std2 = np.sqrt(var2)
    max_var = D * eta / k

    print(
        "c. Simulating Harmonic motion for "
        f"{n_samples} particles, with\n"
        + nice_string_output(
            names=[
                "D",
                "k",
                "eta",
                "lim variance",
                "dt1",
                "dt2",
                "var1",
                "std1",
                "var2",
                "std2",
            ],
            values=[
                f"{val:.4f}"
                for val in (
                    D,
                    k,
                    eta,
                    max_var,
                    dt1,
                    dt2,
                    var1,
                    std1,
                    var2,
                    std2,
                )
            ],
            extra_spacing=4,
        )
    )
    # now adding harmonic potential
    def dX1(x):
        f = F(x) * dt1
        w = np.random.normal(0, np.sqrt(var1), size=np.shape(x))
        return f + w

    def dX2(x):
        f = F(x) * dt2
        w = np.random.normal(0, np.sqrt(var2), size=np.shape(x))
        return f + w

    n_1 = int(t_max / dt1)
    n_2 = int(t_max / dt2)

    ts_1 = np.linspace(0, t_max, num=n_1)
    ts_2 = np.linspace(0, t_max, num=n_2)

    var1 = 2 * D * dt1
    var2 = 2 * D * dt2

    X1 = np.zeros((n_1, n_samples), dtype=float)
    X2 = np.zeros((n_2, n_samples), dtype=float)

    for t in range(1, n_1):
        X1[t] = X1[t - 1] + dX1(X1[t - 1])
    for t in range(1, n_2):
        X2[t] = X2[t - 1] + dX2(X2[t - 1])

    exp_variance_1 = np.var(X1, axis=1)
    exp_variance_2 = np.var(X2, axis=1)
    exp_average_1 = np.mean(X1, axis=1)
    exp_average_2 = np.mean(X2, axis=1)

    analytical_variance = variance_harmonic(ts_1, D, eta, k)
    analytical_variance_str = r"$D\frac{\eta}{k}\left(1-\exp \left( \frac{-2kt}{\eta} \right) \right)$"

    fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, figsize=(8, 5))

    ax[0, 0].plot(ts_1, X1[:, :10], alpha=0.6, lw=1)
    ax[0, 0].set_title("Positions with dt = " + f"{dt1:.4f}")
    ax[0, 0].axhline(0, 0, t_max, ls="--", lw=1, color="k")

    ax[1, 0].plot(ts_2, X2[:, :10], alpha=0.6, lw=1)
    ax[1, 0].set_title("Positions with dt = " + f"{dt2:.4f}")
    ax[1, 0].axhline(0, 0, t_max, ls="--", lw=1, color="k")

    ax[0, 1].set_title("Positions with dt = " + f"{dt1:.4f}")
    ax[0, 1].plot(
        ts_1, exp_variance_1, label=r"$\operatorname{Var}[X]$", alpha=0.7
    )
    ax[0, 1].plot(
        ts_1,
        analytical_variance,
        label=analytical_variance_str,
        ls="--",
        alpha=0.7,
    )
    ax[0, 1].plot(
        ts_1, exp_average_1, label=r"$\operatorname{E}[X]$", alpha=0.7
    )

    ax[1, 1].set_title("Positions with dt = " + f"{dt2:.4f}")
    ax[1, 1].plot(
        ts_2, exp_variance_2, label=r"$\operatorname{Var}[X]$", alpha=0.7
    )
    ax[1, 1].plot(
        ts_1,
        analytical_variance,
        label=analytical_variance_str,
        ls="--",
        alpha=0.7,
    )
    ax[1, 1].plot(
        ts_2, exp_average_2, label=r"$\operatorname{E}[X]$", alpha=0.7
    )

    for a in ax.ravel():
        if sum(True if h else False for h in a.get_legend_handles_labels()) > 0:
            a.legend()

    fig.suptitle(
        "Harmonic Brownian Motion "
        + rf"$D={D:.2f}, N={n_samples},\eta={eta:.2f}, k={k:.2f}, D={D:.2f}$"
    )

    basename = args["out_name"] + "_harmonic." + args["filetype"]
    fig.savefig(args["outdir"] / basename)
    fig.clf()


def main(args: Dict):
    _run = args["run"]
    if _run in ["a", "all"]:
        run_a(args)
    if _run in ["b", "all"]:
        run_b(args)
    if _run in ["c", "all"]:
        run_c(args)


def parse_arguments() -> Dict[str, Union[int, float, str, Path]]:
    parser = argparse.ArgumentParser(description="Chapter 2 exercise 10 code")
    parser.add_argument(
        "-r", "--run", type=str, choices=["a", "b", "c", "all"], default="all"
    )
    parser.add_argument("-N", "--samples", type=int, default=1024)
    parser.add_argument("-T", "--tmax", type=float, default=10.0)
    parser.add_argument("-D", "--diffusion", type=float, default=1.0)
    parser.add_argument("--dt1", type=float, default=0.1)
    parser.add_argument("--dt2", type=float, default=0.01)
    parser.add_argument("-k", "--spring", type=float, default=1.0)
    parser.add_argument("-eta", "--friction", type=float, default=1.0)
    parser.add_argument("-o", "--outdir", type=str, default="./figs")
    parser.add_argument(
        "--filetype", type=str, choices=["png", "pdf"], default="png"
    )

    args = parser.parse_args()
    argdict = vars(args)  # returns a dict, easier to deal with
    po = Path(argdict["outdir"])
    if not po.exists():
        po.mkdir()
    print("Set output dir to: " + str(po.absolute()))
    argdict["outdir"] = po
    argdict["inv_friction"] = 1.0 / argdict["friction"]
    # Set output name here
    timestr = timestr = time.strftime("%Y%m%d_%H%M%S")
    out_str = f"ch2_e10_" + timestr

    argdict["out_name"] = out_str

    return argdict


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
