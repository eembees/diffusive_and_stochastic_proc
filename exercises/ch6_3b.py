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
    dr = args["dr"]
    n_samples = args["samples"]
    tmax = args["tmax"]

    mu = args["mu"]
    sigma = args["sigma"]

    money_start = args["money"]

    # calculate stuff here
    n_steps = int(tmax / dt)
    rs = np.arange(dr, 1 + dr, step=dr)
    n_r = len(rs)

    g_ensemble = mu
    g_log = mu - sigma ** 2 / 2

    r_opt = mu / (sigma ** 2)

    desc_str = nice_string_output(
        names=["dt", "tmax", "n_steps", "mu", "sigma", "N_r"],
        values=[
            f"{val:.3f}" for val in (dt, tmax, n_steps, mu, sigma, n_r)
        ],
    )
    print(
        f"Simulating Geometric Brownian Motion as a Financial Market Model with Bet Hedging"
        f"for {n_samples}, with the Ito Interpretation\n"
    )
    print(desc_str)

    # Ito interpretation Euler Method

    def g_expected(r: np.ndarray) -> np.ndarray:
        """
        Expected growth rate.
        """
        return r * mu - (r ** 2) * ((sigma ** 2) / 2)

    S = np.zeros((n_steps, n_samples))
    S[0] = 1.0 # intial stock price is 1.0

    dBs = np.random.normal(
        0,
        # scale=dt,
        scale=np.sqrt(dt),
        size=(n_steps - 1, n_samples)
    )

    Bs = np.cumsum(dBs, axis=0)

    timepoints = np.arange(0, n_steps) * dt

    for i in range(n_samples):
        S[1:, i] = np.exp(g_ensemble * timepoints[1:] + sigma * Bs[:, i])

    deltaS = np.diff(S, axis=0)

    g_ens_arr = np.zeros((n_r, 2))
    g_log_arr = np.zeros((n_r, 2))

    for r_i, r in enumerate(rs):
        Mr = np.zeros_like(S)
        Mr[0] = money_start

        for i_t in range(n_steps - 1):
            Mr_t = Mr[i_t]
            Mr_tplus1 = np.copy(Mr_t)
            deltaS_t = deltaS[i_t]
            deltaM_t = Mr_t * deltaS_t * r
            Mr_tplus1 += deltaM_t
            Mr[i_t + 1] = Mr_tplus1

        M_last = Mr[-1]
        interm = np.log(M_last)

        g_ens_mean = np.nanmean() / tmax
        g_ens_std = np.nanstd(np.log(M_last)) / tmax

        g_log_mean = np.log(np.nanmean(M_last)) / tmax
        g_log_std = np.log(np.nanstd(M_last)) / tmax

        g_ens_arr[r_i] = [g_ens_mean, g_ens_std]
        g_log_arr[r_i] = [g_log_mean, g_log_std]


    ax_ens: plt.Axes
    fig, axes = plt.subplots(nrows=4, sharex=True, figsize = (6, 9))

    ax_M, ax_ens, ax_log, ax_long = axes.ravel()
    ax_M.set_title(rf"Analytical expectation of $g(M)$ with $\mu={mu:.2f}, \sigma={sigma:.2f}$")
    ax_ens.set_title(
        rf"$T_{{max}}={tmax:.1f}, dt={dt:.3f}, N={int(n_samples)}, \mu={mu:.2f}, \sigma={sigma:.2f}$"
    )
    ax_ens.set_ylabel(r"Ensemble Growth Rate $\ln \langle M(t) \rangle$")
    ax_log.set_ylabel(r"Log Growth Rate $\ln \langle M(t) \rangle$")

    ax_long.set_xlabel("Rate of Investment $r$")
    ax_ens.plot(rs, g_ens_arr[:, 0], label=r"$G_{ensemble} = \ln \langle M(t) \rangle$", alpha=0.7, ls="--")

    ax_M.plot(rs, g_expected(rs), label=r"$G_{M} = r\mu - r^2 \frac{\sigma^2}{2}$", alpha=0.7, ls="dotted")

    ax_log.plot(rs, g_log_arr[:, 0], label=r"$G_{log} = \ln \langle M(t) \rangle$", alpha=0.7, ls="--")

    ax_ens.fill_between(rs, g_ens_arr[:, 0] - g_ens_arr[:, 1], g_ens_arr[:, 0] + g_ens_arr[:, 1], alpha=0.3)
    ax_log.fill_between(rs, g_log_arr[:, 0] - g_log_arr[:, 1], g_log_arr[:, 0] + g_log_arr[:, 1], alpha=0.3)

    ax_ens.axvline(r_opt, label=r"$r_{opt} = \frac{\mu}{\sigma^2}$", color="red", alpha=0.6)
    ax_log.axvline(r_opt, label=r"$r_{opt} = \frac{\mu}{\sigma^2}$", color="red", alpha=0.6)
    ax_M.axvline(r_opt, label=r"$r_{opt} = \frac{\mu}{\sigma^2}$", color="red", alpha=0.6)

    # ax_log.get_shared_y_axes().join(ax_log, ax_ens)
    ax_ens.legend(loc="upper right")
    ax_M.legend(loc="upper right")
    ax_log.legend(loc="upper right")

    fig.tight_layout()

    # plt.show()
    outname = args["outdir"] / (args["out_name"] + "." + args["filetype"])
    fig.savefig(outname)
    print("Saved to")
    print(outname)


def parse_arguments() -> Dict[str, Union[int, float, str, Path]]:
    parser = argparse.ArgumentParser(description="Chapter 6 exercise 3a code")
    parser.add_argument("-N", "--samples", type=int, default=100)
    parser.add_argument("-T", "--tmax", type=float, default=5.0)
    parser.add_argument("-M", "--money", type=float, default=1000.0)

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dr", type=float, default=0.01)

    parser.add_argument("-u", "--mu", type=float, default=0.1, help="Rate of Return.")
    parser.add_argument(
        "-s", "--sigma", type=float, default=0.5, help="Volatility term."
    )

    parser.add_argument("--seed", type=int, default=24)

    parser.add_argument("-o", "--outdir", type=str, default="./figs")
    parser.add_argument("--filetype", type=str, choices=["png", "pdf"], default="pdf")

    args = parser.parse_args()
    argdict = vars(args)
    po = Path(argdict["outdir"])
    if not po.exists():
        po.mkdir()
    print("Set output dir to: " + str(po.absolute()))
    argdict["outdir"] = po
    np.random.seed(argdict['seed'])
    # Set output name here
    timestr = time.strftime("%Y%m%d_%H%M%S")
    out_str = f"ch6_e3b_" + timestr

    argdict["out_name"] = out_str

    return argdict


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
