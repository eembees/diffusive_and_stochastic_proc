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

# set std params here because lazy
color_xa = "xkcd:pastel orange"
color_xb = "xkcd:pastel pink"
color_xc = "xkcd:forest green"


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


# math
def U(x):
    return (-1.0 / 3) * args["a"] * (x ** 3) + (1 / 2) * args["b"] * (x ** 2)


def F(x):
    return -1 * args["inv_friction"] * (-1 * args["a"] * x ** 2 + args["b"] * x)


def run(args):
    eta = args["friction"]
    eta_inv = args["inv_friction"]
    t_max = args["tmax"]
    D = args["diffusion"]
    dt = args["dt"]
    n_samples = args["samples"]

    a = args["a"]
    b = args["b"]

    x_a = 0
    x_b = (3 * b) / (2 * a)
    x_c = b / a

    x_range = (x_a - np.abs(x_a - x_c), x_b)
    desc_str = nice_string_output(
        names=["Diffusion", "friction", "dt", "a", "b", "tmax"],
        values=[f"{val:.3f}" for val in (D, eta, dt, a, b, t_max)],
    )
    print(f"Simulating Langevin Motion for {n_samples}, with\n")
    print(desc_str)

    def dX(x):
        f = F(x) * dt
        w = np.random.normal(0, np.sqrt(2 * D * dt), size=np.shape(x))
        return f + w

    n_steps = int(t_max / dt)

    ts = np.linspace(0, t_max, num=n_steps)

    X = np.zeros((n_steps, n_samples), dtype=float)

    for t in range(1, n_steps):
        X[t] = X[t - 1] + dX(X[t - 1])

    # find first passage times for each trajectory
    first_passage_idx = np.zeros_like(X[0], dtype=int)

    for i, trajectory in enumerate(X.T):
        first_passage_idx[i] = np.argmax(trajectory >= x_b)

    first_passage_times = first_passage_idx * dt

    fig = plt.Figure(figsize=(8, 5))
    outer_grid = GridSpec(nrows=1, ncols=2, figure=fig)
    inner_rgt = GridSpecFromSubplotSpec(
        ncols=1, nrows=2, subplot_spec=outer_grid[1]
    )

    ax_hist = fig.add_subplot(inner_rgt[1])  # type:plt.Axes
    ax_traj = fig.add_subplot(inner_rgt[0], sharex=ax_hist)  # type:plt.Axes
    ax_pot = fig.add_subplot(outer_grid[0])  # type:plt.Axes

    pot_x = np.linspace(*x_range, num=1000)
    pot_y = U(pot_x)
    ax_pot.plot(pot_x, pot_y, ls="--", c="xkcd:hot pink", label="Potential")
    ax_pot.axvline(x_a, ls="-.", c=color_xa, label=r"$x_A$", lw=0.7)
    ax_pot.axvline(x_b, ls="-.", c=color_xb, label=r"$x_B$", lw=0.7)
    ax_pot.axvline(x_c, ls="-.", c=color_xc, label=r"$x_C$", lw=0.7)
    ax_pot.set_title("Potentials")

    ax_pot.text(
        s=desc_str,
        y=0.45 * max(pot_y),
        x=x_a - np.abs(x_a - x_c) / 2,
        alpha=0.7,
    )

    ax_pot.legend()

    n_trajs_to_plot = 100
    counter = 0
    idx = 0
    trajs_to_plot: List[Tuple[np.ndarray, np.ndarray]] = []
    while counter < n_trajs_to_plot:
        fp_idx = first_passage_idx[idx]
        if fp_idx > 0:
            trajs_to_plot.append((ts[:fp_idx], X.T[idx][:fp_idx]))
            counter += 1
        idx += 1
        if idx >= len(X.T) - 1:
            break

    mean_fp = np.mean(first_passage_times[first_passage_times > 0])

    analytical_rate = (D * b / (2 * np.pi * 1)) * np.exp(
        -(b ** 3) / (6 * a ** 2)
    )

    # ax_traj.plot(ts, trajs_to_plot, lw=1, alpha=0.6)

    for traj in trajs_to_plot:
        ax_traj.plot(*traj, lw=0.5, alpha=0.7)
    # ax_traj.plot(ts, X[:, :10])
    ax_traj.set_title(f"{len(trajs_to_plot)} trajectories")
    ax_traj.axhline(x_b, ls="-.", c=color_xb, label=r"$x_B$", lw=0.7)
    ax_traj.axhline(x_c, ls="-.", c=color_xc, label=r"$x_C$", lw=0.7)
    ax_traj.axhline(x_a, ls="-.", c=color_xa, label=r"$x_A$", lw=0.7)

    ax_traj.set_ylim(x_range[0], x_range[1] + 1)
    ax_traj.xaxis.set_visible(False)

    ax_hist.hist(
        first_passage_times[first_passage_times > 0],
        histtype="step",
        bins=25,
        label=fr"$r=\frac{{1}}{{\mu}}={1 / mean_fp:.3f}$",
        density=True,
    )
    ax_hist.axvline(
        mean_fp,
        label=rf"$\mu = {mean_fp:.1f}$"
        "\n"
        rf"$\frac{{Db}}{{2\pi k_BT}}e^{{\left(\frac{{-b^3}}{{6a^2 k_BT}}\right)}}={analytical_rate:.4f}$",
        ls="--",
        color="r",
        lw=1,
    )
    ax_hist.set_title(
        f"First Passage Times, {sum(first_passage_times > 0)}/{n_samples}"
    )
    ax_hist.legend(loc="lower right")

    basename = args["out_name"] + "_simulation." + args["filetype"]
    fig.savefig(args["outdir"] / basename)
    fig.clf()


def parse_arguments() -> Dict[str, Union[int, float, str, Path]]:
    parser = argparse.ArgumentParser(description="Chapter 2 exercise 10 code")
    parser.add_argument("-N", "--samples", type=int, default=1024)
    parser.add_argument("-T", "--tmax", type=float, default=20.0)
    parser.add_argument("-D", "--diffusion", type=float, default=1.0)
    parser.add_argument("-a", type=float, default=0.2)
    parser.add_argument("-b", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("-eta", "--friction", type=float, default=1.0)
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
    argdict["inv_friction"] = 1.0 / argdict["friction"]
    # Set output name here
    timestr = timestr = time.strftime("%Y%m%d_%H%M%S")
    out_str = f"ch4_e3_" + timestr

    argdict["out_name"] = out_str

    return argdict


def main(args: Dict[str, Union[int, float, str, Path]]):
    pass


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
