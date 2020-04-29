from typing import List
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit



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

def plot_gaussian(
    data, ax: plt.Axes, nBins=100, textpos="l", legend=False, short_text=False
):
    # make sure our data is an ndarray
    if type(data) == list:
        data = np.array(data)

    ### FITTING WITH A GAUSSIAN

    def func_gauss(x, N, mu, sigma):
        return N * stats.norm.pdf(x, mu, sigma)

    counts, bin_edges = np.histogram(data, bins=nBins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    s_counts = np.sqrt(counts)

    x = bin_centers[counts > 0]
    y = counts[counts > 0]
    sy = s_counts[counts > 0]

    popt_gauss, pcov_gauss = curve_fit(
        func_gauss, x, y, p0=[1, data.mean(), data.std()]
    )

    y_func = func_gauss(x, *popt_gauss)

    pKS = stats.ks_2samp(y, y_func)
    pKS_g1, pKS_g2 = pKS[0], pKS[1]

    # print('LOOK! \n \n \n pKS is {} \n \n \n '.format(pKS_g2))
    chi2_gauss = sum((y - y_func) ** 2 / sy ** 2)
    NDOF_gauss = nBins - 3
    prob_gauss = stats.chi2.sf(chi2_gauss, NDOF_gauss)


    if short_text == True:
        namesl = [
            "Gauss_N",
            "Gauss_Mu",
            "Gauss_Sigma",
        ]
        valuesl = [
            "{:.3f} +/- {:.3f}".format(val, unc)
            for val, unc in zip(popt_gauss, np.diagonal(pcov_gauss))
        ]

        del namesl[0]  # remove gauss n
        del valuesl[0]
    else:
        namesl = [
            "Gauss_N",
            "Gauss_Mu",
            "Gauss_Sigma",
            "KS stat",
            "KS_pval",
            "Chi2 / NDOF",
            "Prob",
        ]
        valuesl = (
            [
                "{:.3f} +/- {:.3f}".format(val, unc)
                for val, unc in zip(popt_gauss, np.diagonal(pcov_gauss))
            ]
            + ["{:.3f}".format(pKS_g1)]
            + ["{:.3f}".format(pKS_g2)]
            + ["{:.3f} / {}".format(chi2_gauss, NDOF_gauss)]
            + ["{:.3f}".format(prob_gauss)]
        )

    ax.errorbar(x, y, yerr=sy, xerr=0, fmt=".", elinewidth=1)
    ax.plot(x, y_func, "--", label="Gaussian")
    if textpos == "l":
        ax.text(
            0.02,
            0.98,
            nice_string_output(namesl, valuesl),
            family="monospace",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            alpha=0.5,
        )
    elif textpos == "r":
        ax.text(
            0.6,
            0.98,
            nice_string_output(namesl, valuesl),
            family="monospace",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            alpha=0.5,
        )
    if legend:
        ax.legend(loc="center left")
    return ax



if __name__ == '__main__':
    samples = stats.expon.rvs(5.7, size=10000)
    # samples = stats.poisson.rvs(mu=2, size=10000)
    # samples = stats.cauchy.rvs(size=10000)


    sums = np.zeros(1000)

    for si in range(len(sums)):
        sums[si] = np.mean(np.random.choice(samples, size=10))


    fig, ax = plt.subplots()

    plot_gaussian(sums, ax)

    plt.show()