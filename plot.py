import numpy as np
from scipy.interpolate import interp1d
from jax import vmap
from datetime import datetime

def plot_helper(res, data, ntips):
    end = data.end

    top = (res.global_posteriors["origin"] + res.global_posteriors["origin_start"])[
        :, 0
    ]

    coal_times = []
    kth = 1
    for j, (lp, td) in enumerate(zip(res.local_posteriors, data.tds)):
        root_heights = lp["root_proportion"] * (
            res.global_posteriors["origin"]
            + res.global_posteriors["origin_start"][0]
            - td.sample_times.max()
        )

        node_heights = vmap(td.height_transform)(
            root_heights,
            lp["proportions"],
        )[:, ntips // 2 :]
        kth_node_heights = np.partition(node_heights, kth, axis=1)[:, kth - 1]
        coal_times.append(kth_node_heights)
    coal_times = np.array(coal_times)
    #     start = np.median(coal_times, axis=0)
    start = np.min(coal_times, axis=0)

    x0 = np.linspace(
        end
        - np.quantile(res.global_posteriors["origin"].flatten(), 0.5)
        - data.earliest,
        end,
        100,
    )
    return start, top, end, x0


def plot_one(res, ax, param, m, start, top, end, x0, label, ci, title):
    y0 = []
    for x1, x2, y in zip(start, top, res.global_posteriors[param]):
        intervals = np.linspace(x1, x2, m + 1)
        t = (end - intervals)[::-1]
        y0.append(interp1d(t[1:], y, kind="nearest", bounds_error=False)(x0))
    q25, q50, q75 = np.nanquantile(np.array(y0), q=[0.025, 0.5, 0.975], axis=0)

    color = next(ax._get_lines.prop_cycler)["color"]
    ax.plot(x0, q50, label=label, color=color)

    if ci == "fill":
        ax.fill_between(x0, q25, q75, alpha=0.1, label="_nolegend_", color=color)
    if ci == "lines":
        ax.plot(x0, q25, "--", label="_nolegend_", alpha=0.25, color=color)
        ax.plot(x0, q75, "--", label="_nolegend_", alpha=0.25, color=color)
    # plt.xlim(reversed(plt.xlim()))
    # ax.set_xlabel("Year")
    ax.set_title(title)


def plot_by_param(res, data, axs, m, regions, param, **kwargs):
    if "label" not in kwargs:
        kwargs["label"] = "VBSKY"
    if "ci" not in kwargs:
        kwargs["ci"] = "fill"
    if "ntips" not in kwargs:
        kwargs["ntips"] = 200

    for ax, r in zip(axs, regions):
        start, top, end, x0 = plot_helper(res[r], data[r], kwargs["ntips"])
        if r != "usa":
            title = r.title()
        else:
            title = r.upper()
        plot_one(
            res[r], ax, param, m, start, top, end, x0, kwargs["label"], kwargs["ci"], title
        )


def plot_by_region(res, data, axs, m, params, region, **kwargs):
    if "label" not in kwargs:
        kwargs["label"] = "VBSKY"
    if "ci" not in kwargs:
        kwargs["ci"] = "fill"
    if "ntips" not in kwargs:
        kwargs["ntips"] = 200

    for ax, p in zip(axs, params):
        start, top, end, x0 = plot_helper(res[region], data[region], kwargs["ntips"])
        plot_one(
            res[region], ax, p, m, start, top, end, x0, kwargs["label"], kwargs["ci"], p
        )