from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
import numpy as np

from src.phase0.phase0_5_geometry import box_corners


def render_geometry(arrays: dict, metadata: dict, path: Path) -> None:
    grid = metadata["grid"]
    figure, axes = plt.subplots(2, 3, figsize=(13, 12), constrained_layout=True)
    colors = ("#2678b5", "#dc5635")
    targets = np.array([[0, 0], *metadata["future_waypoints"]])
    for t, axis in enumerate(axes.flat):
        occupancy = arrays["temporal_occupancy"][t]
        extent = (*grid["y_range_m"], *grid["x_range_m"])
        for channel, color in enumerate(colors):
            axis.imshow(np.ma.masked_where(~occupancy[channel], occupancy[channel]),
                        origin="lower", extent=extent, interpolation="nearest",
                        cmap=ListedColormap([color]), vmin=0, vmax=1, alpha=0.25)
        valid = arrays["agent_valid_mask"][t]
        for i in np.flatnonzero(valid):
            state = arrays["agent_numeric_state"][t, i]
            category = arrays["agent_category_id"][t, i]
            axis.add_patch(Polygon(box_corners(state)[:, [1, 0]], closed=True,
                                   fill=False, edgecolor=colors[category - 1], linewidth=0.7))
            if (grid["x_range_m"][0] < state[0] < grid["x_range_m"][1]
                    and grid["y_range_m"][0] < state[1] < grid["y_range_m"][1]):
                axis.text(state[1], state[0], str(i), fontsize=5, clip_on=True)
        axis.plot(targets[:, 1], targets[:, 0], "o-", color="#23904b", markersize=3,
                  label="GT ego trajectory")
        axis.plot(targets[t + 1, 1], targets[t + 1, 0], "o", color="#23904b", markersize=8)
        axis.plot(0, 0, "k^", markersize=6, label="Current ego origin")
        axis.set_xlim(grid["y_range_m"][1], grid["y_range_m"][0])
        axis.set_ylim(*grid["x_range_m"])
        axis.set_aspect("equal")
        axis.set_xlabel("y left (m)")
        axis.set_ylabel("x forward (m)")
        time = metadata["future_relative_times_sec"][t]
        axis.set_title(f"Step {t + 1}: sample offset {time:.3f}s | {valid.sum()} objects")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(loc="upper right", fontsize=7)
    figure.suptitle(
        f"GT evaluator geometry | {metadata['split']} | {metadata['sample_token']}\n"
        "Blue: vehicle | Orange: VRU | Numbers: stable instance slots | Shading: occupied cells\n"
        "All frames in current ego coordinates; unshaded cells are not certified free space",
        fontsize=12,
    )
    figure.savefig(path, dpi=140)
    plt.close(figure)
