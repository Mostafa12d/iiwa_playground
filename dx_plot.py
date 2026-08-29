"""Plot the Cartesian position error dx logged during an impedance-control run."""

import matplotlib

matplotlib.use("Agg")  # headless: mjpython owns the GUI loop

import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots 1-3 for x/y/z, violet for the derived norm.
SERIES = {"x": "#2a78d6", "y": "#eb6834", "z": "#1baf7a"}
NORM = "#4a3aa7"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def _end_label(ax, t, y, color, text):
    """Direct label: colored mark carries identity, text stays in ink."""
    ax.plot(t[-1], y[-1], marker="o", markersize=5, color=color, zorder=5)
    ax.annotate(
        text,
        xy=(t[-1], y[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK_SECONDARY,
    )


def plot_dx(t, dx, path="dx_error.png", duration=None):
    """t: (N,) seconds. dx: (N, 3) target-minus-site position error [m]."""
    t = np.asarray(t)
    dx = np.asarray(dx)
    norm = np.linalg.norm(dx, axis=1)

    fig, (ax_c, ax_n) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True, height_ratios=[1.4, 1],
        constrained_layout=True,
    )
    fig.patch.set_facecolor(SURFACE)

    for i, (axis_name, color) in enumerate(SERIES.items()):
        ax_c.plot(t, dx[:, i], color=color, linewidth=2, label=f"dx {axis_name}")
        _end_label(ax_c, t, dx[:, i], color, axis_name)
    ax_c.axhline(0, color=AXIS, linewidth=1)
    ax_c.set_ylabel("component error [m]", color=INK_SECONDARY, fontsize=10)
    ax_c.set_title(
        "Cartesian tracking error at attachment_site",
        color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=12,
    )
    leg = ax_c.legend(frameon=False, loc="upper right", ncols=3, fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)

    ax_n.plot(t, norm, color=NORM, linewidth=2)
    _end_label(ax_n, t, norm, NORM, f"{norm[-1]:.4f} m")
    ax_n.set_ylabel("‖dx‖ [m]", color=INK_SECONDARY, fontsize=10)
    ax_n.set_xlabel("simulation time [s]", color=INK_SECONDARY, fontsize=10)
    ax_n.set_title(
        "Error magnitude", color=INK_SECONDARY, fontsize=10, loc="left", pad=8
    )

    for ax in (ax_c, ax_n):
        _style(ax)
        if duration is not None and t[-1] > duration:
            ax.axvline(duration, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    if duration is not None and t[-1] > duration:
        ax_c.annotate(
            "trajectory reaches goal",
            xy=(duration, ax_c.get_ylim()[1]),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=9,
            color=INK_MUTED,
        )

    # Headroom so the right-edge direct labels are not clipped.
    ax_n.set_xlim(t[0], t[-1] + 0.06 * (t[-1] - t[0]))

    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path
