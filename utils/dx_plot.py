"""Plots for an impedance-control run: tracking error and commanded vs executed path."""

import os

import matplotlib

matplotlib.use("Agg")  # headless: mjpython owns the GUI loop

import matplotlib.pyplot as plt
import numpy as np

# Every run writes its figures and logs under plots/<run name>/.
PLOTS_ROOT = "plots"


def output_dir(name):
    """Create and return plots/<name>/, this run's output directory."""
    path = os.path.join(PLOTS_ROOT, name)
    os.makedirs(path, exist_ok=True)
    return path

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots 1-3 for x/y/z, violet for the derived norm.
AXES = ("x", "y", "z")
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
NORM = "#4a3aa7"
CMD = "#2a78d6"
ACTUAL = "#eb6834"


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


def _legend(ax, **kwargs):
    leg = ax.legend(frameon=False, fontsize=9, **kwargs)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)


def _label_at(ax, t, y, index, color, text, offset=(7, 0)):
    """Direct label: the colored mark carries identity, the text stays in ink."""
    ax.plot(t[index], y[index], marker="o", markersize=5, color=color, zorder=5)
    ax.annotate(
        text,
        xy=(t[index], y[index]),
        xytext=offset,
        textcoords="offset points",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=INK_SECONDARY,
        zorder=6,
    )


def _peak_label(ax, t, y, color, text):
    """Label a series at its own extremum, where series are best separated.

    Labelling at the right edge piles all three components on top of each other
    once the error converges to zero.
    """
    index = int(np.argmax(np.abs(y)))
    above = y[index] >= 0
    _label_at(ax, t, y, index, color, text, offset=(0, 11 if above else -11))


def _segment_marks(axes, segments, label_ax):
    """segments: list of (t_start, label). The first start is not drawn."""
    if not segments:
        return
    for i, (t_start, label) in enumerate(segments):
        for ax in axes:
            if i > 0:
                ax.axvline(
                    t_start, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3))
                )
        label_ax.annotate(
            label,
            xy=(t_start, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -11),
            textcoords="offset points",
            fontsize=8,
            color=INK_MUTED,
        )


def _headroom(ax, t):
    ax.set_xlim(t[0], t[-1] + 0.07 * (t[-1] - t[0]))


def plot_dx(t, dx, path="dx_error.png", segments=None):
    """t: (N,) seconds. dx: (N, 3) commanded-minus-actual position error [m]."""
    t = np.asarray(t)
    dx = np.asarray(dx)
    norm = np.linalg.norm(dx, axis=1)

    fig, (ax_c, ax_n) = plt.subplots(
        2, 1, figsize=(9.5, 6.5), sharex=True, height_ratios=[1.4, 1],
        constrained_layout=True,
    )
    fig.patch.set_facecolor(SURFACE)

    for i, (name, color) in enumerate(zip(AXES, SERIES)):
        ax_c.plot(t, dx[:, i], color=color, linewidth=2, label=f"dx {name}")
        _peak_label(ax_c, t, dx[:, i], color, name)
    ax_c.axhline(0, color=AXIS, linewidth=1)
    ax_c.set_ylabel("component error [m]", color=INK_SECONDARY, fontsize=10)
    ax_c.set_title(
        "Cartesian tracking error at attachment_site",
        color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=16,
    )
    _legend(ax_c, loc="lower right", ncols=3)

    ax_n.plot(t, norm, color=NORM, linewidth=2)
    _label_at(ax_n, t, norm, -1, NORM, f"{norm[-1]:.4f} m")
    ax_n.set_ylabel("‖dx‖ [m]", color=INK_SECONDARY, fontsize=10)
    ax_n.set_xlabel("simulation time [s]", color=INK_SECONDARY, fontsize=10)
    ax_n.set_title(
        "Error magnitude", color=INK_SECONDARY, fontsize=10, loc="left", pad=8
    )

    for ax in (ax_c, ax_n):
        _style(ax)
    _segment_marks((ax_c, ax_n), segments, ax_c)
    _headroom(ax_n, t)

    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_relative_frame(
    t, rel_pos, rel_angle, path="handle_frame.png", segments=None, name="cab_handle"
):
    """Motion of a tracked frame relative to where it started.

    Position and rotation are separate panels because they are different units -
    they must never share a y-axis.
    """
    t = np.asarray(t)
    rel_pos = np.asarray(rel_pos)
    rel_angle = np.asarray(rel_angle)

    fig, (ax_p, ax_r) = plt.subplots(
        2, 1, figsize=(9.5, 6.5), sharex=True, height_ratios=[1.4, 1],
        constrained_layout=True,
    )
    fig.patch.set_facecolor(SURFACE)

    for i, (axis_name, color) in enumerate(zip(AXES, SERIES)):
        ax_p.plot(t, rel_pos[:, i], color=color, linewidth=2, label=axis_name)
        _peak_label(ax_p, t, rel_pos[:, i], color, axis_name)
    ax_p.axhline(0, color=AXIS, linewidth=1)
    ax_p.set_ylabel("displacement [m]", color=INK_SECONDARY, fontsize=10)
    ax_p.set_title(
        f"{name} relative to its starting frame",
        color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=16,
    )
    # Upper right: a tracked frame starts at zero and moves away, so the
    # bottom-right corner is exactly where the curves converge.
    _legend(ax_p, loc="upper right", ncols=3)

    ax_r.plot(t, rel_angle, color=NORM, linewidth=2)
    _label_at(ax_r, t, rel_angle, -1, NORM, f"{rel_angle[-1]:.3f} rad")
    ax_r.set_ylabel("rotation [rad]", color=INK_SECONDARY, fontsize=10)
    ax_r.set_xlabel("simulation time [s]", color=INK_SECONDARY, fontsize=10)
    ax_r.set_title(
        "Rotation magnitude from the starting frame",
        color=INK_SECONDARY, fontsize=10, loc="left", pad=8,
    )

    for ax in (ax_p, ax_r):
        _style(ax)
    _segment_marks((ax_p, ax_r), segments, ax_p)
    _headroom(ax_r, t)

    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_door_angle(t, angle, path="door_angle.png", segments=None, limit=None,
                    name="lid"):
    """Articulated lid angle over the run. Single series, so no legend box."""
    t = np.asarray(t)
    angle = np.asarray(angle)

    fig, ax = plt.subplots(figsize=(9.5, 3.6), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)

    ax.plot(t, angle, color=NORM, linewidth=2)
    _label_at(ax, t, angle, -1, NORM, f"{angle[-1]:.3f} rad")
    if limit is not None:
        if angle.max() > 0.35 * limit:
            # Only draw the ceiling when the door gets near it; otherwise it
            # rescales the axis and squashes the motion into a flat line.
            ax.axhline(limit, color=INK_MUTED, linewidth=1, linestyle=(0, (2, 3)))
            ax.annotate(
                "hinge limit", xy=(t[-1], limit), xytext=(-4, 5), ha="right",
                textcoords="offset points", fontsize=8, color=INK_MUTED,
            )
        else:
            ax.annotate(
                f"hinge limit {limit:.2f} rad, off-scale",
                xy=(1.0, 1.0), xycoords="axes fraction", xytext=(-4, -12),
                textcoords="offset points", ha="right", fontsize=8,
                color=INK_MUTED,
            )
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_ylabel("hinge angle [rad]", color=INK_SECONDARY, fontsize=10)
    ax.set_xlabel("simulation time [s]", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        f"{name} hinge angle",
        color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=16,
    )

    _style(ax)
    _segment_marks((ax,), segments, ax)
    _headroom(ax, t)

    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_paths(t, cmd, actual, path="dx_paths.png", segments=None):
    """Commanded vs executed site position, one panel per world axis."""
    t = np.asarray(t)
    cmd = np.asarray(cmd)
    actual = np.asarray(actual)

    fig, axs = plt.subplots(
        3, 1, figsize=(9.5, 7.5), sharex=True, constrained_layout=True
    )
    fig.patch.set_facecolor(SURFACE)

    for i, (ax, name) in enumerate(zip(axs, AXES)):
        ax.plot(t, cmd[:, i], color=CMD, linewidth=2, label="commanded")
        ax.plot(t, actual[:, i], color=ACTUAL, linewidth=2, label="executed")
        ax.set_ylabel(f"{name} [m]", color=INK_SECONDARY, fontsize=10)
        _style(ax)

    axs[0].set_title(
        "Commanded trajectory vs executed end-effector path",
        color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=16,
    )
    _legend(axs[0], loc="lower right", ncols=2)
    axs[-1].set_xlabel("simulation time [s]", color=INK_SECONDARY, fontsize=10)

    _segment_marks(axs, segments, axs[0])
    _headroom(axs[-1], t)

    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path
