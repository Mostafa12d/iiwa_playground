"""Draw the commanded trajectory and the executed path into the MuJoCo viewer.

Everything is pushed into `viewer.user_scn`, which the passive viewer renders on
top of the model without touching the physics.
"""

import mujoco
import numpy as np

# Same categorical slots as the plots, so the viewer and the figures agree.
CMD_RGBA = np.array([0.165, 0.471, 0.839, 1.0], dtype=np.float32)  # blue
ACTUAL_RGBA = np.array([0.922, 0.408, 0.204, 1.0], dtype=np.float32)  # orange
GOAL_RGBA = np.array([0.106, 0.686, 0.478, 0.65], dtype=np.float32)  # aqua
GOAL_ACTIVE_RGBA = np.array([0.106, 0.686, 0.478, 1.0], dtype=np.float32)

_IDENTITY = np.eye(3).flatten()

# RViz axis convention: x red, y green, z blue.
_AXIS_RGBA = (
    np.array([0.90, 0.15, 0.15, 1.0], dtype=np.float32),
    np.array([0.15, 0.75, 0.20, 1.0], dtype=np.float32),
    np.array([0.20, 0.45, 0.95, 1.0], dtype=np.float32),
)


def draw_frame(scene, pos, mat, length=0.16, width=0.009, label=None):
    """Draw an RViz-style RGB axis triad (x red, y green, z blue) at a pose.

    scene: an mjvScene (viewer.user_scn or a renderer's scene). pos: (3,) world
    position. mat: (3, 3) or (9,) world rotation whose columns are the axes.
    Appends to scene.geoms; call after whatever clears scene.ngeom.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    for axis in range(3):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
            _IDENTITY, _AXIS_RGBA[axis],
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, width, pos, pos + length * mat[:, axis]
        )
        scene.ngeom += 1
    if label is not None and scene.ngeom < scene.maxgeom:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([width * 1.5, 0.0, 0.0]),
            pos, _IDENTITY, np.array([0.1, 0.1, 0.1, 1.0], dtype=np.float32),
        )
        geom.label = label
        scene.ngeom += 1


def hide_body_geoms(model, body_name):
    """Hide a body's geoms while leaving its sites (and their frames) visible.

    MuJoCo skips fully transparent geoms when building the scene. Sites are left
    alone deliberately: the mocap target's site is what carries the moving
    mjFRAME_SITE frame. Alpha rather than site/geom groups, since the target and
    attachment_site share group 1.
    """
    body_id = model.body(body_name).id
    start = model.body_geomadr[body_id]
    for geom_id in range(start, start + model.body_geomnum[body_id]):
        model.geom_rgba[geom_id, 3] = 0.0


class TrajectoryOverlay:
    """Commanded polyline + executed trace + goal markers."""

    def __init__(self, user_scn, n_path=60, trace_max=400):
        self.scn = user_scn
        self.n_path = n_path
        self.trace_max = trace_max

        self.cmd_points = np.zeros((0, 3))
        self.goals = np.zeros((0, 3))
        self.active_goal = -1

        self._trace = []
        self._stride = 1  # grows as the trace is progressively decimated
        self._since_kept = 0

    # -- content ---------------------------------------------------------

    def set_commanded(self, traj):
        """Resample a DualQuaternionTrajectory into a drawable polyline."""
        ts = np.linspace(0.0, traj.duration, self.n_path)
        self.cmd_points = np.array([traj.sample(t)[0] for t in ts])

    def set_goals(self, positions):
        self.goals = np.asarray(positions).reshape(-1, 3)

    def append_actual(self, pos):
        """Record an executed point. Call every step; decimation is internal."""
        self._since_kept += 1
        if self._since_kept < self._stride:
            return False
        self._since_kept = 0
        self._trace.append(np.asarray(pos).copy())
        if len(self._trace) > self.trace_max:
            # Halve the resolution of the whole trace rather than forgetting
            # the start of the run.
            self._trace = self._trace[::2]
            self._stride *= 2
        return True

    def clear_trace(self):
        self._trace = []
        self._stride = 1
        self._since_kept = 0

    @property
    def trace(self):
        return np.array(self._trace) if self._trace else np.zeros((0, 3))

    # -- rendering -------------------------------------------------------

    def _next_geom(self):
        if self.scn.ngeom >= self.scn.maxgeom:
            return None
        geom = self.scn.geoms[self.scn.ngeom]
        self.scn.ngeom += 1
        return geom

    def _sphere(self, pos, radius, rgba):
        geom = self._next_geom()
        if geom is None:
            return
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0]),
            np.asarray(pos, dtype=np.float64),
            _IDENTITY,
            rgba,
        )

    def _polyline(self, points, width, rgba):
        for a, b in zip(points[:-1], points[1:]):
            geom = self._next_geom()
            if geom is None:
                return
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                _IDENTITY,
                rgba,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                width,
                np.asarray(a, dtype=np.float64),
                np.asarray(b, dtype=np.float64),
            )

    def redraw(self):
        """Rebuild the overlay. Cheap enough to call at viewer refresh rate."""
        self.scn.ngeom = 0

        for i, goal in enumerate(self.goals):
            active = i == self.active_goal
            self._sphere(
                goal,
                0.022 if active else 0.015,
                GOAL_ACTIVE_RGBA if active else GOAL_RGBA,
            )

        if len(self.cmd_points) > 1:
            self._polyline(self.cmd_points, 0.004, CMD_RGBA)
        if len(self._trace) > 1:
            self._polyline(self._trace, 0.006, ACTUAL_RGBA)
