"""Cartesian trajectories with a common `sample(t) -> (pos, quat)` interface."""

import numpy as np
from .dual_quaternions import DualQuaternion


def _s_of_tau(tau):
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5  # zero vel/accel at endpoints


class DualQuaternionTrajectory:
    """Smooth ScLERP trajectory between two poses, min-jerk timed."""

    def __init__(self, pos_start, quat_start, pos_goal, quat_goal, duration):
        # quat_* must be (w, x, y, z)
        self.dq_start = DualQuaternion.from_quat_pose_array(
            np.concatenate([quat_start, pos_start]))
        self.dq_goal = DualQuaternion.from_quat_pose_array(
            np.concatenate([quat_goal, pos_goal]))
        self.duration = duration

    def sample(self, t):
        tau = np.clip(t / self.duration, 0.0, 1.0)
        s = _s_of_tau(tau)
        dq_t = DualQuaternion.sclerp(self.dq_start, self.dq_goal, s)
        pose = dq_t.quat_pose_array()  # [qw, qx, qy, qz, x, y, z]
        quat = np.array(pose[:4])
        pos = np.array(pose[4:])
        return pos, quat


class PositionTrajectory:
    """Straight line in position, min-jerk timed, orientation held fixed.

    ScLERP is deliberately not reused for this case: with identical start and
    goal orientations the screw axis is undefined, and even where it is defined
    the screw path bulges off the straight line between the endpoints. When only
    the position is meant to move, a straight line is what you actually want.
    """

    def __init__(self, pos_start, pos_goal, quat_hold, duration):
        # quat_hold is (w, x, y, z) and comes back unchanged at every sample.
        self.pos_start = np.asarray(pos_start, dtype=float)
        self.pos_goal = np.asarray(pos_goal, dtype=float)
        self.quat_hold = np.asarray(quat_hold, dtype=float)
        self.duration = duration

    def sample(self, t):
        tau = np.clip(t / self.duration, 0.0, 1.0)
        pos = self.pos_start + _s_of_tau(tau) * (self.pos_goal - self.pos_start)
        return pos, self.quat_hold