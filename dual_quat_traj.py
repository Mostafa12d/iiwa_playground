import numpy as np
from dual_quaternions import DualQuaternion


class DualQuaternionTrajectory:
    """Smooth ScLERP trajectory between two poses, min-jerk timed."""

    def __init__(self, pos_start, quat_start, pos_goal, quat_goal, duration):
        # quat_* must be (w, x, y, z)
        self.dq_start = DualQuaternion.from_quat_pose_array(
            np.concatenate([quat_start, pos_start]))
        self.dq_goal = DualQuaternion.from_quat_pose_array(
            np.concatenate([quat_goal, pos_goal]))
        self.duration = duration

    @staticmethod
    def _s_of_tau(tau):
        return 10 * tau**3 - 15 * tau**4 + 6 * tau**5  # zero vel/accel at endpoints

    def sample(self, t):
        tau = np.clip(t / self.duration, 0.0, 1.0)
        s = self._s_of_tau(tau)
        dq_t = DualQuaternion.sclerp(self.dq_start, self.dq_goal, s)
        pose = dq_t.quat_pose_array()  # [qw, qx, qy, qz, x, y, z]
        quat = np.array(pose[:4])
        pos = np.array(pose[4:])
        return pos, quat