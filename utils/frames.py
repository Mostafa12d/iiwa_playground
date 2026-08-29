"""Track a site's pose relative to the frame it occupied at a reference instant.

The cabinet knob starts somewhere in world coordinates and then swings on the
door hinge. What you usually want to log is not its absolute world pose but how
far it has moved *from where it started*, expressed in its own initial frame -
so "x" means "along the knob's original x-axis", independent of where the
cabinet happens to sit in the world.
"""

import mujoco
import numpy as np


class FrameTracker:
    """Pose of a site relative to its pose when the tracker was constructed.

    Call `capture()` again to re-zero the reference frame mid-run.
    """

    def __init__(self, model, data, site_name):
        self.model = model
        self.site_id = model.site(site_name).id
        self.site_name = site_name
        self.capture(data)

    def capture(self, data):
        """Freeze the current site pose as the reference frame."""
        self.ref_pos = data.site(self.site_id).xpos.copy()
        self.ref_mat = data.site(self.site_id).xmat.reshape(3, 3).copy()
        self.ref_quat = np.zeros(4)
        mujoco.mju_mat2Quat(self.ref_quat, self.ref_mat.flatten())
        return self.ref_pos, self.ref_quat

    def world(self, data):
        """Current site pose in world coordinates, as (pos, quat)."""
        pos = data.site(self.site_id).xpos.copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site(self.site_id).xmat)
        return pos, quat

    def relative(self, data):
        """Current pose expressed in the reference frame, as (pos, quat).

        pos = R_ref^T (p_now - p_ref), quat = quat(R_ref^T R_now). Both are zero
        / identity at the instant the reference was captured.
        """
        pos = data.site(self.site_id).xpos
        mat = data.site(self.site_id).xmat.reshape(3, 3)
        rel_pos = self.ref_mat.T @ (pos - self.ref_pos)
        rel_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rel_quat, (self.ref_mat.T @ mat).flatten())
        return rel_pos, rel_quat


def quat_angle(quat):
    """Rotation magnitude of a unit quaternion, in radians, on [0, pi]."""
    w = np.clip(abs(np.asarray(quat)[..., 0]), 0.0, 1.0)
    return 2.0 * np.arccos(w)
