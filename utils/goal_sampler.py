"""Sample end-effector goal poses that are guaranteed reachable.

Rather than sampling points on a sphere and hoping an IK solution exists, we
sample joint configurations inside the joint limits, run forward kinematics, and
keep the resulting site pose if it lands in the workspace shell we care about.
Every goal is therefore reachable by construction, orientation included.
"""

import mujoco
import numpy as np

# Workspace shell, measured from the base origin. The iiwa14 attachment_site
# spans 0.08 m (fully folded) to 1.31 m (fully stretched); this shell keeps the
# arm well inside that range and away from the fully-extended singularity.
R_MIN: float = 0.45
R_MAX: float = 0.75

# World-frame height band, and minimum horizontal distance from the base column
# so goals do not sit on top of the robot itself.
Z_MIN: float = 0.25
Z_MAX: float = 0.90
XY_MIN: float = 0.20

# Fraction of each joint's range to sample from, centred on the range midpoint.
# Staying off the hard stops keeps the sampled configurations well-conditioned.
JOINT_RANGE_FRACTION: float = 0.85


def sample_positions_around(center, n, r_min=0.05, r_max=0.20, seed=None):
    """Return (n, 3) positions in a spherical shell around `center`.

    Radii are drawn uniformly by volume rather than uniformly in r, so the
    samples are not bunched toward the inner surface.
    """
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    u = rng.uniform(0.0, 1.0, n)
    radii = np.cbrt(r_min**3 + u * (r_max**3 - r_min**3))
    return np.asarray(center) + directions * radii[:, None]


def sample_goal_poses(
    model,
    n,
    site_name="attachment_site",
    seed=None,
    r_min=R_MIN,
    r_max=R_MAX,
    z_min=Z_MIN,
    z_max=Z_MAX,
    xy_min=XY_MIN,
    max_draws=200_000,
):
    """Return (positions (n, 3), quats (n, 4) as (w, x, y, z)).

    Raises RuntimeError if the shell is too tight to fill in `max_draws` tries.
    """
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    site_id = model.site(site_name).id
    base = model.body("base").pos

    nq = model.nq
    lo, hi = model.jnt_range[:nq, 0], model.jnt_range[:nq, 1]
    mid = 0.5 * (lo + hi)
    half = 0.5 * JOINT_RANGE_FRACTION * (hi - lo)

    positions, quats = [], []
    quat = np.zeros(4)
    for _ in range(max_draws):
        if len(positions) == n:
            break
        data.qpos[:nq] = rng.uniform(mid - half, mid + half)
        mujoco.mj_kinematics(model, data)
        pos = data.site(site_id).xpos.copy()

        offset = pos - base
        if not (r_min <= np.linalg.norm(offset) <= r_max):
            continue
        if not (z_min <= pos[2] <= z_max):
            continue
        if np.linalg.norm(offset[:2]) < xy_min:
            continue

        mujoco.mju_mat2Quat(quat, data.site(site_id).xmat)
        positions.append(pos)
        quats.append(quat.copy())
    else:
        raise RuntimeError(
            f"Only found {len(positions)}/{n} goals in {max_draws} draws; "
            "widen the shell (r_min/r_max/z_min/z_max)."
        )

    return np.array(positions), np.array(quats)
