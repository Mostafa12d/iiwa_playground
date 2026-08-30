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

    Free end-effector only. Welded to an articulated object, the reachable set
    is the constraint manifold, not a shell - use `poses_along_hinge_arc`.
    """
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    u = rng.uniform(0.0, 1.0, n)
    radii = np.cbrt(r_min**3 + u * (r_max**3 - r_min**3))
    return np.asarray(center) + directions * radii[:, None]


def sample_feasible_targets(survey_npz, n, seed=None):
    """Draw n EE goal poses from the feasible set of a reachability survey.

    Selects without replacement from the targets a reachability_survey marked
    feasible (reachable and collision-free). Every goal keeps the survey's grasp
    orientation, since that survey holds orientation fixed across all targets.

    :param survey_npz: path to reachability.npz written by reachability_survey
    :param n: number of goals to draw
    :param seed: seed for the draw
    :returns: (positions (n, 3), quats (n, 4) as (w, x, y, z))
    """
    data = np.load(survey_npz)
    idx = np.flatnonzero(data["feasible"])
    if n > len(idx):
        raise ValueError(
            f"asked for {n} feasible targets but the survey has {len(idx)}"
        )
    chosen = np.random.default_rng(seed).choice(idx, size=n, replace=False)
    positions = data["targets"][chosen]
    quats = np.tile(data["grasp_quat"], (n, 1))
    return positions, quats


def poses_along_hinge_arc(
    model,
    data,
    angles,
    hinge_name,
    driven_site,
    tool_site="attachment_site",
    check_range=True,
):
    """Tool poses on the arc a hinge sweeps out, as (positions, quats).

    `data` fixes the reference configuration; the rigid offset between
    `driven_site` (on the hinged body) and `tool_site` (on the arm) is read off
    it and carried along the arc:

        T_tool(theta) = T_driven(theta) . T_driven(ref)^-1 . T_tool(ref)

    Taking the offset from the sites rather than assuming they coincide is what
    keeps the orientation right - on a grasp the tool z-axis points into the
    surface while the surface normal points out, so the frames sit half a turn
    apart. FK runs on a scratch MjData; `data` is untouched.

    :param angles: hinge angles in radians, one goal pose per angle
    :returns: (positions (n, 3), quats (n, 4) as (w, x, y, z))
    """
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    joint_id = model.joint(hinge_name).id
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError(f"joint '{hinge_name}' is not a hinge")
    if check_range and model.jnt_limited[joint_id]:
        lo, hi = model.jnt_range[joint_id]
        outside = (angles < lo) | (angles > hi)
        if outside.any():
            raise ValueError(
                f"angles {angles[outside]} fall outside the range of "
                f"'{hinge_name}' ([{lo:.4f}, {hi:.4f}] rad)"
            )
    qadr = model.jnt_qposadr[joint_id]

    driven_id = model.site(driven_site).id
    tool_id = model.site(tool_site).id

    def pose_of(d, site_id):
        return (
            d.site(site_id).xpos.copy(),
            d.site(site_id).xmat.reshape(3, 3).copy(),
        )

    pos_driven_ref, mat_driven_ref = pose_of(data, driven_id)
    pos_tool_ref, mat_tool_ref = pose_of(data, tool_id)
    # Tool pose expressed in the driven site's frame. Constant under the weld.
    mat_offset = mat_driven_ref.T @ mat_tool_ref
    pos_offset = mat_driven_ref.T @ (pos_tool_ref - pos_driven_ref)

    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos

    positions = np.empty((len(angles), 3))
    quats = np.empty((len(angles), 4))
    for i, theta in enumerate(angles):
        scratch.qpos[qadr] = theta
        mujoco.mj_kinematics(model, scratch)
        pos_driven, mat_driven = pose_of(scratch, driven_id)
        positions[i] = pos_driven + mat_driven @ pos_offset
        mujoco.mju_mat2Quat(
            quats[i], np.ascontiguousarray(mat_driven @ mat_offset).flatten()
        )
    return positions, quats


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
