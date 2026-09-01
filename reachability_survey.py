"""Which Cartesian EE targets around an object's handle can the arm reach?

    python3 reachability_survey.py                 # kitchen-cabinet
    python3 reachability_survey.py toolbox

The interaction point (the handle) is known. This does NOT constrain motion to
the object's joint axis: it samples EE target positions in a shell around the
handle, solves IK to each with the grasp orientation held, and collision-checks
the resulting arm configuration. The output is the set of feasible targets - the
different Cartesian directions the arm could push or pull the handle toward,
including ones not aligned with the joint.

Runs at one or more hinge angles: the shell follows the handle, and the collision
check sees the lid where it actually is.

Feasible = reachable (IK converged) & collision-free & conditioned (Jacobian
sigma_min above threshold). The last is not implied by the first.
"""

import argparse
import sys

import mujoco
import numpy as np

sys.path.append("scenes")
from build_scene import solve_ik  # noqa: E402

from utils.dx_plot import output_dir  # noqa: E402
from utils.goal_sampler import (  # noqa: E402
    poses_along_hinge_arc,
    sample_positions_around,
)

# Object directory under data/. Its scene must already be built:
#   python3 scenes/build_scene.py <object_name>
object_name = "kitchen-cabinet"
PREFIX = "obj_"

# Shell of EE targets around the handle, and how many to sample.
n_targets: int = 250
r_min: float = 0.05
r_max: float = 0.30
sample_seed: int | None = 0

ik_tol: float = 1e-4  # residual below this counts as reached
pen_tol: float = 1e-3  # contact penetration deeper than this counts as a hit

# Hinge angles to survey, as fractions of the joint's travel.
hinge_fractions = (0.0, 0.25, 0.5, 0.75)

# Smallest singular value of the arm Jacobian below which a configuration counts
# as too close to singular. Picked from the measured spread over the three
# subset objects: the bulk sits at 0.08-0.27, with a thin tail below. Run with
# --report to re-check on a new object.
sigma_min: float = 0.08

# Render size and marker radius.
img_wh = (960, 1280)
marker_r: float = 0.012

# Green feasible, orange reachable-but-colliding, faded gray unreachable.
FEASIBLE_RGBA = np.array([0.106, 0.686, 0.478, 0.95], dtype=np.float32)
COLLIDE_RGBA = np.array([0.922, 0.408, 0.204, 0.9], dtype=np.float32)
UNREACH_RGBA = np.array([0.55, 0.55, 0.55, 0.35], dtype=np.float32)
HANDLE_RGBA = np.array([0.85, 0.2, 0.2, 1.0], dtype=np.float32)
_IDENTITY = np.eye(3).flatten()


def _arm_collision(model, data):
    """True if the arm penetrates the object, the floor, or itself.

    The object's own base/lid meshes overlap by construction, so contacts where
    both geoms are on the object are ignored; only contacts touching an arm geom
    count, and only past `pen_tol` so grazing and the grasp touch do not.
    """
    for i in range(data.ncon):
        con = data.contact[i]
        b1 = model.body(model.geom_bodyid[con.geom1]).name
        b2 = model.body(model.geom_bodyid[con.geom2]).name
        both_object = b1.startswith(PREFIX) and b2.startswith(PREFIX)
        if both_object or con.dist >= -pen_tol:
            continue
        return True
    return False


def _sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(pos, dtype=np.float64),
        _IDENTITY,
        rgba,
    )
    scene.ngeom += 1


def _survey_one(model, data, collide, arm_dofs, theta, hinge_qadr, grasp_q):
    """Sample and score a shell of targets with the hinge held at `theta`."""
    site_id = model.site("attachment_site").id
    lo = model.jnt_range[arm_dofs, 0]
    hi = model.jnt_range[arm_dofs, 1]
    jac = np.zeros((6, model.nv))

    # Handle position and the tool orientation the weld would impose, at theta.
    data.qpos[hinge_qadr] = theta
    mujoco.mj_kinematics(model, data)
    handle_pos = data.site(f"{PREFIX}handle").xpos.copy()
    _, quats = poses_along_hinge_arc(
        model, data, [theta], f"{PREFIX}lid_hinge", f"{PREFIX}handle"
    )
    target_quat = quats[0]

    targets = sample_positions_around(
        handle_pos, n_targets, r_min, r_max, seed=sample_seed
    )

    reachable = np.zeros(n_targets, dtype=bool)
    collision_free = np.zeros(n_targets, dtype=bool)
    conditioned = np.zeros(n_targets, dtype=bool)
    residuals = np.zeros(n_targets)
    sigmas = np.zeros(n_targets)
    margins = np.zeros(n_targets)
    configs = np.zeros((n_targets, len(arm_dofs)))

    for i, target in enumerate(targets):
        # solve_ik zeroes qpos and drives the arm only; the hinge does not enter
        # the arm's kinematics, so theta matters for collision, not for the IK.
        residual, q = solve_ik(model, data, arm_dofs, target, target_quat, grasp_q)
        residuals[i] = residual
        configs[i] = q
        if residual >= ik_tol:
            continue
        reachable[i] = True

        collide.qpos[:] = 0
        collide.qpos[hinge_qadr] = theta
        collide.qpos[arm_dofs] = q
        mujoco.mj_forward(model, collide)
        collision_free[i] = not _arm_collision(model, collide)

        mujoco.mj_jacSite(model, collide, jac[:3], jac[3:], site_id)
        sigmas[i] = np.linalg.svd(jac[:, arm_dofs], compute_uv=False)[-1]
        margins[i] = float(np.minimum(q - lo, hi - q).min())
        conditioned[i] = sigmas[i] >= sigma_min

    return dict(
        targets=targets, reachable=reachable, collision_free=collision_free,
        conditioned=conditioned, residuals=residuals, sigmas=sigmas,
        margins=margins, configs=configs, handle_pos=handle_pos,
        target_quat=target_quat, theta=theta,
    )


def survey(object_name=object_name, report=False):
    scene_path = f"scenes/{object_name}_iiwa_scene.xml"
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    site_id = model.site("attachment_site").id
    arm_dofs = np.array([model.joint(f"joint{i}").id for i in range(1, 8)])
    hinge_id = model.joint(f"{PREFIX}lid_hinge").id
    hinge_qadr = model.jnt_qposadr[hinge_id]
    hinge_hi = float(model.jnt_range[hinge_id, 1])

    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    mujoco.mj_forward(model, data)
    grasp_q = data.qpos[arm_dofs].copy()
    grasp_quat = np.zeros(4)
    mujoco.mju_mat2Quat(grasp_quat, data.site(site_id).xmat)

    # Collisions are off in the scene (the weld handles the coupling); turn them
    # on just for the static feasibility check.
    collide = mujoco.MjData(model)
    model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

    runs = []
    for frac in hinge_fractions:
        theta = frac * hinge_hi
        r = _survey_one(model, data, collide, arm_dofs, theta,
                        hinge_qadr, grasp_q)
        r["fraction"] = frac
        runs.append(r)
        _render(object_name, scene_path, r, theta, hinge_qadr,
                suffix=f"_{int(frac * 100):03d}")

    stack = {k: np.concatenate([r[k] for r in runs]) for k in
             ("targets", "reachable", "collision_free", "conditioned",
              "residuals", "sigmas", "margins", "configs")}
    stack["hinge_angle"] = np.concatenate(
        [np.full(n_targets, r["theta"]) for r in runs]
    )
    stack["feasible"] = (stack["reachable"] & stack["collision_free"]
                         & stack["conditioned"])

    out = output_dir(object_name)
    np.savez(
        f"{out}/reachability.npz",
        handle_pos=np.array([r["handle_pos"] for r in runs]),
        target_quats=np.array([r["target_quat"] for r in runs]),
        hinge_fractions=np.array(hinge_fractions),
        grasp_q=grasp_q,
        grasp_quat=grasp_quat,
        sigma_min=sigma_min,
        **stack,
    )

    print(f"{object_name}: {n_targets} targets x {len(runs)} hinge angles, "
          f"shell [{r_min:.2f}, {r_max:.2f}] m")
    print(f"  {'theta':>6} {'frac':>5} {'reach':>6} {'nocol':>6} "
          f"{'cond':>6} {'feasible':>9}")
    for r in runs:
        f = r["reachable"] & r["collision_free"] & r["conditioned"]
        print(f"  {r['theta']:6.3f} {r['fraction']:5.2f} "
              f"{r['reachable'].sum():6d} {r['collision_free'].sum():6d} "
              f"{r['conditioned'].sum():6d} {f.sum():9d}")
    print(f"  total feasible = {stack['feasible'].sum()} "
          f"of {len(stack['feasible'])}")

    if report:
        s = stack["sigmas"][stack["reachable"] & stack["collision_free"]]
        qs = [0, 1, 5, 10, 25, 50, 75, 100]
        print(f"  sigma_min spread over reachable & collision-free (n={len(s)}):")
        print("    " + "  ".join(f"p{q}={np.percentile(s, q):.4f}" for q in qs))
    print(f"  wrote {out}/reachability.npz")


def _render(object_name, scene_path, run, theta, hinge_qadr, suffix=""):
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    data.qpos[hinge_qadr] = theta
    mujoco.mj_forward(model, data)

    feasible = run["reachable"] & run["collision_free"] & run["conditioned"]
    renderer = mujoco.Renderer(model, *img_wh)
    renderer.update_scene(data, camera="scene_view")
    scene = renderer.scene
    _sphere(scene, run["handle_pos"], marker_r * 1.6, HANDLE_RGBA)
    for target, r, f in zip(run["targets"], run["reachable"], feasible):
        rgba = FEASIBLE_RGBA if f else (COLLIDE_RGBA if r else UNREACH_RGBA)
        _sphere(scene, target, marker_r, rgba)
    img = renderer.render()
    renderer.close()

    from PIL import Image

    out = output_dir(object_name)
    Image.fromarray(img).save(f"{out}/reachability{suffix}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("object", nargs="?", default=object_name,
                        help="object directory under data/")
    parser.add_argument("--report", action="store_true",
                        help="print the sigma_min spread, to pick a threshold")
    args = parser.parse_args()
    survey(args.object, report=args.report)
