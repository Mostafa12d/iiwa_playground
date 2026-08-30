"""Which Cartesian EE targets around an object's handle can the arm reach?

    python3 reachability_survey.py                 # kitchen-cabinet
    python3 reachability_survey.py toolbox

The interaction point (the handle) is known. This does NOT constrain motion to
the object's joint axis: it samples EE target positions in a shell around the
handle, solves IK to each with the grasp orientation held, and collision-checks
the resulting arm configuration. The output is the set of feasible targets - the
different Cartesian directions the arm could push or pull the handle toward,
including ones not aligned with the joint.

Feasibility is split so the two failure modes stay visible:
  reachable      - IK converged within joint limits
  collision-free - no contacts at the solved configuration
A target is feasible only if both hold.
"""

import argparse
import sys

import mujoco
import numpy as np

sys.path.append("scenes")
from build_scene import solve_ik  # noqa: E402

from utils.dx_plot import output_dir  # noqa: E402
from utils.goal_sampler import sample_positions_around  # noqa: E402

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


def survey(object_name=object_name):
    scene_path = f"scenes/{object_name}_iiwa_scene.xml"
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    site_id = model.site("attachment_site").id
    arm_dofs = np.array([model.joint(f"joint{i}").id for i in range(1, 8)])

    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    mujoco.mj_forward(model, data)
    handle_pos = data.site(f"{PREFIX}handle").xpos.copy()
    grasp_q = data.qpos[arm_dofs].copy()
    grasp_quat = np.zeros(4)
    mujoco.mju_mat2Quat(grasp_quat, data.site(site_id).xmat)

    targets = sample_positions_around(
        handle_pos, n_targets, r_min, r_max, seed=sample_seed
    )

    reachable = np.zeros(n_targets, dtype=bool)
    collision_free = np.zeros(n_targets, dtype=bool)
    residuals = np.zeros(n_targets)
    configs = np.zeros((n_targets, len(arm_dofs)))

    # Collisions are off in the scene (the weld handles the coupling); turn them
    # on just for the static feasibility check.
    collide = mujoco.MjData(model)
    model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

    for i, target in enumerate(targets):
        residual, q = solve_ik(
            model, data, arm_dofs, target, grasp_quat, grasp_q
        )
        residuals[i] = residual
        configs[i] = q
        if residual >= ik_tol:
            continue
        reachable[i] = True
        collide.qpos[:] = 0
        collide.qpos[arm_dofs] = q
        mujoco.mj_forward(model, collide)
        collision_free[i] = not _arm_collision(model, collide)

    feasible = reachable & collision_free
    _render(object_name, scene_path, targets, reachable, feasible, handle_pos)

    out = output_dir(object_name)
    np.savez(
        f"{out}/reachability.npz",
        targets=targets,
        reachable=reachable,
        collision_free=collision_free,
        feasible=feasible,
        residuals=residuals,
        configs=configs,
        handle_pos=handle_pos,
        grasp_q=grasp_q,
        grasp_quat=grasp_quat,
    )
    print(
        f"{object_name}: {n_targets} targets in shell "
        f"[{r_min:.2f}, {r_max:.2f}] m around the handle\n"
        f"  reachable      = {reachable.sum()}\n"
        f"  collision-free = {collision_free.sum()}\n"
        f"  feasible       = {feasible.sum()}\n"
        f"  wrote {out}/reachability.png and reachability.npz"
    )


def _render(object_name, scene_path, targets, reachable, feasible, handle_pos):
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, *img_wh)
    renderer.update_scene(data, camera="scene_view")
    scene = renderer.scene
    _sphere(scene, handle_pos, marker_r * 1.6, HANDLE_RGBA)
    for target, r, f in zip(targets, reachable, feasible):
        rgba = FEASIBLE_RGBA if f else (COLLIDE_RGBA if r else UNREACH_RGBA)
        _sphere(scene, target, marker_r, rgba)
    img = renderer.render()
    renderer.close()

    from PIL import Image

    out = output_dir(object_name)
    Image.fromarray(img).save(f"{out}/reachability.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("object", nargs="?", default=object_name,
                        help="object directory under data/")
    survey(parser.parse_args().object)
