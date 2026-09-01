"""Build an iiwa + articulated-object scene for any object in data/.

    python3 scenes/build_scene.py kitchen-cabinet
    python3 scenes/build_scene.py --all
    python3 scenes/build_scene.py --list

Every object in data/ has the same skeleton - a `base` body, a `lid` body, one
`lid_hinge` revolute joint, a root free joint and a lid position servo - so the
structural half of this is deterministic. For each object the builder:

  1. detects the handle on the lid,
  2. checks it against three sanity gates and refuses to emit a scene if any
     fail, rather than quietly producing a bad grasp,
  3. writes data/<name>/object_static.xml (free joint and lid servo removed, a
     "handle" site added),
  4. searches arm standoff and tool roll for the best-conditioned grasp IK,
  5. writes scenes/<name>_iiwa_scene.xml with the grasp keyframe and the weld.

Objects known to trip a gate today: locker (lid is a single collision hull, so
there is no protrusion to find), medicine-cabinet and toilet (the grasp lands on
a far lid corner rather than a feature), microwave (the grasp lands off the lid
geometry entirely). Give those an entry in HANDLE_OVERRIDES to build them.
"""

import argparse
import glob
import os

import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM = "kuka_iiwa_14/iiwa14_torque.xml"
PREFIX = "obj_"  # the object is namespaced; the arm keeps its own names

# --- handle detection -------------------------------------------------------

# A lid piece counts as "proud" if it reaches past this percentile of all lid
# pieces along the outward normal, i.e. it stands off the panel face.
PROUD_PERCENTILE = 90
# The grasp is backed this far into the knob from its outermost face.
GRASP_INSET = 0.012

# Gates. A detected handle must clear all three.
MAX_RADIUS_FRACTION = 0.75  # of the lid's own extent; beyond this it's a corner
MAX_GAP = 0.02  # m from the nearest lid collision geom

# Objects whose handle the heuristic gets wrong, as lid-frame coordinates.
HANDLE_OVERRIDES: dict[str, tuple[float, float, float]] = {}

# --- grasp search -----------------------------------------------------------

STANDOFFS = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)  # arm base distance from handle
ROLLS = 6  # rotations about the tool axis to try
# Tilt of the approach axis away from the lid's outward normal, toward the
# horizontal. A side-hinged door wants 0 (straight in through the face); a
# top-opening lid is far easier to reach at an angle than from directly above.
TILTS = (0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8)
IK_RESTARTS = 12
IK_SEED = 1
# Score penalty per radian of approach tilt off the lid normal.
TILT_PENALTY = 0.6
# Below this, the outward normal is treated as vertical and the arm is placed
# using the hinge-to-handle direction instead.
MIN_HORIZONTAL = 0.3

SIGNS = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)


def geom_corners(model, data, geom_id):
    center, half = model.geom_aabb[geom_id][:3], model.geom_aabb[geom_id][3:]
    rot = data.geom_xmat[geom_id].reshape(3, 3)
    pos = data.geom_xpos[geom_id]
    return np.array([pos + rot @ (center + s * half) for s in SIGNS])


def detect_handle(name):
    """Find the handle on the lid. Returns a dict of geometry + diagnostics.

    The lid swings about `axis`; `outward` is the direction the lid moves away
    from the base, with the axis component removed so only the swing direction
    survives. A handle is a lid piece that stands proud of the panel face along
    `outward` and sits far from the hinge axis - the natural place to grasp,
    both because it protrudes and because it has the longest moment arm.
    """
    path = f"data/{name}/object.xml"
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    lid = model.body("lid").id
    base = model.body("base").id
    hinge = model.joint("lid_hinge").id
    axis, anchor = data.xaxis[hinge].copy(), data.xanchor[hinge].copy()

    outward = data.body(lid).xipos - data.body(base).xipos
    outward = outward - axis * (outward @ axis)
    outward /= np.linalg.norm(outward)

    gids = [
        g for g in range(model.ngeom)
        if model.geom_bodyid[g] == lid and model.geom_group[g] == 3
    ]
    reach, radius = {}, {}
    for g in gids:
        cs = geom_corners(model, data, g)
        reach[g] = float((cs @ outward).max())
        rel = cs - anchor
        radius[g] = float(
            np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1).max()
        )

    proud = [g for g in gids if reach[g] > np.percentile(list(reach.values()),
                                                         PROUD_PERCENTILE)]
    knob = max(proud or gids, key=lambda g: radius[g])
    cs = geom_corners(model, data, knob)
    grasp_world = cs.mean(0) + outward * (
        (cs @ outward).max() - cs.mean(0) @ outward - GRASP_INSET
    )

    lid_rot, lid_pos = data.body(lid).xmat.reshape(3, 3), data.body(lid).xpos
    if name in HANDLE_OVERRIDES:
        grasp_local = np.array(HANDLE_OVERRIDES[name], dtype=float)
        grasp_world = lid_pos + lid_rot @ grasp_local
        problems = []
    else:
        grasp_local = lid_rot.T @ (grasp_world - lid_pos)
        all_corners = np.vstack([geom_corners(model, data, g) for g in gids])
        extent = float(np.linalg.norm(all_corners.max(0) - all_corners.min(0)))
        gap = min(
            float(np.linalg.norm(
                np.clip(grasp_world,
                        geom_corners(model, data, g).min(0),
                        geom_corners(model, data, g).max(0)) - grasp_world))
            for g in gids
        )
        problems = []
        if not proud:
            problems.append("no lid piece stands proud of the panel face")
        if radius[knob] > MAX_RADIUS_FRACTION * extent:
            problems.append(
                f"grasp radius {radius[knob]:.3f} m is "
                f"{radius[knob]/extent:.0%} of the lid extent - looks like a corner"
            )
        if gap > MAX_GAP:
            problems.append(f"grasp sits {gap*100:.0f} cm off any lid geometry")

    return {
        "grasp_world": grasp_world,
        "grasp_local": grasp_local,
        "outward": outward,
        "axis": axis,
        "anchor": anchor,
        "problems": problems,
    }


# --- static object ----------------------------------------------------------

HEADER = """<mujoco model="{name}_static">
  <!-- Generated from object.xml by scenes/build_scene.py. Three changes:
       1. the root free joint is dropped - the object is anchored here, so the
          arm pulling on the lid must not drag the whole body around.
       2. a "handle" site is added to the lid, at the detected grasp point.
       3. the lid position servo is dropped - it would hold the lid shut against
          the welded arm. The hinge keeps its damping and frictionloss.
       Edit object.xml and re-run the builder rather than editing this file. -->"""

LID_GEOM = '        <geom class="visual" type="mesh" mesh="lid_visual"/>\n'


def write_static(name, grasp_local):
    src = open(f"data/{name}/object.xml").read()
    site = (
        f'        <site name="handle" pos="{grasp_local[0]:.5f} '
        f'{grasp_local[1]:.5f} {grasp_local[2]:.5f}" '
        f'size="0.012" rgba="0.85 0.2 0.2 1"/>\n'
    )
    out = src.replace('<mujoco model="asset">', HEADER.format(name=name))

    start = src.find("  <actuator>")
    end = src.find("</actuator>", start)
    if start < 0 or end < 0:
        raise SystemExit(f"{name}: no <actuator> block to remove")
    actuator_block = src[start:end + len("</actuator>") + 1]

    for old, new in (
        ('      <freejoint name="root"/>\n', ""),
        (LID_GEOM, LID_GEOM + site),
        (actuator_block, ""),
    ):
        if old not in out:
            raise SystemExit(f"{name}: object.xml no longer contains:\n{old[:80]}")
        out = out.replace(old, new)

    path = f"data/{name}/object_static.xml"
    open(path, "w").write(out)
    return path


# --- scene ------------------------------------------------------------------

def camera_attrs(target, outward, distance=1.7, height=0.7):
    """A three-quarter view of the grasp, looking back along the outward normal."""
    up = np.array([0.0, 0.0, 1.0])
    offset = outward * distance + up * height
    # swing a little to the side so the arm does not hide the lid
    side = np.cross(up, outward)
    if np.linalg.norm(side) > 1e-6:
        offset += side / np.linalg.norm(side) * distance * 0.45
    pos = target + offset
    forward = target - pos
    forward /= np.linalg.norm(forward)
    z_cam = -forward
    x_cam = np.cross(up, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    return pos, np.concatenate([x_cam, y_cam])


SCENE = """<mujoco model="{name}_iiwa">
  <!-- KUKA iiwa 14 welded to the {name} handle.
       Generated by scenes/build_scene.py - re-run it rather than hand-editing.

       Both models are pulled in with <attach> rather than copied, so the arm
       and the object stay single-sourced. The arm keeps its own names, the
       object is namespaced "{prefix}".

       Grasp configuration and weld: {restarts} random-restart damped-least-squares
       IK runs over {standoffs} arm standoffs x {rolls} tool rolls, placing
       attachment_site on {prefix}handle with the tool axis along the lid's outward
       normal, scored on joint-limit margin and Jacobian conditioning. Winner:
       standoff {standoff:.2f} m, limit margin {margin:.3f} rad, condition number {cond:.1f}. -->

  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast">
    <!-- Contacts off: the weld already couples the arm to the lid, and letting
         the arm's collision geoms hit the object just fights the constraint. -->
    <flag contact="disable"/>
  </option>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="-125" elevation="-20" offwidth="1280" offheight="960"/>
  </visual>

  <asset>
    <model name="iiwa" file="../{arm}"/>
    <model name="object" file="../data/{name}/object_static.xml"/>

    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="{light}" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="groundplane"/>

    <!-- The arm base sits at the world origin, unrotated, so the world frame IS
         the robot base frame - every logged position, velocity and wrench is
         already base-relative with no transform. The object carries the offset
         instead: it is pushed back by the arm standoff along the lid's outward
         normal, which puts the handle exactly where it was before. -->
    <body name="arm_mount" pos="0 0 0">
      <attach model="iiwa" prefix=""/>
    </body>

    <body name="object_mount" pos="{object_pos}">
      <attach model="object" prefix="{prefix}"/>
    </body>

    <camera name="scene_view" pos="{cam_pos}" xyaxes="{cam_axes}"/>
  </worldbody>
{tail}</mujoco>
"""

TAIL = """
  <equality>
    <!-- MuJoCo's weld residual is
             ({prefix}lid.xpos + {prefix}lid.xmat @ anchor)
           - (link7.xpos   + link7.xmat   @ relpose_pos)
         so anchor and relpose_pos must name the SAME physical point in their two
         body frames - the handle. -->
    <!-- Compliant grasp, not a rigid one: a real hand on a handle gives a
         little. The compliance lives in solref's time constant, NOT in solimp.
         Lowering solimp's d_min/d_max makes the constraint permanently leaky
         instead of springy - the arm then slides off the handle rather than
         deflecting against it (measured on this scene at d=0.3: 66 mm of slip
         and divergence at Kp=100). With d left at 0.99/0.999 the weld stays
         enforced; tau=0.1 gives ~3-4 mm of give and is stable across Kp
         50-1600. tau=0.2 gives ~12-17 mm, also stable; tau=0.4 diverges. -->
    <weld name="ee_to_handle" body1="link7" body2="{prefix}lid"
          anchor="{anchor}"
          relpose="{relpose}"
          solref="0.1 1" solimp="0.99 0.999 0.001 0.5 2"/>
  </equality>

  <!-- qpos: joint1..7 then {prefix}lid_hinge. Lid closed, EE on the handle. -->
  <keyframe>
    <key name="grasp" qpos="{qpos}" ctrl="0 0 0 0 0 0 0"/>
  </keyframe>
"""


def render_scene(name, mount, cam_pos, cam_axes, target, tail="", **meta):
    """`mount`, `cam_pos` and `target` are all in object-file coordinates.

    They are shifted by -mount on the way out, which moves the arm base to the
    origin and carries everything else with it.
    """
    return SCENE.format(
        name=name, arm=ARM, prefix=PREFIX,
        object_pos=" ".join(f"{v:.4f}" for v in -mount),
        light=" ".join(f"{v:.2f}" for v in (target - mount + np.array([0, 0, 2.0]))),
        cam_pos=" ".join(f"{v:.4f}" for v in (cam_pos - mount)),
        cam_axes=" ".join(f"{v:.4f}" for v in cam_axes),
        tail=tail, **meta,
    )


# --- grasp IK ---------------------------------------------------------------

def tool_frame(tool_z, roll):
    """Rotation matrix with its z-axis on `tool_z`, rolled about that axis."""
    ref = np.array([0.0, 0.0, 1.0])
    if abs(ref @ tool_z) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    x0 = np.cross(ref, tool_z)
    x0 /= np.linalg.norm(x0)
    y0 = np.cross(tool_z, x0)
    x = np.cos(roll) * x0 + np.sin(roll) * y0
    y = np.cross(tool_z, x)
    return np.column_stack([x, y, tool_z])


def solve_ik(model, data, arm_dofs, target_pos, target_quat, seed_q):
    lo = model.jnt_range[arm_dofs, 0]
    hi = model.jnt_range[arm_dofs, 1]
    jac = np.zeros((6, model.nv))
    err = np.zeros(6)
    site_id = model.site("attachment_site").id
    sq, sqc, eq = np.zeros(4), np.zeros(4), np.zeros(4)

    data.qpos[:] = 0
    data.qpos[arm_dofs] = seed_q
    for it in range(600):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        err[:3] = target_pos - data.site(site_id).xpos
        mujoco.mju_mat2Quat(sq, data.site(site_id).xmat)
        mujoco.mju_negQuat(sqc, sq)
        mujoco.mju_mulQuat(eq, target_quat, sqc)
        mujoco.mju_quat2Vel(err[3:], eq, 1.0)
        residual = np.linalg.norm(err)
        if residual < 1e-7:
            break
        # Unreachable targets otherwise burn the full iteration budget on every
        # restart, and the search runs hundreds of them.
        if it > 150 and residual > 5e-2:
            return float(residual), data.qpos[arm_dofs].copy()
        mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        J = jac[:, arm_dofs]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        nullspace = np.eye(len(arm_dofs)) - np.linalg.pinv(J) @ J
        dq += nullspace @ (0.5 * (lo + hi) - data.qpos[arm_dofs]) * 0.05
        data.qpos[arm_dofs] = np.clip(
            data.qpos[arm_dofs] + 0.5 * dq, lo + 1e-4, hi - 1e-4
        )
    return float(np.linalg.norm(err)), data.qpos[arm_dofs].copy()


def search_grasp(scene_path, tool_axes, standoff):
    """Best IK over approach tilts, tool rolls and restarts, for one standoff.

    The model is loaded once here rather than per candidate - it carries the
    object's collision meshes and loading dominates the search otherwise.
    """
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    arm_dofs = np.array([model.joint(f"joint{i}").id for i in range(1, 8)])
    lo = model.jnt_range[arm_dofs, 0]
    hi = model.jnt_range[arm_dofs, 1]

    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    target_pos = data.site(f"{PREFIX}handle").xpos.copy()

    rng = np.random.default_rng(IK_SEED)
    jac = np.zeros((6, model.nv))
    best = None
    for tilt, tool_z in tool_axes:
        for roll in np.linspace(0, 2 * np.pi, ROLLS, endpoint=False):
            target_quat = np.zeros(4)
            mujoco.mju_mat2Quat(target_quat, tool_frame(tool_z, roll).flatten())
            for _ in range(IK_RESTARTS):
                residual, q = solve_ik(
                    model, data, arm_dofs, target_pos, target_quat,
                    rng.uniform(lo * 0.6, hi * 0.6),
                )
                if residual > 1e-5:
                    continue
                data.qpos[arm_dofs] = q
                mujoco.mj_forward(model, data)
                mujoco.mj_jacSite(model, data, jac[:3], jac[3:],
                                  model.site("attachment_site").id)
                sv = np.linalg.svd(jac[:, arm_dofs], compute_uv=False)
                margin = float(np.minimum(q - lo, hi - q).min())
                # Reward joint-limit headroom and conditioning, but penalise
                # approaching off the lid normal: without this the search
                # happily hooks a door knob sideways because a nearly straight
                # arm scores well on limit margin.
                score = margin + 0.5 * sv[-1] - TILT_PENALTY * tilt
                if best is None or score > best["score"]:
                    best = {"score": score, "margin": margin, "sv": sv,
                            "q": q.copy(), "standoff": standoff,
                            "roll": float(roll), "tilt": float(tilt)}
    if best is None:
        return None

    # Weld numbers, at the winning configuration.
    data.qpos[:] = 0
    data.qpos[arm_dofs] = best["q"]
    mujoco.mj_forward(model, data)
    b1 = model.body("link7").id
    b2 = model.body(f"{PREFIX}lid").id
    R1, p1 = data.body(b1).xmat.reshape(3, 3), data.body(b1).xpos
    R2, p2 = data.body(b2).xmat.reshape(3, 3), data.body(b2).xpos
    handle = data.site(f"{PREFIX}handle").xpos
    relquat = np.zeros(4)
    mujoco.mju_mat2Quat(relquat, (R1.T @ R2).flatten())
    best["anchor"] = R2.T @ (handle - p2)
    best["relpose"] = np.concatenate([R1.T @ (handle - p1), relquat])
    best["qpos"] = np.zeros(model.nq)
    best["qpos"][arm_dofs] = best["q"]
    return best


# --- driver -----------------------------------------------------------------

def build(name, verbose=True):
    handle = detect_handle(name)
    if handle["problems"]:
        raise SystemExit(
            f"{name}: handle detection failed\n  - "
            + "\n  - ".join(handle["problems"])
            + f"\nAdd a lid-frame position to HANDLE_OVERRIDES to build it anyway."
        )

    write_static(name, handle["grasp_local"])
    grasp = handle["grasp_world"]
    outward = handle["outward"]
    scene_path = f"scenes/{name}_iiwa_scene.xml"

    # Where to stand the arm, horizontally. A side-hinged door has a horizontal
    # outward normal and the arm belongs in front of it. A top-opening lid has a
    # vertical normal, which says nothing about where to stand - use the
    # hinge-to-handle direction instead, so the arm works from the free edge.
    horizontal = np.array([outward[0], outward[1], 0.0])
    if np.linalg.norm(horizontal) >= MIN_HORIZONTAL:
        mount_dir = horizontal / np.linalg.norm(horizontal)
    else:
        away = grasp[:2] - handle["anchor"][:2]
        mount_dir = np.array([away[0], away[1], 0.0])
        mount_dir /= np.linalg.norm(mount_dir)

    # Approach axes: the lid's inward normal, tilted toward the arm. Straight
    # down onto a low lid is usually unreachable; at an angle it is not.
    approach = -mount_dir
    tool_axes = []
    for tilt in TILTS:
        axis = np.cos(tilt) * (-outward) + np.sin(tilt) * approach
        tool_axes.append((tilt, axis / np.linalg.norm(axis)))

    best = None
    for standoff in STANDOFFS:
        mount = np.array([grasp[0] + mount_dir[0] * standoff,
                          grasp[1] + mount_dir[1] * standoff, 0.0])
        cam_pos, cam_axes = camera_attrs(grasp, mount_dir)
        open(scene_path, "w").write(
            render_scene(name, mount, cam_pos, cam_axes, grasp,
                         restarts=IK_RESTARTS, standoffs=len(STANDOFFS),
                         rolls=ROLLS, standoff=standoff, margin=0.0, cond=0.0)
        )
        candidate = search_grasp(scene_path, tool_axes, standoff)
        if candidate and (best is None or candidate["score"] > best["score"]):
            best = candidate
            best["mount"] = mount

    if best is None:
        os.remove(scene_path)
        raise SystemExit(
            f"{name}: no IK solution at any standoff in {STANDOFFS}. The handle "
            f"is at z={grasp[2]:.2f} m; try mounting the arm on a pedestal."
        )

    cam_pos, cam_axes = camera_attrs(grasp, mount_dir)
    tail = TAIL.format(
        prefix=PREFIX,
        anchor=" ".join(f"{v:.8f}" for v in best["anchor"]),
        relpose=" ".join(f"{v:.8f}" for v in best["relpose"]),
        qpos=" ".join(f"{v:.6f}" for v in best["qpos"]),
    )
    open(scene_path, "w").write(
        render_scene(name, best["mount"], cam_pos, cam_axes, grasp, tail=tail,
                     restarts=IK_RESTARTS, standoffs=len(STANDOFFS), rolls=ROLLS,
                     standoff=best["standoff"], margin=best["margin"],
                     cond=best["sv"][0] / best["sv"][-1])
    )

    check = verify(scene_path)
    if verbose:
        print(f"{name}")
        print(f"  handle       {np.round(grasp, 4)}  outward {np.round(outward, 2)}")
        print(f"  standoff     {best['standoff']:.2f} m, tilt "
              f"{np.degrees(best['tilt']):.0f} deg, roll {best['roll']:.2f} rad")
        print(f"  limit margin {best['margin']:.3f} rad, cond "
              f"{best['sv'][0] / best['sv'][-1]:.1f}")
        print(f"  weld gap     {check['gap']:.2e} m at reset, "
              f"{check['settled']:.2e} m after 2 s")
        print(f"  wrote        {scene_path}")
    return scene_path, check


def verify(scene_path, seconds=2.0):
    """Load the scene, hold under gravity compensation, measure the weld gap."""
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    mujoco.mj_forward(model, data)
    site = model.site("attachment_site").id
    handle = model.site(f"{PREFIX}handle").id
    gap = float(np.linalg.norm(data.site(site).xpos - data.site(handle).xpos))

    arm_dofs = np.array([model.joint(f"joint{i}").id for i in range(1, 8)])
    for _ in range(int(seconds / model.opt.timestep)):
        data.ctrl[:] = np.clip(data.qfrc_bias[arm_dofs], *model.actuator_ctrlrange.T)
        mujoco.mj_step(model, data)
    settled = float(np.linalg.norm(data.site(site).xpos - data.site(handle).xpos))
    return {"gap": gap, "settled": settled, "hinge": float(data.qpos[-1])}


def object_names():
    return sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob("data/*/object.xml"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("name", nargs="?", help="object directory under data/")
    parser.add_argument("--all", action="store_true", help="build every object")
    parser.add_argument("--list", action="store_true", help="list objects")
    args = parser.parse_args()

    os.chdir(REPO)

    if args.list:
        for name in object_names():
            handle = detect_handle(name)
            status = "ok" if not handle["problems"] else handle["problems"][0]
            print(f"  {name:18s} {status}")
        return

    if args.all:
        built, skipped = [], []
        for name in object_names():
            try:
                build(name)
                built.append(name)
            except SystemExit as exc:
                skipped.append((name, str(exc).split("\n")[1].strip(" -")
                                if "\n" in str(exc) else str(exc)))
                print(f"{name}\n  SKIPPED: {skipped[-1][1]}")
        print(f"\nbuilt {len(built)}, skipped {len(skipped)}")
        return

    if not args.name:
        parser.error("give an object name, or --all, or --list")
    build(args.name)


if __name__ == "__main__":
    main()
