"""Cartesian impedance control of an iiwa welded to an articulated object's handle.

    python3 articulation_controller.py                 # kitchen-cabinet
    python3 articulation_controller.py toolbox

Works with any scene built by scenes/build_scene.py. The arm starts already
grasping the handle (scene keyframe "grasp") with a rigid weld between link7 and
the object's lid. From there it is given a sequence of straight-line POSITION
trajectories around the handle while the commanded orientation is held fixed at
the grasp orientation.

Note what the weld does to the degrees of freedom: 7 arm joints + 1 hinge = 8,
minus 6 weld constraints = 2. Only one of those moves the end-effector, and it
is the lid arc. So the end-effector cannot follow an arbitrary position target
- it follows the component of the target that lies along the arc, and the rest
shows up as tracking error and constraint force. That is the point of the
experiment, not a bug: compare dx_error.png against door_angle.png.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from utils.dual_quat_traj import PositionTrajectory
from utils.dx_plot import (
    output_dir,
    plot_door_angle,
    plot_dx,
    plot_paths,
    plot_relative_frame,
)
from utils.frames import FrameTracker, quat_angle
from utils.goal_sampler import sample_positions_around
from utils.traj_overlay import TrajectoryOverlay

# Cartesian impedance control gains.
impedance_pos = np.asarray([200.0, 200.0, 200.0])  # [N/m]
impedance_ori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])

damping_ratio = 1.0
Kpos: float = 0.95
Kori: float = 0.95
integration_dt: float = 1.0
gravity_compensation: bool = True
dt: float = 0.002

# Hold the commanded orientation at the grasp orientation. The weld ties the
# end-effector orientation to the door, so holding it fixed actively resists the
# door rotating. Set False to drop the orientation task entirely and let the
# weld alone decide the orientation.
hold_orientation: bool = True

# Position goals, sampled in a shell around the handle's starting position.
n_goals: int = 4
goal_seed: int | None = 0
goal_r_min: float = 0.06
goal_r_max: float = 0.18
segment_duration: float = 3.0
settle_time: float = 1.0

overlay_every: int = 25

# Object directory under data/. Its scene must already be built:
#   python3 scenes/build_scene.py <object_name>
object_name = "kitchen-cabinet"

# The prefix build_scene.py namespaces the object under.
PREFIX = "obj_"


def main(object_name=object_name) -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    scene_path = f"scenes/{object_name}_iiwa_scene.xml"
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    model.opt.timestep = dt

    damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
    damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
    Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
    Kd = np.concatenate([damping_pos, damping_ori], axis=0)
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

    site_id = model.site("attachment_site").id

    joint_names = [f"joint{i}" for i in range(1, 8)]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    key_id = model.key("grasp").id
    q0 = model.key("grasp").qpos[dof_ids]

    hinge_dof = model.joint(f"{PREFIX}lid_hinge").id
    hinge_limit = model.jnt_range[hinge_dof, 1]

    jac = np.zeros((6, model.nv))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((model.nv, model.nv))

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # Reference frame for the handle: wherever it is right now, at the grasp.
        handle = FrameTracker(model, data, f"{PREFIX}handle")
        print(f"handle reference frame: pos {np.round(handle.ref_pos, 4)} "
              f"quat {np.round(handle.ref_quat, 4)}")

        # Orientation held for the whole run: the grasp orientation.
        grasp_quat = np.zeros(4)
        mujoco.mju_mat2Quat(grasp_quat, data.site(site_id).xmat)

        goals = sample_positions_around(
            handle.ref_pos, n_goals, goal_r_min, goal_r_max, seed=goal_seed
        )

        overlay = TrajectoryOverlay(viewer.user_scn)
        overlay.set_goals(goals)

        def start_segment(index):
            segment = PositionTrajectory(
                pos_start=data.site(site_id).xpos.copy(),
                pos_goal=goals[index],
                quat_hold=grasp_quat,
                duration=segment_duration,
            )
            overlay.set_commanded(segment)
            overlay.active_goal = index
            return segment

        segment_index = 0
        traj = start_segment(segment_index)
        t_seg = t_sim = 0.0
        step_count = 0

        log_t, log_dx, log_cmd, log_actual = [], [], [], []
        log_rel_pos, log_rel_ang, log_door, log_segment = [], [], [], []
        segment_marks = [(0.0, "goal 1")]

        while viewer.is_running():
            step_start = time.time()

            pos, quat = traj.sample(t_seg)
            site_pos = data.site(site_id).xpos.copy()
            dx = pos - site_pos

            rel_pos, rel_quat = handle.relative(data)
            log_t.append(t_sim)
            log_dx.append(dx.copy())
            log_cmd.append(pos.copy())
            log_actual.append(site_pos)
            log_rel_pos.append(rel_pos.copy())
            log_rel_ang.append(quat_angle(rel_quat))
            log_door.append(float(data.qpos[hinge_dof]))
            log_segment.append(segment_index)

            twist[:3] = Kpos * dx / integration_dt
            if hold_orientation:
                mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
                mujoco.mju_negQuat(site_quat_conj, site_quat)
                mujoco.mju_mulQuat(error_quat, quat, site_quat_conj)
                mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
                twist[3:] *= Kori / integration_dt
            else:
                twist[3:] = 0.0

            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
            Mx_inv = jac @ M_inv @ jac.T
            if abs(np.linalg.det(Mx_inv)) >= 1e-2:
                Mx = np.linalg.inv(Mx_inv)
            else:
                Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

            tau_full = jac.T @ Mx @ (Kp * twist - Kd * (jac @ data.qvel))

            # Joint task in the nullspace. Only the arm DOFs are actuated, so the
            # joint-space term is written into the arm block alone; the hinge DOF
            # rides along in the Jacobian but takes no command.
            Jbar = M_inv @ jac.T @ Mx
            ddq = np.zeros(model.nv)
            ddq[dof_ids] = (
                Kp_null * (q0 - data.qpos[dof_ids]) - Kd_null * data.qvel[dof_ids]
            )
            tau_full += (np.eye(model.nv) - jac.T @ Jbar.T) @ ddq

            tau = tau_full[dof_ids]
            if gravity_compensation:
                tau += data.qfrc_bias[dof_ids]

            np.clip(tau, *model.actuator_ctrlrange.T, out=tau)
            data.ctrl[actuator_ids] = tau
            mujoco.mj_step(model, data)

            t_seg += dt
            t_sim += dt
            step_count += 1

            if t_seg >= segment_duration + settle_time and segment_index + 1 < n_goals:
                segment_index += 1
                traj = start_segment(segment_index)
                segment_marks.append((t_sim, f"goal {segment_index + 1}"))
                t_seg = 0.0

            overlay.append_actual(data.site(site_id).xpos)
            if step_count % overlay_every == 0:
                overlay.redraw()

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    if log_t:
        t = np.array(log_t)
        dx_log = np.array(log_dx)
        rel_pos_log = np.array(log_rel_pos)
        door_log = np.array(log_door)

        out = output_dir(object_name)
        np.savez(
            f"{out}/run_log.npz",
            t=t,
            dx=dx_log,
            commanded=np.array(log_cmd),
            actual=np.array(log_actual),
            handle_rel_pos=rel_pos_log,
            handle_rel_angle=np.array(log_rel_ang),
            handle_ref_pos=handle.ref_pos,
            handle_ref_quat=handle.ref_quat,
            door_angle=door_log,
            segment=np.array(log_segment),
            goals=goals,
        )
        plot_dx(t, dx_log, path=f"{out}/dx_error.png", segments=segment_marks)
        plot_paths(t, np.array(log_cmd), np.array(log_actual),
                   path=f"{out}/paths.png", segments=segment_marks)
        plot_relative_frame(t, rel_pos_log, np.array(log_rel_ang),
                            path=f"{out}/handle_frame.png", segments=segment_marks,
                            name=f"{object_name} handle")
        plot_door_angle(t, door_log, path=f"{out}/door_angle.png",
                        segments=segment_marks, limit=hinge_limit,
                        name=object_name)
        print(
            f"{len(t)} samples over {t[-1]:.2f} s, {segment_index + 1} goal(s)\n"
            f"  final |dx|          = {np.linalg.norm(dx_log[-1]):.4f} m\n"
            f"  handle displacement = {np.linalg.norm(rel_pos_log[-1]):.4f} m "
            f"(max {np.linalg.norm(rel_pos_log, axis=1).max():.4f} m)\n"
            f"  hinge angle         = {door_log[-1]:.4f} rad "
            f"(max {door_log.max():.4f} of {hinge_limit:.4f})\n"
            f"  wrote {out}/"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("object", nargs="?", default=object_name,
                        help="object directory under data/")
    main(parser.parse_args().object)
