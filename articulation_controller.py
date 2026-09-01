"""Cartesian impedance control of an iiwa welded to an articulated object's handle.

    python3 articulation_controller.py                 # kitchen-cabinet
    python3 articulation_controller.py toolbox
"""

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

from utils.dual_quat_traj import DualQuaternionTrajectory
from utils.dx_plot import (
    output_dir,
    plot_door_angle,
    plot_dtheta,
    plot_dx,
    plot_paths,
    plot_relative_frame,
)
from utils.frames import FrameTracker, quat_angle
from utils.goal_sampler import sample_feasible_targets
from utils.impedance import CartesianImpedance
from utils.traj_overlay import TrajectoryOverlay

# Cartesian impedance control gains.
impedance_pos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
# impedance_pos = np.asarray([1.0, 1.0, 1.0])  # [N/m]

impedance_ori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
# Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])


damping_ratio = 1.0
Kpos: float = 0.95
Kori: float = 0.95
integration_dt: float = 1.0
gravity_compensation: bool = True
dt: float = 0.002

# Track the commanded orientation along the arc. Set False to drop the
# orientation task and let the weld alone decide the orientation.
track_orientation: bool = True

# Goals are drawn from the feasible targets of a reachability survey. Run
#   python3 reachability_survey.py <object_name>
# first to produce plots/<object_name>/reachability.npz.
n_goals: int = 10
goal_seed: int | None = 0

# A phase ends when the EE settles (speed below settle_vel for settle_window),
# not after a fixed time. segment_duration only sets the setpoint slew rate;
# max_phase_time caps a phase that never settles.
segment_duration: float = 3.0
settle_vel: float = 0.005  # [m/s]
settle_window: float = 1.0  # [s]
max_phase_time: float = 8.0  # [s]

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

    ctrl = CartesianImpedance(
        model, impedance_pos, impedance_ori, Kp_null,
        damping_ratio=damping_ratio, kpos=Kpos, kori=Kori,
        integration_dt=integration_dt,
        gravity_compensation=gravity_compensation,
        track_orientation=track_orientation,
    )

    site_id = model.site("attachment_site").id

    # Joint IDs are not qpos/DOF addresses. They coincide for a chain of hinges
    # but diverge the moment a scene gains a free joint, so resolve both.
    joint_names = [f"joint{i}" for i in range(1, 8)]
    joint_ids = np.array([model.joint(name).id for name in joint_names])
    qpos_ids = model.jnt_qposadr[joint_ids]

    key_id = model.key("grasp").id
    q0 = model.key("grasp").qpos[qpos_ids]

    hinge_name = f"{PREFIX}lid_hinge"
    hinge_id = model.joint(hinge_name).id
    hinge_qadr = model.jnt_qposadr[hinge_id]
    hinge_limit = model.jnt_range[hinge_id, 1]

    ctrl.q0 = q0

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Reference frame for the handle: wherever it is right now, at the grasp.
        handle = FrameTracker(model, data, f"{PREFIX}handle")
        print(f"handle reference frame: pos {np.round(handle.ref_pos, 4)} "
              f"quat {np.round(handle.ref_quat, 4)}")

        # Goals: feasible EE targets from the reachability survey. The arm is
        # welded to the handle, so most of these are off the lid arc and cannot
        # actually be reached - commanding toward them is the interaction test.
        survey_npz = f"{output_dir(object_name)}/reachability.npz"
        if not os.path.exists(survey_npz):
            raise SystemExit(
                f"{survey_npz} not found. Run:\n"
                f"  python3 reachability_survey.py {object_name}"
            )
        goals, goal_quats = sample_feasible_targets(survey_npz, n_goals, goal_seed)
        print(f"{n_goals} feasible goals from {survey_npz}")

        overlay = TrajectoryOverlay(viewer.user_scn)
        overlay.set_goals(goals)

        # Each goal is an independent attempt from the initial grasp pose: drive
        # out toward the goal, then drive back to this pose before the next goal.
        init_pos = data.site(site_id).xpos.copy()
        init_quat = np.zeros(4)
        mujoco.mju_mat2Quat(init_quat, data.site(site_id).xmat)

        start_quat = np.zeros(4)

        def start_segment(pos_goal, quat_goal, index):
            mujoco.mju_mat2Quat(start_quat, data.site(site_id).xmat)
            segment = DualQuaternionTrajectory(
                pos_start=data.site(site_id).xpos.copy(),
                quat_start=start_quat.copy(),
                pos_goal=pos_goal,
                quat_goal=quat_goal,
                duration=segment_duration,
            )
            overlay.set_commanded(segment)
            overlay.active_goal = index
            return segment

        segment_index = 0
        returning = False
        done = False
        below_time = 0.0
        traj = start_segment(goals[0], goal_quats[0], 0)
        t_seg = t_sim = 0.0
        step_count = 0

        log_t, log_dx, log_dtheta, log_cmd, log_actual = [], [], [], [], []
        log_rel_pos, log_rel_ang, log_door, log_segment = [], [], [], []
        segment_marks = [(0.0, "goal 1")]

        while viewer.is_running() and not done:
            step_start = time.time()

            pos, quat = traj.sample(t_seg)
            out = ctrl.compute(data, pos, quat)
            site_pos, dx, dtheta = out.ee_pos, out.dx, out.dtheta

            rel_pos, rel_quat = handle.relative(data)
            log_t.append(t_sim)
            log_dx.append(dx.copy())
            log_dtheta.append(dtheta.copy())
            log_cmd.append(pos.copy())
            log_actual.append(site_pos)
            log_rel_pos.append(rel_pos.copy())
            log_rel_ang.append(quat_angle(rel_quat))
            log_door.append(float(data.qpos[hinge_qadr]))
            log_segment.append(segment_index)

            ee_speed = np.linalg.norm(out.ee_twist[:3])
            below_time = below_time + dt if ee_speed < settle_vel else 0.0

            ctrl.apply(data, out)
            mujoco.mj_step(model, data)

            t_seg += dt
            t_sim += dt
            step_count += 1

            settled = t_seg >= segment_duration and below_time >= settle_window
            if settled or t_seg >= max_phase_time:
                below_time = 0.0
                if not returning:
                    # Out phase done; drive back to the initial grasp pose.
                    returning = True
                    traj = start_segment(init_pos, init_quat, segment_index)
                    segment_marks.append((t_sim, f"return {segment_index + 1}"))
                    t_seg = 0.0
                elif segment_index + 1 < n_goals:
                    # Back at the start; begin the next goal's out phase.
                    segment_index += 1
                    returning = False
                    traj = start_segment(
                        goals[segment_index], goal_quats[segment_index], segment_index
                    )
                    segment_marks.append((t_sim, f"goal {segment_index + 1}"))
                    t_seg = 0.0
                else:
                    done = True

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
        dtheta_log = np.array(log_dtheta)
        rel_pos_log = np.array(log_rel_pos)
        door_log = np.array(log_door)

        out = output_dir(object_name)
        np.savez(
            f"{out}/run_log.npz",
            t=t,
            dx=dx_log,
            dtheta=dtheta_log,
            commanded=np.array(log_cmd),
            actual=np.array(log_actual),
            handle_rel_pos=rel_pos_log,
            handle_rel_angle=np.array(log_rel_ang),
            handle_ref_pos=handle.ref_pos,
            handle_ref_quat=handle.ref_quat,
            door_angle=door_log,
            segment=np.array(log_segment),
            goals=goals,
            goal_quats=goal_quats,
        )
        plot_dx(t, dx_log, path=f"{out}/dx_error.png", segments=segment_marks)
        plot_dtheta(t, dtheta_log, path=f"{out}/dtheta_error.png",
                    segments=segment_marks)
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
            f"  final |dtheta|      = {np.linalg.norm(dtheta_log[-1]):.4f} rad\n"
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
