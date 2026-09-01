import mujoco
import mujoco.viewer
import numpy as np
import time
from utils.dual_quat_traj import DualQuaternionTrajectory
from utils.dx_plot import output_dir, plot_dx, plot_paths
from utils.goal_sampler import sample_goal_poses
from utils.impedance import CartesianImpedance
from utils.traj_overlay import TrajectoryOverlay, hide_body_geoms

# Cartesian impedance control gains.
impedance_pos = np.asarray([10.0, 10.0, 10.0])  # [N/m]
impedance_ori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])

# Damping ratio for both Cartesian and joint impedance control.
damping_ratio = 1.0

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori: float = 0.95

# Integration timestep in seconds.
integration_dt: float = 1.0

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Goal sequence. Endpoints are sampled from the reachable workspace shell in
# goal_sampler.py; each segment runs for segment_duration then holds for
# settle_time so the impedance response can settle before the next goal.
n_goals: int = 4
goal_seed: int | None = 0
segment_duration: float = 3.0
settle_time: float = 1.0

# Drive the mocap box along the commanded pose so the target is visible.
drive_mocap: bool = True

# Redraw the viewer overlay every N physics steps (~20 Hz at dt = 0.002).
overlay_every: int = 25

# Figures and the run log land in plots/<run_name>/.
run_name = "free_space"


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("kuka_iiwa_14/scene.xml")
    data = mujoco.MjData(model)

    model.opt.timestep = dt

    ctrl = CartesianImpedance(
        model, impedance_pos, impedance_ori, Kp_null,
        damping_ratio=damping_ratio, kpos=Kpos, kori=Kori,
        integration_dt=integration_dt,
        gravity_compensation=gravity_compensation,
    )

    # End-effector site we wish to control.
    site_name = "attachment_site"
    site_id = model.site(site_name).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos

    # Mocap body carrying the commanded pose. Its box is hidden; the site frame
    # it holds is what shows the commanded pose travelling.
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]
    hide_body_geoms(model, mocap_name)

    ctrl.q0 = q0

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation. mj_forward populates the site pose, which the
        # trajectory start below reads.
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # Goal endpoints, sampled from the reachable workspace shell.
        goal_pos, goal_quat = sample_goal_poses(model, n_goals, seed=goal_seed)

        overlay = TrajectoryOverlay(viewer.user_scn)
        overlay.set_goals(goal_pos)

        def start_segment(index):
            """Trajectory from wherever the end-effector is now to goal `index`."""
            quat_now = np.zeros(4)
            mujoco.mju_mat2Quat(quat_now, data.site(site_id).xmat)
            segment = DualQuaternionTrajectory(
                pos_start=data.site(site_id).xpos.copy(),
                quat_start=quat_now,
                pos_goal=goal_pos[index],
                quat_goal=goal_quat[index],
                duration=segment_duration,
            )
            overlay.set_commanded(segment)
            overlay.active_goal = index
            return segment

        segment_index = 0
        traj = start_segment(segment_index)
        t_seg = 0.0
        t_sim = 0.0
        step_count = 0

        # Run log, written out once the viewer closes.
        log_t: list[float] = []
        log_dx: list[np.ndarray] = []
        log_cmd: list[np.ndarray] = []
        log_actual: list[np.ndarray] = []
        log_segment: list[int] = []
        segment_marks: list[tuple[float, str]] = [(0.0, "goal 1")]

        while viewer.is_running():
            step_start = time.time()

            pos, quat = traj.sample(t_seg)
            if drive_mocap:
                data.mocap_pos[mocap_id] = pos
                data.mocap_quat[mocap_id] = quat

            out = ctrl.compute(data, pos, quat)
            site_pos, dx = out.ee_pos, out.dx
            log_t.append(t_sim)
            log_dx.append(dx.copy())
            log_cmd.append(pos.copy())
            log_actual.append(site_pos)
            log_segment.append(segment_index)

            ctrl.apply(data, out)
            mujoco.mj_step(model, data)

            t_seg += dt
            t_sim += dt
            step_count += 1

            # Advance to the next goal once this one has run and settled.
            if (
                t_seg >= segment_duration + settle_time
                and segment_index + 1 < n_goals
            ):
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
        cmd_log = np.array(log_cmd)
        actual_log = np.array(log_actual)

        out = output_dir(run_name)
        np.savez(
            f"{out}/run_log.npz",
            t=t,
            dx=dx_log,
            commanded=cmd_log,
            actual=actual_log,
            segment=np.array(log_segment),
            goal_pos=goal_pos,
            goal_quat=goal_quat,
        )
        plot_dx(t, dx_log, path=f"{out}/dx_error.png", segments=segment_marks)
        plot_paths(t, cmd_log, actual_log, path=f"{out}/paths.png",
                   segments=segment_marks)
        print(
            f"{len(t)} samples over {t[-1]:.2f} s, {segment_index + 1} goal(s)\n"
            f"  final |dx| = {np.linalg.norm(dx_log[-1]):.4f} m\n"
            f"  wrote {out}/"
        )


if __name__ == "__main__":
    main()