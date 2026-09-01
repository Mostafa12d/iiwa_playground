"""Collect a per-timestep dataset of iiwa-on-articulated-object interaction.

    python3.10 collect.py                        # all subset objects
    python3.10 collect.py --objects safe -n 5

Writes datasets/<tag>.npz (one row per timestep, all episodes concatenated) and
datasets/<tag>_schema.json (units, frames, field meanings).

Fields are namespaced:
  obs/   what a real robot could measure
  gt/    privileged ground truth - analysis only, never a model input
  meta/  bookkeeping and labels
  flag/  validity flags

NOTE: the impedance control law here is duplicated from articulation_controller.py
rather than imported, since that file is a script. Keep them in sync by hand.
"""

import argparse
import json
import os
import time

import mujoco
import numpy as np

from utils.dual_quat_traj import DualQuaternionTrajectory
from utils.dx_plot import output_dir

PREFIX = "obj_"
OBJECTS = ("kitchen-cabinet", "minifridge", "safe")
FAMILY = {"kitchen-cabinet": "door", "minifridge": "door", "safe": "door"}

# --- controller ------------------------------------------------------------
# Low impedance: friction at the handle is only 0.06-0.16 N on these objects, so
# stiffer gains make the arm insensitive to the object. See notes in the schema.
KP_POS = 50.0    # [N/m]
KP_ORI = 25.0    # [Nm/rad]
DAMPING_RATIO = 1.0
KP_NULL = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
KPOS = KORI = 0.95
INTEGRATION_DT = 1.0
DT = 0.002

# --- episode ---------------------------------------------------------------
SEGMENT_DURATION = 3.0
SETTLE_VEL = 0.005      # [m/s]
SETTLE_WINDOW = 1.0     # [s]
MAX_PHASE_TIME = 12.0   # [s] raised from 8: `safe` is ~15x heavier at the handle
NEAR_LIMIT_FRAC = 0.05  # within this fraction of travel counts as near-limit
QACC_BAD = 1e4
JUMP_SIGMA = 8.0        # single-step jump this many MADs out is an outlier


def ground_truth(model, data, hinge_id, handle_id):
    """Privileged object mechanics. Read once per episode."""
    hdof = model.jnt_dofadr[hinge_id]
    lid = model.body(f"{PREFIX}lid").id
    base = model.body(f"{PREFIX}base").id
    axis, anchor = data.xaxis[hinge_id].copy(), data.xanchor[hinge_id].copy()
    handle = data.site(handle_id).xpos.copy()

    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)

    lever = float(np.linalg.norm(np.cross(handle - anchor, axis)))
    return {
        "lid_mass": float(model.body_mass[lid]),
        "base_mass": float(model.body_mass[base]),
        "lid_inertia_diag": model.body_inertia[lid].copy(),
        "inertia_about_hinge": float(M[hdof, hdof]),
        "hinge_damping": float(model.dof_damping[hdof]),
        "hinge_frictionloss": float(model.dof_frictionloss[hdof]),
        "hinge_stiffness": float(model.jnt_stiffness[hinge_id]),
        "hinge_armature": float(model.dof_armature[hdof]),
        "hinge_axis": axis,
        "hinge_anchor": anchor,
        "hinge_range": model.jnt_range[hinge_id].copy(),
        "lever_arm": lever,
        "effective_mass_at_handle": float(M[hdof, hdof]) / lever**2,
        "weld_solref": model.eq_solref[model.equality("ee_to_handle").id].copy(),
        "weld_solimp": model.eq_solimp[model.equality("ee_to_handle").id].copy(),
    }


def run_episode(model, data, goal_pos, goal_quat, start_theta, rng):
    """One reach attempt. Returns a dict of (T, ...) arrays."""
    site_id = model.site("attachment_site").id
    handle_id = model.site(f"{PREFIX}handle").id
    lid_id = model.body(f"{PREFIX}lid").id
    jids = np.array([model.joint(f"joint{i}").id for i in range(1, 8)])
    dof_ids = model.jnt_dofadr[jids]
    qpos_ids = model.jnt_qposadr[jids]
    act_ids = np.array([model.actuator(f"joint{i}").id for i in range(1, 8)])
    hinge_id = model.joint(f"{PREFIX}lid_hinge").id
    hq, hdof = model.jnt_qposadr[hinge_id], model.jnt_dofadr[hinge_id]
    lo, hi = model.jnt_range[hinge_id]
    weld_eq = model.equality("ee_to_handle").id

    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    data.qpos[hq] = start_theta
    mujoco.mj_forward(model, data)
    q0 = data.qpos[qpos_ids].copy()

    Kp = np.concatenate([np.full(3, KP_POS), np.full(3, KP_ORI)])
    Kd = DAMPING_RATIO * 2 * np.sqrt(Kp)
    Kd_null = DAMPING_RATIO * 2 * np.sqrt(KP_NULL)

    sq = np.zeros(4)
    mujoco.mju_mat2Quat(sq, data.site(site_id).xmat)
    traj = DualQuaternionTrajectory(
        pos_start=data.site(site_id).xpos.copy(), quat_start=sq.copy(),
        pos_goal=goal_pos, quat_goal=goal_quat, duration=SEGMENT_DURATION,
    )

    jac = np.zeros((6, model.nv))
    twist = np.zeros(6)
    a, b, c = np.zeros(4), np.zeros(4), np.zeros(4)
    dth = np.zeros(3)
    M_inv = np.zeros((model.nv, model.nv))
    obj_vel = np.zeros(6)

    rec = {k: [] for k in (
        "t", "hinge_q", "hinge_qvel", "hinge_qacc",
        "hinge_qfrc_constraint", "hinge_qfrc_bias", "hinge_qfrc_applied",
        "hinge_qfrc_actuator", "travel_frac", "dist_to_limit",
        "qfrc_constraint_all", "efc_force", "efc_pos", "efc_type",
        "weld_wrench_ee", "handle_pos", "handle_quat", "handle_linvel",
        "handle_angvel", "arm_qpos", "arm_qvel", "arm_tau_cmd",
        "arm_tau_actuator", "ee_pos", "ee_quat", "ee_twist",
        "cmd_pos", "cmd_quat", "dx", "dtheta",
        "solver_iter", "solver_residual", "nefc",
        "flag_saturated", "flag_near_limit", "flag_diverged",
    )}

    below, t_seg, blew = 0.0, 0.0, False
    while t_seg < MAX_PHASE_TIME:
        pos, quat = traj.sample(t_seg)
        ee_pos = data.site(site_id).xpos.copy()
        dx = pos - ee_pos
        mujoco.mju_mat2Quat(a, data.site(site_id).xmat)
        mujoco.mju_negQuat(b, a)
        mujoco.mju_mulQuat(c, quat, b)
        mujoco.mju_quat2Vel(dth, c, 1.0)

        twist[:3] = KPOS * dx / INTEGRATION_DT
        twist[3:] = dth * (KORI / INTEGRATION_DT)

        mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        ee_twist = jac @ data.qvel
        mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
        Mxi = jac @ M_inv @ jac.T
        Mx = (np.linalg.inv(Mxi) if abs(np.linalg.det(Mxi)) >= 1e-2
              else np.linalg.pinv(Mxi, rcond=1e-2))

        tau_full = jac.T @ Mx @ (Kp * twist - Kd * ee_twist)
        Jbar = M_inv @ jac.T @ Mx
        ddq = np.zeros(model.nv)
        ddq[dof_ids] = (KP_NULL * (q0 - data.qpos[qpos_ids])
                        - Kd_null * data.qvel[dof_ids])
        tau_full += (np.eye(model.nv) - jac.T @ Jbar.T) @ ddq

        tau = tau_full[dof_ids] + data.qfrc_bias[dof_ids]
        pre = tau.copy()
        np.clip(tau, *model.actuator_ctrlrange.T, out=tau)
        saturated = bool(np.any(np.abs(pre - tau) > 1e-9))
        data.ctrl[act_ids] = tau

        # --- log before stepping, so state and action line up -------------
        eq_rows = np.flatnonzero(
            data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        efc_f = np.zeros(6)
        efc_p = np.zeros(6)
        efc_t = np.full(6, -1)
        n = min(len(eq_rows), 6)
        efc_f[:n] = data.efc_force[eq_rows[:n]]
        efc_p[:n] = data.efc_pos[eq_rows[:n]]
        efc_t[:n] = data.efc_type[eq_rows[:n]]

        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_SITE, handle_id, obj_vel, 0)

        hquat = np.zeros(4)
        mujoco.mju_mat2Quat(hquat, data.site(handle_id).xmat)
        equat = np.zeros(4)
        mujoco.mju_mat2Quat(equat, data.site(site_id).xmat)

        theta = float(data.qpos[hq])
        travel = (theta - lo) / (hi - lo)

        rec["t"].append(len(rec["t"]) * DT)
        rec["hinge_q"].append(theta)
        rec["hinge_qvel"].append(float(data.qvel[hdof]))
        rec["hinge_qacc"].append(float(data.qacc[hdof]))
        rec["hinge_qfrc_constraint"].append(float(data.qfrc_constraint[hdof]))
        rec["hinge_qfrc_bias"].append(float(data.qfrc_bias[hdof]))
        rec["hinge_qfrc_applied"].append(float(data.qfrc_applied[hdof]))
        rec["hinge_qfrc_actuator"].append(float(data.qfrc_actuator[hdof]))
        rec["travel_frac"].append(float(travel))
        rec["dist_to_limit"].append(float(min(theta - lo, hi - theta)))
        rec["qfrc_constraint_all"].append(data.qfrc_constraint.copy())
        rec["efc_force"].append(efc_f)
        rec["efc_pos"].append(efc_p)
        rec["efc_type"].append(efc_t)
        # 6D wrench the weld transmits at the EE, in the base frame. The weld's
        # only path into the arm's DOFs is J_arm^T F, so invert that against the
        # constraint force the solver actually produced.
        rec["weld_wrench_ee"].append(
            np.linalg.pinv(jac[:, dof_ids].T) @ data.qfrc_constraint[dof_ids])
        rec["handle_pos"].append(data.site(handle_id).xpos.copy())
        rec["handle_quat"].append(hquat)
        rec["handle_linvel"].append(obj_vel[3:].copy())
        rec["handle_angvel"].append(obj_vel[:3].copy())
        rec["arm_qpos"].append(data.qpos[qpos_ids].copy())
        rec["arm_qvel"].append(data.qvel[dof_ids].copy())
        rec["arm_tau_cmd"].append(tau.copy())
        rec["arm_tau_actuator"].append(data.qfrc_actuator[dof_ids].copy())
        rec["ee_pos"].append(ee_pos)
        rec["ee_quat"].append(equat)
        rec["ee_twist"].append(ee_twist.copy())
        rec["cmd_pos"].append(pos.copy())
        rec["cmd_quat"].append(quat.copy())
        rec["dx"].append(dx.copy())
        rec["dtheta"].append(dth.copy())
        rec["solver_iter"].append(int(data.solver_niter[0]))
        rec["solver_residual"].append(
            float(data.solver[0].improvement) if data.nefc else 0.0)
        rec["nefc"].append(int(data.nefc))
        rec["flag_saturated"].append(saturated)
        rec["flag_near_limit"].append(
            bool(min(theta - lo, hi - theta) < NEAR_LIMIT_FRAC * (hi - lo)))

        mujoco.mj_step(model, data)

        bad = (not np.isfinite(data.qacc).all()
               or np.abs(data.qacc).max() > QACC_BAD)
        rec["flag_diverged"].append(bad)
        if bad:
            blew = True
            break

        speed = float(np.linalg.norm(jac[:3] @ data.qvel))
        below = below + DT if speed < SETTLE_VEL else 0.0
        t_seg += DT
        if t_seg >= SEGMENT_DURATION and below >= SETTLE_WINDOW:
            break

    out = {k: np.asarray(v) for k, v in rec.items()}
    out["flag_diverged"] = np.asarray(
        rec["flag_diverged"] + [blew] * (len(out["t"]) - len(rec["flag_diverged"]))
    )
    return out, blew


def flag_outliers(x):
    """Single-step jump outliers, by median absolute deviation."""
    if len(x) < 3:
        return np.zeros(len(x), dtype=bool)
    d = np.abs(np.diff(x, prepend=x[0]))
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    scale = max(1.4826 * mad, 0.01 * float(np.ptp(x)), 1e-12)
    return d > med + JUMP_SIGMA * scale


def collect(objects, n_episodes, seed, tag):
    rng = np.random.default_rng(seed)
    rows, episodes, gts = [], [], {}
    ep_id = 0

    for obj_id, name in enumerate(objects):
        model = mujoco.MjModel.from_xml_path(f"scenes/{name}_iiwa_scene.xml")
        model.opt.timestep = DT
        data = mujoco.MjData(model)
        hinge_id = model.joint(f"{PREFIX}lid_hinge").id
        handle_id = model.site(f"{PREFIX}handle").id

        mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
        mujoco.mj_forward(model, data)
        gts[name] = ground_truth(model, data, hinge_id, handle_id)

        surv = np.load(f"{output_dir(name)}/reachability.npz")
        feas = np.flatnonzero(surv["feasible"])
        if not len(feas):
            print(f"{name}: no feasible targets, skipping")
            continue

        for _ in range(n_episodes):
            k = int(rng.choice(feas))
            theta = float(surv["hinge_angle"][k])
            goal = surv["targets"][k]
            # target orientation for the hinge angle this target was surveyed at
            ai = int(np.argmin(np.abs(surv["hinge_fractions"]
                                      * gts[name]["hinge_range"][1] - theta)))
            gquat = surv["target_quats"][ai]

            t0 = time.time()
            ep, blew = run_episode(model, data, goal, gquat, theta, rng)
            T = len(ep["t"])
            if T == 0:
                continue

            ep["flag_outlier_qfrc"] = flag_outliers(ep["hinge_qfrc_constraint"])
            ep["episode_id"] = np.full(T, ep_id)
            ep["object_id"] = np.full(T, obj_id)
            ep["step"] = np.arange(T)
            rows.append(ep)

            start_handle = ep["handle_pos"][0]
            dev = float(np.linalg.norm(goal - start_handle))
            episodes.append({
                "episode_id": ep_id, "object": name, "object_id": obj_id,
                "family": FAMILY[name], "start_hinge_angle": theta,
                "goal": goal.tolist(), "goal_quat": gquat.tolist(),
                "deviation_from_handle_m": dev,
                "steps": T, "diverged": bool(blew),
                "travel_reached": float(ep["travel_frac"].max()),
                "wall_s": round(time.time() - t0, 2),
            })
            print(f"  ep {ep_id:3d} {name:16s} theta0={theta:5.3f} "
                  f"T={T:5d} travel={ep['travel_frac'].max():.3f} "
                  f"{'DIVERGED' if blew else ''}")
            ep_id += 1

    if not rows:
        raise SystemExit("no episodes collected")

    keys = rows[0].keys()
    ds = {k: np.concatenate([r[k] for r in rows]) for k in keys}

    os.makedirs("datasets", exist_ok=True)
    path = f"datasets/{tag}.npz"
    gt_flat = {f"gt/{n}/{k}": v for n, g in gts.items() for k, v in g.items()}
    np.savez_compressed(path, **ds, **gt_flat,
                        objects=np.array(objects, dtype=object))

    with open(f"datasets/{tag}_schema.json", "w") as f:
        json.dump(schema(objects, episodes, gts), f, indent=2, default=str)

    n = len(ds["t"])
    print(f"\n{len(episodes)} episodes, {n} timesteps -> {path} "
          f"({os.path.getsize(path)/1e6:.1f} MB)")
    print(f"  diverged episodes : {sum(e['diverged'] for e in episodes)}")
    print(f"  saturated steps   : {int(ds['flag_saturated'].sum())} ({100*ds['flag_saturated'].mean():.1f}%)")
    print(f"  near-limit steps  : {int(ds['flag_near_limit'].sum())}")
    print(f"  qfrc outlier steps: {int(ds['flag_outlier_qfrc'].sum())}")
    print(f"  travel reached    : "
          f"mean {np.mean([e['travel_reached'] for e in episodes]):.3f} "
          f"max {np.max([e['travel_reached'] for e in episodes]):.3f}")
    return path


def schema(objects, episodes, gts):
    return {
        "frames": {
            "world": "IDENTICAL to the robot base frame. Scenes are built with "
                     "arm_mount at the origin, unrotated, so every position, "
                     "velocity, Jacobian and wrench below is already "
                     "base-relative. No transform is applied anywhere.",
            "orientation": "quaternions are (w, x, y, z), MuJoCo convention",
            "twist": "ee_twist and handle vel are [linear(3), angular(3)] in "
                     "the base frame, from mj_jacSite / mj_objectVelocity",
        },
        "units": {
            "length": "m", "angle": "rad", "time": "s", "force": "N",
            "torque": "Nm", "mass": "kg", "inertia": "kg m^2",
            "stiffness_pos": "N/m", "stiffness_ori": "Nm/rad",
        },
        "controller": {
            "type": "Cartesian impedance + nullspace posture task",
            "Kp_pos": KP_POS, "Kp_ori": KP_ORI,
            "damping_ratio": DAMPING_RATIO, "Kp_null": KP_NULL.tolist(),
            "dt": DT,
            "note": "Kp is LOW on purpose: stiction+damping at the handle is "
                    "only 0.06-0.16 N on these objects, so stiffer gains make "
                    "the arm insensitive to the object.",
            "caveat": "the nullspace posture reference is the CLOSED-door "
                      "keyframe, so it pulls the door shut and costs roughly "
                      "20-25% of achievable travel. Not yet fixed.",
        },
        "namespaces": {
            "obs/": "measurable on a real robot",
            "gt/": "privileged. analysis only, never a model input",
            "flag_": "validity flags",
        },
        "known_limitations": [
            "Goals come from a free-space IK survey, so most are off the weld "
            "constraint manifold; the door typically parks well short of its "
            "limit. Travel coverage is partial.",
            "No mechanics randomization yet: every object has "
            "damping=0.08, frictionloss=0.03, stiffness=0. The impedance gain "
            "is therefore only weakly identifiable from this dataset.",
            "No sensor-noise / delay variant. All signals are idealized.",
            "weld_wrench_ee is recovered as pinv(J_arm^T) @ qfrc_constraint[arm], "
            "i.e. the wrench at the EE consistent with the arm-side constraint "
            "force. Raw efc_force/efc_pos and qfrc_constraint_all are logged "
            "so it can be re-derived differently offline.",
            "A goal surveyed at one hinge angle may be run from a different "
            "start angle in principle; here start angle is taken from the "
            "survey row, so they match.",
        ],
        "objects": {n: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in g.items()} for n, g in gts.items()},
        "episodes": episodes,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--objects", nargs="*", default=list(OBJECTS))
    p.add_argument("-n", "--episodes", type=int, default=10,
                   help="episodes per object")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="v1")
    a = p.parse_args()
    collect(a.objects, a.episodes, a.seed, a.tag)


if __name__ == "__main__":
    main()
