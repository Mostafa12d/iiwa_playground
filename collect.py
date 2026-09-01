"""Collect a per-timestep dataset of iiwa-on-articulated-object interaction.

    python3.10 collect.py                        # all subset objects
    python3.10 collect.py --objects safe -n 5

Writes datasets/<tag>.npz (one row per timestep, all episodes concatenated) and
datasets/<tag>_schema.json (units, frames, controller and episode settings).

gt/<object>/* is privileged ground truth: analysis only, never a model input.
flag_* are validity flags.
"""

import argparse
import json
import os
import time

import mujoco
import numpy as np

from utils.dual_quat_traj import DualQuaternionTrajectory
from utils.dx_plot import output_dir
from utils.impedance import CartesianImpedance

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

# --- mechanics randomization ------------------------------------------------
# Per-episode draws, logged as gt_* columns. Ranges bracket each object's stock
# value and reach up to where the hinge actually resists the arm.
MECHANICS = {
    "hinge_damping": (0.01, 2.0),        # [Nms/rad]  stock 0.08
    "hinge_frictionloss": (0.005, 1.5),  # [Nm]       stock 0.03
    "hinge_stiffness": (0.0, 3.0),       # [Nm/rad]   stock 0.0
    "hinge_armature": (0.0, 0.05),       # [kg m^2]   stock 0.0
    "lid_mass_scale": (0.5, 2.0),        # scales lid mass and inertia
}
# Sampled in log space, so the stiff end is not oversampled.
LOG_UNIFORM = {"hinge_damping", "hinge_frictionloss", "lid_mass_scale"}


def sample_mechanics(rng, randomize=True):
    """Draw one episode's hinge/lid parameters. Deterministic given `rng`."""
    out = {}
    for name, (lo, hi) in MECHANICS.items():
        if not randomize:
            out[name] = None          # keep whatever the model already has
        elif name in LOG_UNIFORM:
            out[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            out[name] = float(rng.uniform(lo, hi))
    return out


def apply_mechanics(model, hinge_id, lid_id, params, base):
    """Write sampled parameters into the model. `base` holds the stock values."""
    hdof = model.jnt_dofadr[hinge_id]
    if params["hinge_damping"] is not None:
        model.dof_damping[hdof] = params["hinge_damping"]
    if params["hinge_frictionloss"] is not None:
        model.dof_frictionloss[hdof] = params["hinge_frictionloss"]
    if params["hinge_stiffness"] is not None:
        model.jnt_stiffness[hinge_id] = params["hinge_stiffness"]
    if params["hinge_armature"] is not None:
        model.dof_armature[hdof] = params["hinge_armature"]
    s = params["lid_mass_scale"]
    if s is not None:
        model.body_mass[lid_id] = base["lid_mass"] * s
        model.body_inertia[lid_id] = base["lid_inertia"] * s
        # body_mass/inertia feed cached model constants (subtree mass etc.),
        # which do not update on their own.
        mujoco.mj_setConst(model, mujoco.MjData(model))


def derived_gt(model, data, hinge_id, handle_id, lid_id):
    """Quantities that follow from the sampled mechanics. Call after applying."""
    hdof = model.jnt_dofadr[hinge_id]
    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    mujoco.mj_forward(model, data)

    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)
    axis, anchor = data.xaxis[hinge_id], data.xanchor[hinge_id]
    lever = float(np.linalg.norm(
        np.cross(data.site(handle_id).xpos - anchor, axis)))
    return {
        "lid_mass": float(model.body_mass[lid_id]),
        "inertia_about_hinge": float(M[hdof, hdof]),
        "lever_arm": lever,
        "effective_mass_at_handle": float(M[hdof, hdof]) / lever**2,
    }


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
    sq = np.zeros(4)
    mujoco.mju_mat2Quat(sq, data.site(site_id).xmat)
    traj = DualQuaternionTrajectory(
        pos_start=data.site(site_id).xpos.copy(), quat_start=sq.copy(),
        pos_goal=goal_pos, quat_goal=goal_quat, duration=SEGMENT_DURATION,
    )

    ctrl = CartesianImpedance(
        model, KP_POS, KP_ORI, KP_NULL, damping_ratio=DAMPING_RATIO,
        kpos=KPOS, kori=KORI, integration_dt=INTEGRATION_DT,
    )
    ctrl.capture_posture(data)
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
        out = ctrl.compute(data, pos, quat)
        ee_pos, dx, dth = out.ee_pos, out.dx, out.dtheta
        ee_twist, jac, tau = out.ee_twist, out.jac, out.tau
        saturated = out.saturated
        ctrl.apply(data, out)

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
        rec["ee_quat"].append(out.ee_quat)
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


def collect(objects, n_episodes, seed, tag, randomize=True):
    rows, episodes, gts = [], [], {}
    ep_id = 0

    for obj_id, name in enumerate(objects):
        model = mujoco.MjModel.from_xml_path(f"scenes/{name}_iiwa_scene.xml")
        model.opt.timestep = DT
        data = mujoco.MjData(model)
        hinge_id = model.joint(f"{PREFIX}lid_hinge").id
        handle_id = model.site(f"{PREFIX}handle").id
        lid_id = model.body(f"{PREFIX}lid").id
        base_mech = {"lid_mass": float(model.body_mass[lid_id]),
                     "lid_inertia": model.body_inertia[lid_id].copy()}

        mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
        mujoco.mj_forward(model, data)
        gts[name] = ground_truth(model, data, hinge_id, handle_id)

        surv = np.load(f"{output_dir(name)}/reachability.npz")
        feas = np.flatnonzero(surv["feasible"])
        if not len(feas):
            print(f"{name}: no feasible targets, skipping")
            continue

        for ep_k in range(n_episodes):
            # Seeded from (root, object, episode), so a given episode is
            # reproducible regardless of how many objects or episodes were
            # requested alongside it.
            rng = np.random.default_rng([seed, obj_id, ep_k])

            mech = sample_mechanics(rng, randomize)
            apply_mechanics(model, hinge_id, lid_id, mech, base_mech)
            ep_gt = derived_gt(model, data, hinge_id, handle_id, lid_id)

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

            # Randomized mechanics vary per episode, so they are per-timestep
            # columns rather than static per-object entries.
            for k_, v_ in mech.items():
                ep[f"gt_{k_}"] = np.full(T, np.nan if v_ is None else v_)
            for k_, v_ in ep_gt.items():
                ep[f"gt_{k_}"] = np.full(T, v_)
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
                "mechanics": {k_: v_ for k_, v_ in mech.items()},
                **{k_: round(v_, 6) for k_, v_ in ep_gt.items()},
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
    randomized = set(MECHANICS) | {"lid_mass", "inertia_about_hinge",
                                   "lever_arm", "effective_mass_at_handle"}
    gt_flat = {f"gt/{n}/{k}": v for n, g in gts.items()
               for k, v in g.items() if k not in randomized}
    np.savez_compressed(path, **ds, **gt_flat,
                        objects=np.array(objects, dtype=object))

    with open(f"datasets/{tag}_schema.json", "w") as f:
        json.dump(schema(objects, episodes, gts, seed, randomize), f, indent=2, default=str)

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


def schema(objects, episodes, gts, seed, randomize):
    return {
        "frames": {
            "reference": "world == robot_base",
            "quat": "wxyz",
            "twist": "linear3_angular3",
        },
        "units": {
            "length": "m", "angle": "rad", "time": "s", "force": "N",
            "torque": "Nm", "mass": "kg", "inertia": "kg m^2",
            "stiffness_pos": "N/m", "stiffness_ori": "Nm/rad",
        },
        "controller": {
            "type": "cartesian_impedance_nullspace_posture",
            "Kp_pos": KP_POS, "Kp_ori": KP_ORI, "Kp_null": KP_NULL.tolist(),
            "damping_ratio": DAMPING_RATIO, "kpos": KPOS, "kori": KORI,
            "integration_dt": INTEGRATION_DT, "dt": DT,
            "posture_reference": "closed_door_keyframe",
        },
        "episode": {
            "segment_duration": SEGMENT_DURATION, "settle_vel": SETTLE_VEL,
            "settle_window": SETTLE_WINDOW, "max_phase_time": MAX_PHASE_TIME,
            "near_limit_frac": NEAR_LIMIT_FRAC, "jump_sigma": JUMP_SIGMA,
        },
        "derived": {
            "weld_wrench_ee": "pinv(J_arm.T) @ qfrc_constraint[arm]",
        },
        "mechanics": {
            "randomized": randomize,
            "seed": seed,
            "seed_scheme": "default_rng([seed, object_id, episode_index])",
            "ranges": {k: list(v) for k, v in MECHANICS.items()},
            "log_uniform": sorted(LOG_UNIFORM),
            "per_timestep_columns": "gt_*",
            "static_columns": "gt/<object>/* (geometry only)",
        },
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
    p.add_argument("--no-randomize", action="store_true",
                   help="keep each object's stock mechanics")
    a = p.parse_args()
    collect(a.objects, a.episodes, a.seed, a.tag, not a.no_randomize)


if __name__ == "__main__":
    main()
