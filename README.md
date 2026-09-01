# iiwa_playground

## Requirements

```bash
pip install dual_quaternions
mjpython -m pip install dual_quaternions
```

Download the [Articulated Hinge Dataset](https://huggingface.co/datasets/leejason2025/njc-articulated-hinge) and add its contents under `data/` folder such that it's data/kitchen-cabinet or data/cooler, etc.

## Build scenes

```bash
python3 scenes/build_scene.py --list            # detection status per object
python3 scenes/build_scene.py kitchen-cabinet   # one object
python3 scenes/build_scene.py --all             # all objects
```

Builds `scenes/<object>_iiwa_scene.xml` (+ `data/<object>/object_static.xml`) for
objects in `data/`. 11 of 15 build; `locker`, `medicine-cabinet`, `microwave` and
`toilet` fail handle detection and need an entry in `HANDLE_OVERRIDES`.

Scenes are built with the arm base at the world origin.

## Reachability survey

```bash
python3 reachability_survey.py kitchen-cabinet           # one object
python3 reachability_survey.py kitchen-cabinet --report  # + sigma_min spread
```

Samples a shell of EE targets around the handle at several hinge angles and
scores each: IK reachable, collision-free, and away from singularity
(`sigma_min`). Writes `plots/<object>/reachability.npz` + one PNG per angle.

## Run

```bash
mjpython impedence_controller.py                   # free-space, no object
mjpython articulation_controller.py                # kitchen-cabinet
mjpython articulation_controller.py toolbox        # any built object
```

Use `mjpython` on macOS, `python3` elsewhere.

## Collect a dataset

```bash
python3 collect.py                                 # all subset objects
python3 collect.py --objects safe -n 5 --tag test  # one object, 5 episodes
```

Runs episodes from randomized hinge start angles toward survey targets, logging
one record per timestep. Writes `datasets/<tag>.npz` + `<tag>_schema.json`.
Ground truth is under `gt/<object>/*`; validity flags are `flag_*`.

## Output

Figures and `run_log.npz` land in `plots/<run name>/`; datasets in `datasets/`.

## Controller (`utils/impedance.py`)

`CartesianImpedance` is the single source for the control law — the three
scripts above all use it. Works welded or free: the Jacobian spans all `nv`
DOFs, only the 7 arm DOFs are commanded.

```python
from utils.impedance import CartesianImpedance

ctrl = CartesianImpedance(model, kp_pos=50.0, kp_ori=25.0,
                          kp_null=[75, 75, 50, 50, 40, 25, 25])
ctrl.capture_posture(data)          # nullspace reference = current arm config

while running:
    out = ctrl.compute(data, pos_des, quat_des)   # does not write ctrl or step
    ctrl.apply(data, out)                         # writes data.ctrl
    mujoco.mj_step(model, data)
```

`compute()` returns a `ControlOutput` carrying the intermediates worth logging,
so callers never recompute them:


| field                               | meaning                               |
| ----------------------------------- | ------------------------------------- |
| `tau`, `tau_unclipped`, `saturated` | commanded torque, pre-clip, clip flag |
| `dx`, `dtheta`                      | position / orientation error          |
| `ee_pos`, `ee_quat`, `ee_twist`     | measured EE pose and twist            |
| `jac`                               | `(6, nv)` site Jacobian               |


Optional: `damping_ratio`, `kpos`, `kori`, `integration_dt`,
`gravity_compensation`, `track_orientation`, `site_name`, `joint_names`.
Set `ctrl.q0` directly to use a posture reference other than the current pose
(e.g. a keyframe). Damping is derived as `2 * damping_ratio * sqrt(Kp)`.

## Viewer

Double-click the target to select it, then `ctrl` + right-drag to move it.
Plain drag / scroll control the camera.