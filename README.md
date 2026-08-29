# iiwa_playground

## Build scenes

```bash
python3.10 scenes/build_scene.py --list            # detection status per object
python3.10 scenes/build_scene.py kitchen-cabinet   # one object
python3.10 scenes/build_scene.py --all             # all objects
```

Builds `scenes/<object>_iiwa_scene.xml` (+ `data/<object>/object_static.xml`) for
objects in `data/`. 11 of 15 build; `locker`, `medicine-cabinet`, `microwave` and
`toilet` fail handle detection and need an entry in `HANDLE_OVERRIDES`.

## Run

```bash
mjpython impedence_controller.py                   # free-space, no object
mjpython articulation_controller.py                # kitchen-cabinet
mjpython articulation_controller.py toolbox        # any built object
```

Use `mjpython` on macOS, `python3.10` elsewhere.

## Output

Figures and `run_log.npz` land in `plots/<run name>/`.

## Viewer

Double-click the target to select it, then `ctrl` + right-drag to move it.
Plain drag / scroll control the camera.
