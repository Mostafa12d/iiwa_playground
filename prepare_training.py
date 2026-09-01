"""Convert a collect.py dataset into the W/U/ID/D2 format train.py expects.

    python3.10 prepare_training.py datasets/v1.npz

  W  (N,2)  [hinge_q, hinge_qvel]
  U  (N,1)  hinge qfrc_constraint
  ID (N,)   object id
  D2 (N,1)  hinge_qvel[t+1] - hinge_qvel[t]

Transitions that would straddle an episode boundary are dropped, as are steps
flagged diverged.
"""

import argparse

import numpy as np


def build(path, drop_flagged=True, id_by="episode"):
    d = np.load(path, allow_pickle=True)
    q = d["hinge_q"].astype(np.float64)
    dq = d["hinge_qvel"].astype(np.float64)
    u = d["hinge_qfrc_constraint"].astype(np.float64)
    ep = d["episode_id"]
    # The belief vector is indexed by ID. Mechanics are randomized per episode,
    # so a per-object belief cannot represent them - default to per-episode.
    oid = d["episode_id"] if id_by == "episode" else d["object_id"]

    # A transition needs t and t+1 inside the same episode.
    valid = np.zeros(len(q), dtype=bool)
    valid[:-1] = ep[:-1] == ep[1:]

    dropped_flag = 0
    if drop_flagged:
        bad = d["flag_diverged"].astype(bool)
        bad |= np.r_[bad[1:], False]      # a transition into a bad step is bad
        dropped_flag = int((valid & bad).sum())
        valid &= ~bad

    idx = np.flatnonzero(valid)
    W = np.stack([q[idx], dq[idx]], axis=1).astype(np.float32)
    U = u[idx].reshape(-1, 1).astype(np.float32)
    D2 = (dq[idx + 1] - dq[idx]).reshape(-1, 1).astype(np.float32)
    ID = oid[idx].astype(np.int64)

    print(f"{path}: {len(q)} steps -> {len(idx)} transitions")
    print(f"  dropped {len(q) - len(idx)} "
          f"({len(np.unique(ep))} episode boundaries, {dropped_flag} flagged)")
    print(f"  belief ids {ID.max()+1} ({id_by})  "
          f"min/max rows per id {np.bincount(ID).min()}/{np.bincount(ID).max()}")
    print(f"  q        [{W[:,0].min():.4f}, {W[:,0].max():.4f}] rad")
    print(f"  dq       [{W[:,1].min():.4f}, {W[:,1].max():.4f}] rad/s")
    print(f"  u        [{U.min():.4f}, {U.max():.4f}] Nm")
    print(f"  d2       [{D2.min():.3e}, {D2.max():.3e}] rad/s  "
          f"std {D2.std():.3e}")
    return {"W": W, "U": U, "ID": ID, "D2": D2}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dataset")
    p.add_argument("--out")
    p.add_argument("--keep-flagged", action="store_true")
    p.add_argument("--id-by", choices=("episode", "object"), default="episode",
                   help="what the belief vector is indexed by")
    a = p.parse_args()
    out = a.out or a.dataset.replace(".npz", "_train.npz")
    np.savez_compressed(out, **build(a.dataset, not a.keep_flagged, a.id_by))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
