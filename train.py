"""Train the second-difference dynamics model on any dataset.

Expects a .npz with:
  W   (N, 2) float   -- [q, dq] observed state at each transition
  U   (N, 1) float   -- action (e.g. qfrc_constraint at the joint)
  ID  (N,)   int     -- object id, so each object gets its own belief vector
  and ONE of:
    D2  (N, 1) float -- clean second difference target (preferred if you have it)
    WN  (N, 2) float -- observed next state (target built as WN[:,1]-W[:,1])

Usage:
  python train_dynamics.py --data mydata.npz --out mymodel.pt
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from dynamics_model import BELIEF_DIM, SecondDifferenceModel, save


def load_dataset(path):
    d = np.load(path)
    W = d["W"].astype(np.float32)
    U = d["U"].astype(np.float32).reshape(len(W), 1)
    ID = d["ID"].astype(np.int64) if "ID" in d else np.zeros(len(W), dtype=np.int64)
    D2 = d["D2"].astype(np.float32).reshape(len(W), 1) if "D2" in d else None
    WN = d["WN"].astype(np.float32) if "WN" in d else None
    if D2 is None and WN is None:
        raise ValueError("dataset must contain either 'D2' or 'WN'")
    return W, U, ID, D2, WN


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="path to .npz dataset")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--belief-dim", type=int, default=BELIEF_DIM)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="held-out fraction used for monitoring during training")
    ap.add_argument("--test-frac", type=float, default=0.1,
                    help="held-out fraction touched only once, after training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="model.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.time()

    W, U, ID, D2, WN = load_dataset(args.data)
    n_objects = int(ID.max()) + 1
    print(f"{len(W)} transitions, {n_objects} object(s), belief dim {args.belief_dim}")

    model = SecondDifferenceModel(belief_dim=args.belief_dim)
    delta = D2 if D2 is not None else (WN[:, 1:2] - W[:, 1:2])
    model.set_norm(W, U, delta)
    beliefs = torch.nn.Parameter(torch.randn(n_objects, args.belief_dim) * 0.1)

    opt = torch.optim.Adam([
        {"params": model.parameters(), "lr": 1e-3, "weight_decay": 1e-5},
        {"params": [beliefs], "lr": 1e-2, "weight_decay": 1e-4},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Wt, Ut, IDt = (torch.tensor(a) for a in (W, U, ID))
    D2t = torch.tensor(delta, dtype=torch.float32)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(Wt), generator=g)
    n_val = int(len(Wt) * args.val_frac)
    n_test = int(len(Wt) * args.test_frac)
    val, test, tr = perm[:n_val], perm[n_val:n_val + n_test], perm[n_val + n_test:]

    def rmse_on(idx) -> float:
        model.eval()
        with torch.no_grad():
            tgt = model.normalize_d2(D2t[idx])
            pred = model.normalized_delta(Wt[idx], Ut[idx], beliefs[IDt[idx]])
            return float(torch.sqrt(torch.nn.functional.mse_loss(pred, tgt)))

    print(f"Training {args.epochs} epochs on {len(tr)} transitions "
          f"(val {len(val)}, test {len(test)})...")
    for ep in range(args.epochs):
        model.train()
        order = tr[torch.randperm(len(tr), generator=g)]
        total = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i: i + args.batch_size]
            tgt = model.normalize_d2(D2t[idx])
            pred = model.normalized_delta(Wt[idx], Ut[idx], beliefs[IDt[idx]])
            loss = torch.nn.functional.mse_loss(pred, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
        sched.step()

        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  train {total / len(order):.5f}  "
                  f"val RMSE {rmse_on(val):.3e}")

    test_rmse = rmse_on(test)
    print(f"Test RMSE (touched once, after training): {test_rmse:.3e}")

    z = beliefs.detach().numpy()
    meta = {
        "n_objects": n_objects, "n_transitions": int(len(W)),
        "epochs": args.epochs, "seed": args.seed, "belief_dim": args.belief_dim,
        "formulation": "second_difference", "source": args.data,
        "val_frac": args.val_frac, "test_frac": args.test_frac,
        "test_rmse": test_rmse,
    }
    path = save(args.out, model, z, meta)
    print(f"Saved {path} ({path.stat().st_size / 1e3:.0f} kB) in {time.time() - t0:.0f}s")
    print(f"  belief spread: {np.linalg.norm(z - z.mean(0), axis=1).mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())