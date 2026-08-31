"""Second-difference dynamics model: w=[q, dq] -> d2 = dq' - dq.

g_theta(w, u, x) -> w_hat'. Checkpoint format matches the original dynamics.py
(same dict keys), so files saved here load with the old loader and vice versa.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OBS_DIM = 2
ACTION_DIM = 1
BELIEF_DIM = 8
FORMULATION = "second_difference"

CHECKPOINTS = Path(__file__).resolve().parent / "checkpoints"


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.SiLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class SecondDifferenceModel(nn.Module):
    formulation = FORMULATION
    out_dim = 1
    canonical = False  # set by `load` from checkpoint meta

    def __init__(self, belief_dim: int = BELIEF_DIM,
                 hidden: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        self.belief_dim = belief_dim
        self.net = _mlp(OBS_DIM + ACTION_DIM + belief_dim, hidden, self.out_dim)
        for group, dim in (("obs", OBS_DIM), ("action", ACTION_DIM),
                           ("delta", self.out_dim)):
            self.register_buffer(f"{group}_mean", torch.zeros(dim))
            self.register_buffer(f"{group}_std", torch.ones(dim))

    def set_norm(self, obs, action, delta) -> None:
        for name, arr in (("obs", obs), ("action", action), ("delta", delta)):
            a = np.asarray(arr, dtype=np.float32)
            getattr(self, f"{name}_mean").copy_(torch.tensor(a.mean(0)))
            getattr(self, f"{name}_std").copy_(
                torch.tensor(a.std(0)).clamp_min(1e-8))

    def normalized_delta(self, w, u, x) -> torch.Tensor:
        w, u, x = _broadcast(w, u, x)
        z = torch.cat([(w - self.obs_mean) / self.obs_std,
                       (u - self.action_mean) / self.action_std,
                       x], dim=-1)
        return self.net(z)

    def normalize_d2(self, d2) -> torch.Tensor:
        return (d2 - self.delta_mean) / self.delta_std

    def target(self, w, w_next) -> torch.Tensor:
        """Fallback target if you only have (w, w_next) and not a clean d2."""
        return self.normalize_d2(w_next[..., 1:2] - w[..., 1:2])

    def forward(self, w, u, x) -> torch.Tensor:
        d = self.normalized_delta(w, u, x)               # (N,1) normalised d2
        w = torch.broadcast_to(w, (*d.shape[:-1], OBS_DIM))
        dq_next = w[..., 1:2] + d * self.delta_std + self.delta_mean
        return torch.cat([w[..., 0:1] + dq_next, dq_next], dim=-1)

    @torch.no_grad()
    def predict(self, w: np.ndarray, u: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self(_t(w, OBS_DIM), _t(u, ACTION_DIM),
                    _t(x, self.belief_dim))[0].numpy()

    @torch.no_grad()
    def predict_normalized_batch(self, w: np.ndarray, u: np.ndarray,
                                 X: np.ndarray) -> np.ndarray:
        """h(x) for a batch of beliefs (e.g. UKF sigma points), normalised out."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        n = len(X)
        return self.normalized_delta(
            _t(w, OBS_DIM).expand(n, -1),
            _t(u, ACTION_DIM).expand(n, -1),
            torch.as_tensor(X),
        ).numpy().astype(np.float64)

    @torch.no_grad()
    def measurement(self, w: np.ndarray, w_next: np.ndarray) -> np.ndarray:
        return self.target(_t(w, OBS_DIM),
                           _t(w_next, OBS_DIM))[0].numpy().astype(np.float64)

    def freeze(self) -> "SecondDifferenceModel":
        for p in self.parameters():
            p.requires_grad_(False)
        return self.eval()

    def checksum(self) -> float:
        with torch.no_grad():
            return float(sum(p.double().abs().sum() for p in self.parameters()))


def _t(a, dim: int) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float32)).reshape(1, dim)


def _broadcast(w, u, x):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.shape[0] == 1 and w.shape[0] != 1:
        x = x.expand(w.shape[0], -1)
    if u.dim() == 1:
        u = u.unsqueeze(-1)
    return w, u, x


def resolve(path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    for cand in (CHECKPOINTS / p.name, CHECKPOINTS.parent / p.name):
        if cand.exists():
            return cand
    have = (sorted(q.name for q in CHECKPOINTS.glob("*.pt"))
            if CHECKPOINTS.exists() else [])
    raise FileNotFoundError(f"checkpoint {path!r} not found. Available: {have or 'none'}")


def save(path, model: SecondDifferenceModel, beliefs: np.ndarray, meta: dict) -> Path:
    path = Path(path)
    if path.parent == Path("."):
        path = CHECKPOINTS / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "belief_dim": model.belief_dim,
                "formulation": model.formulation,
                "train_beliefs": np.asarray(beliefs), "meta": meta}, path)
    return path


def load(path) -> tuple[SecondDifferenceModel, np.ndarray, dict]:
    """Loads only second_difference checkpoints. For state_delta checkpoints,
    use the original dynamics.py loader instead."""
    ck = torch.load(resolve(path), map_location="cpu", weights_only=False)
    kind = ck.get("formulation", "state_delta")
    if kind != FORMULATION:
        raise ValueError(
            f"checkpoint {path!r} is formulation {kind!r}; this loader only "
            f"handles {FORMULATION!r}.")
    meta = ck.get("meta", {})
    model = SecondDifferenceModel(belief_dim=ck["belief_dim"])
    model.load_state_dict(ck["state"])
    model.canonical = bool(meta.get("canonical", False))
    return model.freeze(), np.asarray(ck["train_beliefs"]), meta