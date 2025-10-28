import numpy as np

from .vector_specs import FVSpec


def _pick_idx(n: int, where: str) -> int:
    if where == "e":
        return n - 1
    if where == "s":
        return 0
    if where == "m":
        return n // 2
    raise ValueError("where must be one of 'e', 's', 'm'")


def vectorize(traj: dict, spec: FVSpec) -> np.ndarray:
    n = len(traj["x"])

    def xy(idx):
        return np.array([traj["x"][idx], traj["y"][idx]], dtype=np.float32)

    fv = []
    if spec.use_Re_e:
        fv += xy(_pick_idx(n, "e")).tolist()
    if spec.use_Ve_e:
        fv += [float(traj["vx"][_pick_idx(n, "e")]), float(traj["vy"][_pick_idx(n, "e")])]
    if spec.use_Ae_e:
        fv += [float(traj["ax"][_pick_idx(n, "e")]), float(traj["ay"][_pick_idx(n, "e")])]
    if spec.use_Re_s:
        fv += xy(_pick_idx(n, "s")).tolist()
    if spec.use_Re_m:
        fv += xy(_pick_idx(n, "m")).tolist()
    return np.array(fv, dtype=np.float32)
