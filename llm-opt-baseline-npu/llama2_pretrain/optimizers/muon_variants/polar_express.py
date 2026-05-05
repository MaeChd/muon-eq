from itertools import repeat
import os
import uuid
import warnings

import torch


# How to generate these lists:
# from itertools import islice
# from matsign.methods import OursFixedL, Ours
# hs = list(OursFixedL(l=1e-3, cushion=1e-1, center_squred_svs=False, max_iters=10)(1e-3))
# hs = list(islice(Ours(cushion=1e-1, center_squred_svs=False).uncentered_sequence(1e-3), 10))
# [tuple(float(x) for x in h.coef) for h in hs]

coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
coeffs_list = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_list[:-1]
] + [coeffs_list[-1]]


@torch.compile
def PolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    hs = coeffs_list[:steps] + list(repeat(coeffs_list[-1], steps - len(coeffs_list)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def FastApplyPolarExpress(
    G: torch.Tensor, steps: int, restart_interval: int, shift_eps: float = 0
) -> torch.Tensor:
    assert G.ndim >= 2

    X = G.double()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-7)
    hs = coeffs_list[:steps] + list(repeat(coeffs_list[-1], steps - len(coeffs_list)))
    hs = [(a * 0.99, b * 0.99, c * 0.99) for (a, b, c) in hs]
    I = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
    Y = X @ X.mT + shift_eps * I
    Q = I.clone()
    for iter, (a, b, c) in enumerate(hs):
        if (iter % restart_interval == 0) and (iter > 0):
            X = Q @ X
            Y = X @ X.mT
            Q = I.clone()
        R = Q.mT @ Y @ Q
        Q = Q @ (a * I + R @ (b * I + c * R))
    X = Q @ X
    if (X.norm(dim=(-2, -1), keepdim=False) > 5 * I.shape[0]).any() or not torch.isfinite(X).all():
        warnings.warn("X.norm() is unusually large. Saving G to disk.")
        os.makedirs("bad_G", exist_ok=True)
        filename = f"bad_G_{uuid.uuid4().hex}.pt"
        torch.save(G, os.path.join("bad_G", filename))
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
