import math
from typing import Any, Dict, Iterable, Optional

import torch

from .common import CifarMuonBase, zeropower_via_newtonschulz5


class CifarMuonEq(CifarMuonBase):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 5e-3,
        momentum: float = 0.9,
        nesterov: bool = True,
        ns_steps: int = 5,
        wd: float = 0.0,
        adamw_betas: tuple[float, float] = (0.95, 0.95),
        adamw_eps: float = 1e-8,
        rowcol_scale_exponent: float = -0.5,
        rowcol_eps: float = 1e-8,
        rowcol_clip: Optional[float] = None,
        row_norm: str = "l2",
        col_norm: str = "l2",
        normalize_mode: str = "rowcol",
        phase: Optional[int] = None,
        use_muonplus: bool = False,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            wd=wd,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            rowcol_scale_exponent=rowcol_scale_exponent,
            rowcol_eps=rowcol_eps,
            rowcol_clip=rowcol_clip,
            row_norm=row_norm,
            col_norm=col_norm,
            normalize_mode=self._canonicalize_normalize_mode(normalize_mode),
            phase=phase,
            use_muonplus=use_muonplus,
        )

    @staticmethod
    def _canonicalize_normalize_mode(normalize_mode: str) -> str:
        aliases = {
            "rowcol": "rowcol",
            "row-col": "rowcol",
            "row_col": "rowcol",
            "row": "row",
            "col": "col",
            "column": "col",
        }
        canonical_mode = aliases.get(str(normalize_mode).lower())
        if canonical_mode is None:
            raise ValueError(
                f"Unsupported CifarMuonEq normalize_mode={normalize_mode!r}. "
                "Expected one of: rowcol, row, col."
            )
        return canonical_mode

    def _begin_muon_group(self, group: Dict[str, Any]) -> Dict[str, Any]:
        has_muon_params = any(self.state[param]["use_muon"] for param in group["params"])
        if not has_muon_params:
            return {"muon_step": int(group.get("muon_step", 0))}

        muon_step = int(group.get("muon_step", 0)) + 1
        group["muon_step"] = muon_step
        return {"muon_step": muon_step}

    @torch.no_grad()
    def _apply_scale_exponent_(self, scale: torch.Tensor, scale_exp: float, eps: float) -> None:
        scale.add_(eps)
        if scale_exp == -0.25:
            scale.rsqrt_().sqrt_()
        elif scale_exp == -0.5:
            scale.rsqrt_()
        else:
            scale.pow_(scale_exp)

    @torch.no_grad()
    def _clamp_scale_(self, scale: torch.Tensor, clip: Optional[float]) -> None:
        if clip is None:
            return
        inv_clip = 1.0 / clip
        scale.clamp_(min=inv_clip, max=clip)

    @torch.no_grad()
    def _resolve_norm(self, group: Dict[str, Any], key: str) -> str:
        norm_value = group.get(key)
        if norm_value is None and "rowcol_norm" in group:
            norm_value = group["rowcol_norm"]
        norm = str(norm_value if norm_value is not None else "l2").lower()
        if norm not in {"l2", "inf"}:
            raise ValueError(f"Unsupported {key}={norm!r}; expected 'l2' or 'inf'")
        return norm

    @torch.no_grad()
    def _compute_row_scale(self, matrix: torch.Tensor, group: Dict[str, Any]) -> torch.Tensor:
        if self._resolve_norm(group, "row_norm") == "inf":
            return matrix.abs().amax(dim=1)
        return matrix.square().sum(dim=1)

    @torch.no_grad()
    def _compute_col_scale(self, matrix: torch.Tensor, group: Dict[str, Any]) -> torch.Tensor:
        if self._resolve_norm(group, "col_norm") == "inf":
            return matrix.abs().amax(dim=0)
        return matrix.square().sum(dim=0)

    @torch.no_grad()
    def _parallel_rowcol_normalize(self, matrix: torch.Tensor, group: Dict[str, Any]) -> torch.Tensor:
        row_scale = self._compute_row_scale(matrix, group)
        col_scale = self._compute_col_scale(matrix, group)
        scale_exp = float(group.get("rowcol_scale_exponent", -0.25))
        self._apply_scale_exponent_(row_scale, scale_exp, float(group["rowcol_eps"]))
        self._apply_scale_exponent_(col_scale, scale_exp, float(group["rowcol_eps"]))
        self._clamp_scale_(row_scale, group["rowcol_clip"])
        self._clamp_scale_(col_scale, group["rowcol_clip"])

        normalized = matrix * row_scale[:, None]
        normalized.mul_(col_scale[None, :])
        return normalized

    @torch.no_grad()
    def _row_normalize(self, matrix: torch.Tensor, group: Dict[str, Any]) -> torch.Tensor:
        scale_exp = float(group.get("rowcol_scale_exponent", -0.25))
        row_scale = self._compute_row_scale(matrix, group)
        self._apply_scale_exponent_(row_scale, scale_exp, float(group["rowcol_eps"]))
        self._clamp_scale_(row_scale, group["rowcol_clip"])
        return matrix * row_scale[:, None]

    @torch.no_grad()
    def _col_normalize(self, matrix: torch.Tensor, group: Dict[str, Any]) -> torch.Tensor:
        scale_exp = float(group.get("rowcol_scale_exponent", -0.25))
        col_scale = self._compute_col_scale(matrix, group)
        self._apply_scale_exponent_(col_scale, scale_exp, float(group["rowcol_eps"]))
        self._clamp_scale_(col_scale, group["rowcol_clip"])
        return matrix * col_scale[None, :]

    @torch.no_grad()
    def _resolve_normalize_mode(self, group: Dict[str, Any], step: int) -> str:
        normalize_mode = self._canonicalize_normalize_mode(group.get("normalize_mode", "rowcol"))
        phase = group.get("phase")
        if phase is not None and normalize_mode == "rowcol" and step >= int(phase):
            return "row"
        return normalize_mode

    @torch.no_grad()
    def _normalize_muon_update(self, matrix: torch.Tensor, group: Dict[str, Any], step: int) -> torch.Tensor:
        normalize_mode = self._resolve_normalize_mode(group, step)
        if normalize_mode == "row":
            return self._row_normalize(matrix, group)
        if normalize_mode == "col":
            return self._col_normalize(matrix, group)
        return self._parallel_rowcol_normalize(matrix, group)

    def _build_muon_update(
        self,
        param: torch.Tensor,
        grad_2d: torch.Tensor,
        group: Dict[str, Any],
        muon_group_state: Dict[str, Any],
    ) -> torch.Tensor:
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad_2d)

        buf = state["momentum_buffer"]
        momentum = float(group["momentum"])
        buf.mul_(momentum).add_(grad_2d)
        update = grad_2d.add(buf, alpha=momentum) if group["nesterov"] else buf
        return self._normalize_muon_update(update, group, int(muon_group_state["muon_step"]))

    def _muon_update(
        self,
        param: torch.Tensor,
        grad_2d: torch.Tensor,
        group: Dict[str, Any],
        muon_group_state: Dict[str, Any],
    ) -> torch.Tensor:
        if bool(group.get("use_muonplus", False)):
            return self._muonplus_update(param, grad_2d, group, muon_group_state)
        update = self._build_muon_update(param, grad_2d, group, muon_group_state)
        return zeropower_via_newtonschulz5(update, steps=int(group["ns_steps"]))

    def _muonplus_update(
        self,
        param: torch.Tensor,
        grad_2d: torch.Tensor,
        group: Dict[str, Any],
        muon_group_state: Dict[str, Any],
    ) -> torch.Tensor:
        step = int(muon_group_state["muon_step"])
        update = self._build_muon_update(param, grad_2d, group, muon_group_state)
        polar_update = zeropower_via_newtonschulz5(update, steps=int(group["ns_steps"]))
        # Treat the flattened column dimension as fan-in for Muon+ scaling.
        d_in = max(1, int(grad_2d.size(1)))
        return self._normalize_muon_update(polar_update, group, step) / math.sqrt(d_in)
