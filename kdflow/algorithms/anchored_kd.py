"""
Anchored Learning KD algorithm for KDFlow.

This file implements the paper "Stabilizing LLM Supervised Fine-Tuning via
Explicit Distributional Control" inside KDFlow's off-policy distillation path.

Training stages:
  1) Train a fixed SFT reference model p_sft with KDFlow SFT.
  2) Start a fresh student from p_base and use this algorithm with teacher=p_sft.
  3) Every `anchor_inner_epochs` epochs, refresh p_theta^(t) snapshot.
  4) Minimize KL(q_anchor || p_theta) on response tokens.

The algorithm is intentionally implemented as a KDFlow plugin so it is picked up
by kdflow.algorithms.__init__ automatically.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from kdflow.algorithms import register_algorithm
from kdflow.loss.chunked_loss import chunked_loss
from kdflow.loss.cross_entropy import compute_cross_entropy


def _call_lm_head(head: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Call KDFlow-patched lm_head or a vanilla nn.Linear lm_head."""
    try:
        return head(hidden, skip=False)
    except TypeError:
        return head(hidden)


@register_algorithm("anchored_kd")
class AnchoredKD:
    """Anchored Learning objective for KDFlow.

    q_anchor is built from a frozen outer-loop snapshot p_theta^(t) and the fixed
    SFT teacher p_sft:

      logit:       z_q = (1 - alpha) * z_snapshot + alpha * z_sft
      probability: q   = (1 - alpha) * softmax(z_snapshot) + alpha * softmax(z_sft)

    The student minimizes KL(q_anchor || p_student) over response tokens.
    """

    def __init__(self, strategy, student_model, teacher_lm_head, **kwargs):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head

        if isinstance(self.teacher_lm_head, dict):
            raise ValueError(
                "`anchored_kd` currently expects one fixed SFT reference teacher. "
                "Please use --teacher_name_or_path instead of --multi_teacher_config."
            )

        self.alpha = float(getattr(self.args.kd, "anchor_alpha", 0.5))
        self.interpolation = str(getattr(self.args.kd, "anchor_interpolation", "logit")).lower()
        self.temperature = float(
            getattr(self.args.kd, "anchor_temperature", None)
            or getattr(self.args.kd, "kd_temperature", 1.0)
        )
        self.snapshot_mode = str(getattr(self.args.kd, "anchor_snapshot_mode", "model")).lower()
        self.outer_idx = 0
        self.snapshot = None

        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"anchor_alpha must be in [0, 1], got {self.alpha}.")
        if self.interpolation not in ("logit", "probability"):
            raise ValueError("anchor_interpolation must be 'logit' or 'probability'.")
        if self.temperature <= 0:
            raise ValueError(f"anchor_temperature must be > 0, got {self.temperature}.")
        if self.snapshot_mode not in ("model", "detached_current"):
            raise ValueError("anchor_snapshot_mode must be 'model' or 'detached_current'.")

        if self.snapshot_mode == "model":
            self._try_build_snapshot_model()

    def _try_build_snapshot_model(self) -> None:
        """Create a frozen in-memory model snapshot for exact outer-loop anchoring.

        Some distributed wrappers may not support deepcopy in all environments.
        In that case users can either set --anchor_snapshot_mode detached_current
        or use a KDFlow backend/strategy that permits the extra model copy.
        """
        try:
            self.snapshot = copy.deepcopy(self.student)
            self.snapshot.eval()
            self.snapshot.requires_grad_(False)
            self.strategy.print(
                "[anchored_kd] Built frozen outer snapshot model. "
                "Call refresh_anchor_snapshot() at each outer-loop boundary."
            )
        except Exception as exc:  # pragma: no cover - depends on runtime backend
            self.snapshot = None
            self.snapshot_mode = "detached_current"
            self.strategy.print(
                "[anchored_kd] WARNING: failed to deepcopy the student for exact "
                f"outer snapshot ({type(exc).__name__}: {exc}). Falling back to "
                "`detached_current`, which detaches current logits per step."
            )

    @torch.no_grad()
    def refresh_anchor_snapshot(self, outer_idx: int = 0) -> Dict[str, object]:
        """Refresh p_theta^(t) at the beginning of an outer iteration.

        This method is called by the patched OffPolicyKDTrainer every
        `anchor_inner_epochs` epochs.
        """
        self.outer_idx = int(outer_idx)

        if self.snapshot_mode != "model" or self.snapshot is None:
            return {
                "anchor_snapshot_refreshed": False,
                "mode": self.snapshot_mode,
                "outer_idx": self.outer_idx,
            }

        copied_params = 0
        skipped_params = 0
        snapshot_params = dict(self.snapshot.named_parameters())
        for name, param in self.student.named_parameters():
            target = snapshot_params.get(name)
            if target is None or target.shape != param.shape:
                skipped_params += 1
                continue
            target.data.copy_(param.detach().data)
            copied_params += 1

        copied_buffers = 0
        snapshot_buffers = dict(self.snapshot.named_buffers())
        for name, buf in self.student.named_buffers():
            target = snapshot_buffers.get(name)
            if target is not None and target.shape == buf.shape:
                target.data.copy_(buf.detach().data)
                copied_buffers += 1

        self.snapshot.eval()
        return {
            "anchor_snapshot_refreshed": True,
            "mode": self.snapshot_mode,
            "outer_idx": self.outer_idx,
            "copied_params": copied_params,
            "skipped_params": skipped_params,
            "copied_buffers": copied_buffers,
        }

    @torch.no_grad()
    def _snapshot_logits_fn(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        student_hiddens: torch.Tensor,
        mm_kwargs: Dict,
    ) -> Callable[[int, int], torch.Tensor]:
        """Return a chunked logits function for p_theta^(t)."""
        if self.snapshot_mode == "model" and self.snapshot is not None:
            snapshot_output = self.snapshot(
                input_ids,
                attention_mask=attention_mask,
                allgather_logits=True,
                ring_attn_group=self.strategy.ring_attn_group,
                **mm_kwargs,
            )
            snapshot_hiddens = snapshot_output["hidden_states"][-1][loss_mask].detach()
            del snapshot_output

            def _fn(start: int, end: int) -> torch.Tensor:
                return _call_lm_head(self.snapshot.model.lm_head, snapshot_hiddens[start:end]).detach()

            return _fn

        detached_hiddens = student_hiddens.detach()

        def _fn(start: int, end: int) -> torch.Tensor:
            return _call_lm_head(self.student.model.lm_head, detached_hiddens[start:end]).detach()

        return _fn

    def _anchor_kl_loss(
        self,
        student_logits: torch.Tensor,
        snapshot_logits: torch.Tensor,
        sft_logits: torch.Tensor,
        reduction: str = "none",
    ) -> torch.Tensor:
        """Compute tokenwise KL(q_anchor || p_student)."""
        vocab_size = min(student_logits.shape[-1], snapshot_logits.shape[-1], sft_logits.shape[-1])
        student_logits = student_logits[..., :vocab_size]
        snapshot_logits = snapshot_logits[..., :vocab_size].detach()
        sft_logits = sft_logits[..., :vocab_size].detach()

        t = self.temperature
        student_log_probs = F.log_softmax(student_logits / t, dim=-1, dtype=torch.float32)

        if self.interpolation == "logit":
            anchor_logits = (1.0 - self.alpha) * snapshot_logits + self.alpha * sft_logits
            anchor_probs = F.softmax(anchor_logits / t, dim=-1, dtype=torch.float32)
        else:
            snapshot_probs = F.softmax(snapshot_logits / t, dim=-1, dtype=torch.float32)
            sft_probs = F.softmax(sft_logits / t, dim=-1, dtype=torch.float32)
            anchor_probs = (1.0 - self.alpha) * snapshot_probs + self.alpha * sft_probs

        # Hinton KD convention scales KL by T^2 to keep gradient magnitude stable.
        token_kl = F.kl_div(student_log_probs, anchor_probs, reduction="none").sum(dim=-1) * (t ** 2)

        if reduction == "none":
            return token_kl
        if reduction == "sum":
            return token_kl.sum()
        if reduction == "mean":
            return token_kl.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")

    def training_step(self, micro_batch):
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        teacher_hiddens = micro_batch.get("teacher_hiddens", None)
        teacher_loss_mask = micro_batch["tea_loss_mask"].bool()
        avg_token_num = micro_batch["avg_micro_batch_token_num"]

        assert teacher_hiddens is not None, "micro_batch must contain `teacher_hiddens` for anchored KD"

        mm_kwargs = micro_batch.get("stu_multi_modal_inputs") or {}

        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **mm_kwargs,
        )
        student_hiddens = output["hidden_states"][-1][student_loss_mask]
        del output

        teacher_token_num = int(teacher_loss_mask.sum().item())
        student_token_num = int(student_loss_mask.sum().item())
        if teacher_token_num != student_token_num:
            raise ValueError(
                "anchored_kd currently requires identical student/teacher tokenization "
                f"on loss tokens. Got student={student_token_num}, teacher={teacher_token_num}."
            )

        # Non-chunked case is a special case of chunked loss.
        chunk_size = self.args.train.chunked_loss_size or student_hiddens.shape[0]

        # p_sft logits from the KDFlow teacher hidden cache + fixed teacher lm_head.
        teacher_hiddens = teacher_hiddens.to(next(self.teacher_lm_head.parameters()).device)
        sft_logits_fn = lambda start, end: _call_lm_head(
            self.teacher_lm_head, teacher_hiddens[start:end]
        ).detach()

        # p_theta^(t) logits from the frozen outer snapshot, or a detached current approximation.
        snapshot_logits_fn = self._snapshot_logits_fn(
            input_ids=student_input_ids,
            attention_mask=student_attn_mask,
            loss_mask=student_loss_mask,
            student_hiddens=student_hiddens,
            mm_kwargs=mm_kwargs,
        )

        total_loss = student_hiddens.new_zeros(())
        total_tokens = 0
        for start in range(0, student_hiddens.shape[0], chunk_size):
            end = min(start + chunk_size, student_hiddens.shape[0])
            student_logits = _call_lm_head(self.student.model.lm_head, student_hiddens[start:end])
            snapshot_logits = snapshot_logits_fn(start, end).to(student_logits.device)
            sft_logits = sft_logits_fn(start, end).to(student_logits.device)
            token_loss = self._anchor_kl_loss(
                student_logits=student_logits,
                snapshot_logits=snapshot_logits,
                sft_logits=sft_logits,
                reduction="none",
            )
            total_loss = total_loss + token_loss.sum()
            total_tokens += token_loss.numel()

        kd_loss = total_loss / avg_token_num
        loss_info = {
            "loss": kd_loss,
            "kd_loss": kd_loss,
            "anchor_kd_loss": kd_loss,
            "anchor_outer_idx": student_hiddens.new_tensor(float(self.outer_idx)),
            "anchor_alpha": student_hiddens.new_tensor(float(self.alpha)),
        }

        # Optional CE mix for debugging or warm-starting; paper-style Anchored Learning
        # should use --kd_ratio 1.0.
        kd_ratio = float(getattr(self.args.kd, "kd_ratio", 1.0))
        if kd_ratio < 1.0:
            student_label_ids = student_input_ids.roll(shifts=-1, dims=1)[student_loss_mask]
            ce_loss = chunked_loss(
                student_hiddens,
                self.student.model.lm_head,
                compute_cross_entropy,
                label=student_label_ids,
                chunk_size=chunk_size,
                reduction="sum",
            ) / avg_token_num
            loss = (1.0 - kd_ratio) * ce_loss + kd_ratio * kd_loss
            loss_info["loss"] = loss
            loss_info["ce_loss"] = ce_loss

        return loss_info
