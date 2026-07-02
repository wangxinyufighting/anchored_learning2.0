#!/usr/bin/env python3
"""Install the Anchored Learning patch into a local KDFlow checkout.

Usage:
  python install_patch.py --repo /path/to/KDFlow

The patch is intentionally small and additive:
  - adds kdflow/algorithms/anchored_kd.py
  - adds Anchored Learning CLI arguments to DistillationArguments
  - lets OffPolicyKDTrainer refresh a frozen outer snapshot every K epochs
  - adds a Ray actor method to refresh the snapshot on each student worker
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ANCHOR_FIELDS = r"""
    # Anchored Learning hyperparameters
    anchor_alpha: float = field(
        default=0.5,
        metadata={"help": "Interpolation coefficient alpha for Anchored Learning."}
    )
    anchor_interpolation: str = field(
        default="logit",
        metadata={
            "help": "Anchor interpolation space: 'logit' or 'probability'.",
            "choices": ["logit", "probability"],
        }
    )
    anchor_inner_epochs: int = field(
        default=5,
        metadata={"help": "Number of inner-loop epochs K before refreshing p_theta^(t)."}
    )
    anchor_snapshot_mode: str = field(
        default="model",
        metadata={
            "help": "How to compute p_theta^(t): 'model' keeps an exact frozen model copy; "
                    "'detached_current' detaches current logits per step as a memory-saving fallback.",
            "choices": ["model", "detached_current"],
        }
    )
    anchor_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature used in Anchored Learning KL."}
    )
"""

ANCHOR_VALIDATION = r"""
        if not 0.0 <= self.anchor_alpha <= 1.0:
            raise ValueError(f"anchor_alpha must be in [0, 1], got {self.anchor_alpha}.")
        if self.anchor_interpolation not in ("logit", "probability"):
            raise ValueError(
                f"anchor_interpolation must be 'logit' or 'probability', got {self.anchor_interpolation}."
            )
        if self.anchor_inner_epochs <= 0:
            raise ValueError(f"anchor_inner_epochs must be > 0, got {self.anchor_inner_epochs}.")
        if self.anchor_snapshot_mode not in ("model", "detached_current"):
            raise ValueError(
                f"anchor_snapshot_mode must be 'model' or 'detached_current', got {self.anchor_snapshot_mode}."
            )
        if self.anchor_temperature <= 0:
            raise ValueError(f"anchor_temperature must be > 0, got {self.anchor_temperature}.")
"""

STUDENT_GROUP_METHOD = r"""
    def async_refresh_anchor_snapshot(self, outer_idx=0):
        # Refresh p_theta^(t) snapshot on all student actors.
        return [
            actor.refresh_anchor_snapshot.remote(outer_idx)
            for actor in self._actor_handlers
        ]

"""

STUDENT_ACTOR_METHOD = r"""
    def refresh_anchor_snapshot(self, outer_idx=0):
        # Refresh the algorithm-owned Anchored Learning outer snapshot.
        if hasattr(self, "kd_algorithm") and hasattr(self.kd_algorithm, "refresh_anchor_snapshot"):
            return self.kd_algorithm.refresh_anchor_snapshot(outer_idx)
        return {"anchor_snapshot_refreshed": False, "outer_idx": int(outer_idx)}

"""

OFF_POLICY_METHOD = r"""
    def _maybe_refresh_anchor_snapshot(self, epoch):
        # Refresh p_theta^(t) at outer-loop boundaries for Anchored Learning.
        if getattr(self.args.kd, "kd_algorithm", None) != "anchored_kd":
            return

        inner_epochs = max(1, int(getattr(self.args.kd, "anchor_inner_epochs", 1)))
        anchor_start_epoch = getattr(self, "_anchor_start_epoch", 0)
        if (epoch - anchor_start_epoch) % inner_epochs != 0:
            return

        outer_idx = (epoch - anchor_start_epoch) // inner_epochs
        self.strategy.log(
            f"[anchored_kd] Refreshing outer snapshot p_theta^({outer_idx}) "
            f"at epoch {epoch + 1}; inner_epochs={inner_epochs}"
        )

        # Student may have been offloaded after initialization or previous epoch.
        should_sleep_after = bool(getattr(self.args.train, "enable_sleep", False))
        if should_sleep_after:
            self.student.wakeup()
        try:
            results = ray.get(self.student.async_refresh_anchor_snapshot(outer_idx))
            if results:
                self.strategy.log(f"[anchored_kd] Snapshot refresh result: {results[0]}")
        finally:
            if should_sleep_after:
                self.student.sleep()

"""


def backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak.anchor")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def copy_algorithm(repo: Path, patch_root: Path) -> None:
    src = patch_root / "kdflow" / "algorithms" / "anchored_kd.py"
    dst = repo / "kdflow" / "algorithms" / "anchored_kd.py"
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[OK] copied {dst.relative_to(repo)}")


def patch_file(path: Path, transform) -> bool:
    text = path.read_text(encoding="utf-8")
    new_text = transform(text)
    if new_text == text:
        print(f"[SKIP] {path}")
        return False
    backup(path)
    path.write_text(new_text, encoding="utf-8")
    print(f"[OK] patched {path}")
    return True


def patch_distillation_args(repo: Path) -> None:
    path = repo / "kdflow" / "arguments" / "distillation_args.py"

    def transform(text: str) -> str:
        if "anchor_alpha" not in text:
            marker = "    # DSKD hyperparameters\n"
            if marker not in text:
                raise RuntimeError(f"Cannot find insertion marker in {path}: {marker!r}")
            text = text.replace(marker, ANCHOR_FIELDS + "\n" + marker, 1)

        if "anchor_snapshot_mode must be" not in text:
            marker = (
                "        if self.kd_temperature <= 0:\n"
                "            raise ValueError(f\"kd_temperature must be > 0, got {self.kd_temperature}.\")\n"
            )
            if marker not in text:
                raise RuntimeError(f"Cannot find validation marker in {path}")
            text = text.replace(marker, marker + ANCHOR_VALIDATION, 1)
        return text

    patch_file(path, transform)


def patch_student_group(repo: Path) -> None:
    path = repo / "kdflow" / "ray" / "train" / "student_group.py"

    def transform(text: str) -> str:
        if "async_refresh_anchor_snapshot" in text:
            return text
        marker = "    def async_run_distill(self, data):\n"
        if marker not in text:
            raise RuntimeError(f"Cannot find insertion marker in {path}: {marker!r}")
        return text.replace(marker, STUDENT_GROUP_METHOD + marker, 1)

    patch_file(path, transform)


def patch_student_actor(repo: Path) -> None:
    path = repo / "kdflow" / "ray" / "train" / "student_actor.py"

    def transform(text: str) -> str:
        if "def refresh_anchor_snapshot" in text:
            return text
        marker = "    def fit(self, train_data):\n"
        if marker not in text:
            raise RuntimeError(f"Cannot find insertion marker in {path}: {marker!r}")
        return text.replace(marker, STUDENT_ACTOR_METHOD + marker, 1)

    patch_file(path, transform)


def patch_off_policy_trainer(repo: Path) -> None:
    path = repo / "kdflow" / "trainer" / "off_policy_kd_trainer.py"

    def transform(text: str) -> str:
        if "def _maybe_refresh_anchor_snapshot" not in text:
            marker = "    def fit(self, global_step=0, start_epoch=0):\n"
            if marker not in text:
                raise RuntimeError(f"Cannot find insertion marker in {path}: {marker!r}")
            text = text.replace(marker, OFF_POLICY_METHOD + marker, 1)

        if "self._anchor_start_epoch = start_epoch" not in text:
            marker = "        self.global_step = global_step\n"
            if marker not in text:
                raise RuntimeError(f"Cannot find global_step marker in {path}")
            text = text.replace(marker, marker + "        self._anchor_start_epoch = start_epoch\n", 1)

        if "self._maybe_refresh_anchor_snapshot(epoch)" not in text:
            marker = "            self.current_epoch = epoch\n"
            if marker not in text:
                raise RuntimeError(f"Cannot find epoch marker in {path}")
            text = text.replace(marker, marker + "            self._maybe_refresh_anchor_snapshot(epoch)\n", 1)

        return text

    patch_file(path, transform)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Path to a local KDFlow checkout.")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    patch_root = Path(__file__).resolve().parent

    if not (repo / "kdflow").is_dir():
        raise FileNotFoundError(f"{repo} does not look like a KDFlow repository: missing kdflow/")

    copy_algorithm(repo, patch_root)
    patch_distillation_args(repo)
    patch_student_group(repo)
    patch_student_actor(repo)
    patch_off_policy_trainer(repo)

    print("\nDone. You can now use --kd_algorithm anchored_kd with kdflow.cli.train_kd_off_policy.")


if __name__ == "__main__":
    main()
