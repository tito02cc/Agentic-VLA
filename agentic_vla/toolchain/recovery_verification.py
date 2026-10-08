"""Post-action verification lifecycle; issuing a recovery is not recovery success."""

import copy
import math
import time

import numpy as np

from .inspection import _rgb_hash
from .verifier import VisualVerificationContext


class RecoveryVerificationLifecycle:
    """One pending bounded action attempt, then one fresh visual verification.

    The host supplies observations after action execution and planned chunk ends.
    No robot commands or procedure-memory writes are issued by this component.
    A same-pixel new camera observation is allowed; image change is not proof.
    """

    def __init__(self, *, max_checks=2, deadline_s=12.0, clock=time.monotonic):
        if type(max_checks) is not int or not 1 <= max_checks <= 8:
            raise ValueError("max_checks must be in [1,8]")
        if (
            type(deadline_s) not in (int, float)
            or not math.isfinite(deadline_s)
            or deadline_s <= 0
        ):
            raise ValueError("deadline must be positive and finite")
        self.max_checks, self.deadline, self.clock = max_checks, deadline_s, clock
        self.observation_seq = 0
        self.observation_step = -1
        self.pending = None
        self.checks = 0

    def observe(self, step):
        if type(step) is not int or step < self.observation_step:
            raise ValueError("observation steps must be monotonic")
        self.observation_seq += 1
        self.observation_step = step

    @property
    def available(self):
        return self.pending is None and self.checks < self.max_checks

    def begin(self, *, step, expected_outcome, image):
        if not self.available:
            raise ValueError("recovery verification pending or budget exhausted")
        if step != self.observation_step or self.observation_seq < 1:
            raise ValueError("recovery requires the current host observation")
        if (
            not isinstance(expected_outcome, str)
            or not expected_outcome.strip()
            or len(expected_outcome) > 2048
        ):
            raise ValueError("bounded expected outcome required")
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError("RGB camera observation required")
        self.pending = {
            "start_step": step,
            "start_observation_seq": self.observation_seq,
            "expected_outcome": expected_outcome,
            "before_rgb_sha256": _rgb_hash(image),
            "last_chunk_end": None,
            "chunks_issued": 0,
        }

    def issued_chunk(self, *, step, execute_steps):
        if self.pending is None:
            return
        if (
            type(execute_steps) is not int
            or execute_steps <= 0
            or step < self.pending["start_step"]
        ):
            raise ValueError("positive recovery execution interval required")
        self.pending["last_chunk_end"] = step + execute_steps
        self.pending["chunks_issued"] += 1

    def ready(self, *, step, work_remaining):
        p = self.pending
        return bool(
            p is not None
            and not work_remaining
            and p["last_chunk_end"] is not None
            and step >= p["last_chunk_end"]
            and self.observation_step == step
            and self.observation_seq > p["start_observation_seq"]
        )

    def verify(self, *, step, work_remaining, image, task_instruction, verifier):
        if not self.ready(step=step, work_remaining=work_remaining):
            return None
        pending = copy.deepcopy(self.pending)
        self.checks += 1
        self.pending = None
        started = self.clock()
        result = verifier.verify(
            VisualVerificationContext(
                task_instruction=task_instruction,
                expected_outcome=pending["expected_outcome"],
                frames={"current_head": image.copy()},
                timestep=step,
                active_stage="check the expected outcome after the bounded recovery actions",
            )
        )
        elapsed = self.clock() - started
        accepted = bool(result.accepted and elapsed < self.deadline)
        return {
            "attempt": pending,
            "verification_step": step,
            "observation_seq": self.observation_seq,
            "after_rgb_sha256": _rgb_hash(image),
            "accepted": accepted,
            "elapsed_ms": elapsed * 1000,
            "status": result.report.status.value if accepted else "inconclusive",
            "report": result.report.to_dict(),
            "raw_output": result.raw_output,
            "error": "verification deadline exceeded"
            if elapsed >= self.deadline
            else result.error,
            "checks_used": self.checks,
            "authority": "visual_report_not_task_success_or_memory_promotion",
        }

    def close(self):
        pending, self.pending = self.pending, None
        return {
            "status": "unverified_at_episode_reset" if pending else "closed",
            "pending": pending,
            "checks_used": self.checks,
        }
