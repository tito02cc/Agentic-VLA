# Invalid success label: integration development only

Do NOT report the raw `_result.json` success_rate=1.0 / score=100 as task success.
The adapter returned at step100 after a Planner task-only instruction violation.
Upstream RoboDojo initializes `success=True` for running environments; the old
adapter returned without finalizing that running state. `run_eval` then saved the
default flag as success. The original video filename contains `_success` for the
same reason; it is not evidence of success. The final frame visibly shows the
mouse outside the pad, keyboard outside the frame and drawer still closed.

Raw outputs are preserved for audit, not corrected in place. This run proves
two real VLA chunks and one visual review reached the live environment, followed
by a rejected Planner response and stop. It does not prove task completion or
Agent improvement. Exclude it from task-effectiveness statistics.

Repair: the adapter checks native termination and, if still running at an early
return, marks execution stopped through the existing `success=False` terminal
signal and invokes the native final reward check. Reward definitions, task steps
and success thresholds are unchanged. Four terminal-state regression cases pass.
The next run uses a fresh ID. Compact review only permits continue/safe_stop;
it cannot rewrite task instructions or authorize ledger completion.
