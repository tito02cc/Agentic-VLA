"""Exercise shell cancellation using CPU-only fake policy/client children."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "third_party/robodojo_official/scripts/internal/run_policy_eval.sh"


def _write_worker(path: Path, marker: Path, *, exit_normally: bool = False) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "trap '' TERM\n"
        "sleep 300 &\n"
        f'printf \'%s %s\\n\' "$BASHPID" "$!" > {shlex.quote(str(marker))}\n'
        + ("exit 0\n" if exit_normally else "wait\n")
    )


def _running(pid: int) -> bool:
    try:
        status = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return status.rsplit(")", 1)[1].split()[0] != "Z"


@pytest.mark.parametrize("finish_mode", ["running", "startup", "complete"])
def test_cancel_cleans_owned_server_client_and_readiness_groups(tmp_path, finish_mode):
    cancel_during_startup = finish_mode == "startup"
    policy = tmp_path / "XPolicyLab/policy/fake"
    utils = tmp_path / "XPolicyLab/utils"
    policy.mkdir(parents=True)
    utils.mkdir(parents=True)
    markers = {
        name: tmp_path / f"{name}.pids" for name in ("server", "client", "ready")
    }
    _write_worker(policy / "setup_eval_policy_server.sh", markers["server"])
    _write_worker(
        policy / "setup_eval_env_client.sh",
        markers["client"],
        exit_normally=finish_mode == "complete",
    )
    (utils / "get_free_port.sh").write_text("printf '19000\\n'\n")
    if cancel_during_startup:
        _write_worker(utils / "wait_for_policy_server.sh", markers["ready"])
    else:
        (utils / "wait_for_policy_server.sh").write_text("exit 0\n")

    with (tmp_path / "run.log").open("w") as log:
        process = subprocess.Popen(
            [
                "bash",
                str(SCRIPT),
                str(policy),
                "RoboDojo",
                "task",
                "checkpoint",
                "arx_x5",
                "joint",
                "0",
                "0",
                "0",
                "/unused",
                "/unused",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            expected = "ready" if cancel_during_startup else "client"
            deadline = time.monotonic() + 5
            while not all(markers[name].exists() for name in ("server", expected)):
                assert process.poll() is None
                if time.monotonic() > deadline:
                    pytest.fail("worker startup timed out")
                time.sleep(0.02)
            if finish_mode != "complete":
                process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=10) == (0 if finish_mode == "complete" else 143)
            for marker in markers.values():
                if marker.exists():
                    assert all(
                        not _running(int(value)) for value in marker.read_text().split()
                    )
            if cancel_during_startup:
                assert not markers["client"].exists()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            for marker in markers.values():
                if marker.exists():
                    leader = int(marker.read_text().split()[0])
                    try:
                        os.killpg(leader, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
