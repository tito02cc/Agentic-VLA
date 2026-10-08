#!/usr/bin/env python3
"""Probe CARVE planning through StarVLA's already-resident VLM backbone."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
STARVLA_ROOT = (
    REPO_ROOT
    / "third_party"
    / "robodojo_official"
    / "XPolicyLab"
    / "policy"
    / "starVLA"
    / "source_starvla"
)
for path in (REPO_ROOT, STARVLA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentic_vla.runtime import (  # noqa: E402
    AgentIntent,
    HighLevelAgentContext,
    RiskAssessment,
    build_high_level_agent_request,
    parse_high_level_agent_decision,
)
from deployment.model_server.tools.websocket_policy_client import (  # noqa: E402
    WebsocketClientPolicy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--instruction",
        default="Build a tower using the wooden blocks and wooden boards.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    frame = np.asarray(Image.open(args.image).convert("RGB"))
    context = HighLevelAgentContext(
        task_instruction=args.instruction,
        trigger="no_progress",
        episode_id="shared-backbone-smoke",
        timestep=640,
        frames={"head": frame},
        risk=RiskAssessment(
            score=0.65,
            bucket="high",
            event="no_progress",
            components={"no_progress": 1.0},
            evidence={"event_streak": 2},
        ).to_dict(),
        current_subgoal=args.instruction,
        allowed_intents=(
            AgentIntent.CONTINUE,
            AgentIntent.VLA_ACT,
            AgentIntent.SAFE_STOP,
        ),
        remaining_retries=2,
    )
    request = build_high_level_agent_request(context)
    client = WebsocketClientPolicy(args.host, args.port)
    try:
        metadata = client.get_server_metadata()
        if not metadata.get("shared_semantic_generation", False):
            raise RuntimeError("server does not advertise shared semantic generation")
        started = time.perf_counter()
        response = client.semantic_decision(
            {
                "system_prompt": request["system_prompt"],
                "user_prompt": request["user_prompt"],
                "frames": request["frames"],
                "max_new_tokens": args.max_new_tokens,
            }
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    finally:
        client.close()

    if not response.get("ok", False):
        raise RuntimeError(response.get("error", response))
    raw_text = response["data"]["text"]
    report = {
        "elapsed_ms": elapsed_ms,
        "server_data": response["data"],
        "validated_decision": None,
        "validation_error": None,
    }
    try:
        report["validated_decision"] = parse_high_level_agent_decision(
            raw_text, context
        ).to_dict()
    except (TypeError, ValueError) as exc:
        report["validation_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["validation_error"] is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
