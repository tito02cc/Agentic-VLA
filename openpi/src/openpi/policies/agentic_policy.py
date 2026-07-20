import json
import logging
import re
import threading
import time
from typing import Any

import numpy as np
import torch
from openpi_client import base_policy as _base_policy
from transformers import BitsAndBytesConfig
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)

_ALLOWED_ACTIONS = ("continue", "regrasp", "lift", "retreat", "slow_down")
_ALLOWED_SUBGOALS = ("execute", "recover", "regrasp", "lift", "retreat", "slow_down")


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _default_mapping_for_action(action: str, instruction: str, event: str) -> tuple[str, dict[str, Any]]:
    if action == "slow_down":
        prompt = f"{instruction}, move slowly and carefully"
        return prompt, {"action_scale": 0.6, "apply_steps": 10}
    if action == "retreat":
        prompt = (
            "carefully retract a little to avoid collision, stabilize, then continue the task: "
            f"{instruction}"
        )
        return prompt, {"action_scale": 0.7, "apply_steps": 10}
    if action == "lift":
        prompt = (
            "slightly lift and stabilize while maintaining the current grasp, then continue: "
            f"{instruction}"
        )
        return prompt, {"action_scale": 0.7, "apply_steps": 10}
    if action == "regrasp":
        prompt = (
            "open the gripper, reposition, then grasp firmly and continue the task: "
            f"{instruction}"
        )
        return prompt, {"action_scale": 0.8, "apply_steps": 10}
    return instruction, {"action_scale": 1.0, "apply_steps": 0}


def _parse_planner_output(raw_text: str, instruction: str, event: str) -> dict[str, Any]:
    parsed = _extract_first_json_object(raw_text)
    if parsed is None:
        action = "continue"
        prompt_suggested, client_hints = _default_mapping_for_action(action, instruction, event)
        return {
            "action": action,
            "subgoal": "execute",
            "prompt_suggested": prompt_suggested,
            "client_hints": client_hints,
            "raw": raw_text.strip()[:512],
        }

    action = parsed.get("action", "continue")
    if not isinstance(action, str):
        action = "continue"
    action = action.strip().lower()
    if action not in _ALLOWED_ACTIONS:
        action = "continue"

    subgoal = parsed.get("subgoal", "")
    if not isinstance(subgoal, str):
        subgoal = ""
    subgoal = subgoal.strip().lower()
    if not subgoal:
        subgoal = "execute" if action == "continue" else action
    if subgoal == "continue":
        subgoal = "execute"
    if subgoal not in _ALLOWED_SUBGOALS:
        subgoal = "execute" if action == "continue" else action

    prompt_suggested = parsed.get("prompt_suggested")
    if not isinstance(prompt_suggested, str) or not prompt_suggested.strip():
        prompt_suggested, default_hints = _default_mapping_for_action(action, instruction, event)
    else:
        prompt_suggested = prompt_suggested.strip()
        default_hints = _default_mapping_for_action(action, instruction, event)[1]

    client_hints = parsed.get("client_hints")
    if not isinstance(client_hints, dict):
        client_hints = {}
    merged_hints = dict(default_hints)
    for k in ("action_scale", "apply_steps"):
        if k in client_hints:
            merged_hints[k] = client_hints[k]
    try:
        merged_hints["action_scale"] = float(merged_hints.get("action_scale", 1.0))
    except Exception:
        merged_hints["action_scale"] = 1.0
    merged_hints["action_scale"] = float(np.clip(merged_hints["action_scale"], 0.2, 1.5))
    try:
        merged_hints["apply_steps"] = int(merged_hints.get("apply_steps", 0))
    except Exception:
        merged_hints["apply_steps"] = 0
    merged_hints["apply_steps"] = int(np.clip(merged_hints["apply_steps"], 0, 200))

    rationale = parsed.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = ""
    rationale = rationale.strip()[:256]

    return {
        "action": action,
        "subgoal": subgoal,
        "prompt_suggested": prompt_suggested,
        "client_hints": merged_hints,
        "rationale": rationale,
        "raw": raw_text.strip()[:512],
    }


class Qwen3VLPlanner:
    def __init__(
        self,
        model: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        quant: str = "4bit",
        max_new_tokens: int = 96,
    ) -> None:
        self._model_id = model
        self._device = device
        self._torch_dtype = torch_dtype
        self._quant = quant
        self._max_new_tokens = int(max_new_tokens)
        self._processor = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        dtype = getattr(torch, self._torch_dtype, torch.bfloat16)
        self._processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
        quant = (self._quant or "none").lower()
        quant_cfg = None
        if quant in {"4bit", "nf4"}:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        elif quant in {"8bit", "int8"}:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
            device_map=self._device,
            trust_remote_code=True,
            quantization_config=quant_cfg,
        )
        self._model.eval()

    def plan(self, instruction: str, obs: dict[str, Any], *, previous_plan: str | None = None) -> str:
        try:
            self._ensure_loaded()
        except Exception as exc:
            logger.error("planner_load_failed: %s", exc)
            return json.dumps({"action": "continue", "prompt_suggested": instruction})

        state = obs.get("observation/state")
        state_summary = ""
        if isinstance(state, np.ndarray):
            flat = state.reshape(-1).astype(np.float32)
            state_summary = "state=" + ",".join([f"{x:.3f}" for x in flat[:16].tolist()])
        agentic = obs.get("agentic") if isinstance(obs.get("agentic"), dict) else {}
        event = agentic.get("event", "")
        reason = agentic.get("reason", "")
        context = agentic.get("context", "")
        if not isinstance(event, str):
            event = ""
        if not isinstance(reason, str):
            reason = ""
        if not isinstance(context, str):
            context = ""
        event = event.strip()[:64]
        reason = reason.strip()[:256]
        context = context.strip()[:512]

        sys_text = (
            "你是具身智能Agent的高层规划器，低层控制器会执行连续动作。"
            "现在低层控制器遇到了失败信号(event)，你需要给出一个可验证的结构化决策。"
            "你必须只输出一个JSON对象，格式如下："
            '{"action": one of ["continue","regrasp","lift","retreat","slow_down"], '
            '"subgoal": one of ["execute","recover","regrasp","lift","retreat","slow_down"], '
            '"prompt_suggested": str, "client_hints": {"action_scale": float, "apply_steps": int}, '
            '"rationale": str}'
            "不要输出除JSON之外的任何内容。"
        )
        user_text = (
            f"用户指令：{instruction}\n"
            f"event={event}\n"
            f"reason={reason}\n"
            f"context={context}\n"
            f"{state_summary}\n"
            f"上一轮计划：{previous_plan or ''}"
        )

        processor = self._processor
        model = self._model
        assert processor is not None and model is not None

        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_text},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
        gen = out_ids[:, inputs["input_ids"].shape[1] :]
        out = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return out or json.dumps({"action": "continue", "prompt_suggested": instruction})


class AgenticPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        base_policy: _base_policy.BasePolicy,
        planner: Qwen3VLPlanner,
        *,
        plan_mode: str = "once",
        plan_interval: int = 20,
        plan_cooldown_steps: int = 50,
        max_plans_per_episode: int = 3,
        cache_enabled: bool = True,
        lock: Any = None,
        default_prompt: str | None = None,
    ) -> None:
        self._base = base_policy
        self._planner = planner
        self._plan_mode = str(plan_mode)
        self._plan_interval = int(plan_interval)
        self._plan_cooldown_steps = int(plan_cooldown_steps)
        self._max_plans_per_episode = int(max_plans_per_episode)
        self._cache_enabled = bool(cache_enabled)
        self._lock = lock
        self._default_prompt = default_prompt
        self._step = 0
        self._current_plan: str | None = None
        self._episode_id: str | int | None = None
        self._last_plan_step = -10**9
        self._plans_used = 0
        self._decision_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def infer(self, obs: dict) -> dict:
        incoming_episode_id = obs.get("episode_id")
        if incoming_episode_id is not None and incoming_episode_id != self._episode_id:
            self._episode_id = incoming_episode_id
            self._step = 0
            self._current_plan = None
            self._last_plan_step = -10**9
            self._plans_used = 0
            self._decision_cache = {}

        instruction = obs.get("prompt") or self._default_prompt or ""
        agentic_req = obs.get("agentic") if isinstance(obs.get("agentic"), dict) else {}
        request_plan = bool(agentic_req.get("request_plan", False))
        event = str(agentic_req.get("event", "")) if agentic_req.get("event") is not None else ""
        reason = str(agentic_req.get("reason", "")) if agentic_req.get("reason") is not None else ""
        planner_decision: dict[str, Any] | None = None
        used_cache = False
        throttled = False
        maxed_out = False
        should_plan = False
        if self._plan_mode == "on_demand":
            should_plan = request_plan
        elif self._plan_mode == "once":
            should_plan = self._step == 0
        else:
            should_plan = self._plan_interval > 0 and (self._step % self._plan_interval == 0)

        cache_key = (event, str(instruction))
        if should_plan and self._plan_mode == "on_demand" and self._cache_enabled and cache_key in self._decision_cache:
            planner_decision = self._decision_cache[cache_key]
            used_cache = True
            should_plan = False

        if should_plan and self._plan_mode == "on_demand":
            if self._plans_used >= self._max_plans_per_episode:
                maxed_out = True
                should_plan = False
            elif (self._step - self._last_plan_step) < self._plan_cooldown_steps:
                throttled = True
                should_plan = False

        if should_plan:
            plan_t0 = time.monotonic()
            if self._lock is None:
                raw = self._planner.plan(str(instruction), obs, previous_plan=self._current_plan)
            else:
                with self._lock:
                    raw = self._planner.plan(str(instruction), obs, previous_plan=self._current_plan)
            plan_dt = (time.monotonic() - plan_t0) * 1000
            planner_decision = _parse_planner_output(raw, str(instruction), event)
            self._current_plan = str(planner_decision.get("prompt_suggested") or instruction)
            self._last_plan_step = int(self._step)
            self._plans_used += 1
            if self._cache_enabled:
                self._decision_cache[cache_key] = planner_decision
        else:
            plan_dt = 0.0
            if planner_decision is not None:
                self._current_plan = str(planner_decision.get("prompt_suggested") or instruction)

        obs2 = dict(obs)
        if self._current_plan:
            obs2["prompt"] = self._current_plan

        infer_t0 = time.monotonic()
        if self._lock is None:
            out = self._base.infer(obs2)
        else:
            with self._lock:
                out = self._base.infer(obs2)
        infer_dt = (time.monotonic() - infer_t0) * 1000

        out["agentic"] = {
            "step": int(self._step),
            "episode_id": self._episode_id,
            "plan_mode": self._plan_mode,
            "plan_interval": int(self._plan_interval),
            "request_plan": bool(request_plan),
            "event": event,
            "used_cache": bool(used_cache),
            "throttled": bool(throttled),
            "maxed_out": bool(maxed_out),
            "plans_used": int(self._plans_used),
            "plan_ms": float(plan_dt),
            "base_infer_ms": float(infer_dt),
            "action": (planner_decision or {}).get("action", "continue"),
            "subgoal": (planner_decision or {}).get("subgoal", "execute"),
            "prompt_suggested": (planner_decision or {}).get("prompt_suggested", ""),
            "client_hints": (planner_decision or {}).get("client_hints", {}) if planner_decision is not None else {},
            "rationale": (planner_decision or {}).get("rationale", ""),
            "plan": self._current_plan or "",
        }
        self._step += 1
        return out

    @property
    def metadata(self) -> dict[str, Any]:
        md = {}
        if hasattr(self._base, "metadata"):
            md.update(getattr(self._base, "metadata"))
        md["agentic"] = {"planner": "qwen3-vl-8b-instruct", "plan_interval": self._plan_interval}
        return json.loads(json.dumps(md))
