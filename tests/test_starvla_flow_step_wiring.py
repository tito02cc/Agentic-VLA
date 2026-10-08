from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STARVLA_MODEL_ROOT = (
    REPO_ROOT
    / "third_party/robodojo_official/XPolicyLab/policy/starVLA/source_starvla"
    / "starVLA/model"
)


def test_pi_v3_forwards_explicit_inference_steps_to_action_head() -> None:
    source = (STARVLA_MODEL_ROOT / "framework/VLM4A/QwenPI_v3.py").read_text(
        encoding="utf-8"
    )

    assert 'num_inference_steps = kwargs.pop("num_inference_steps", None)' in source
    assert "num_inference_steps=num_inference_steps" in source


def test_flow_action_head_accepts_runtime_inference_step_override() -> None:
    source = (
        STARVLA_MODEL_ROOT / "modules/action_model/LayerwiseFM_ActionHeader.py"
    ).read_text(encoding="utf-8")

    assert "num_inference_steps: int | None = None" in source
    assert "self.num_inference_timesteps" in source
    assert "else int(num_inference_steps)" in source


def test_inspection_shadow_opt_in_is_forwarded_without_changing_default():
    root = REPO_ROOT / "third_party/robodojo_official/XPolicyLab/policy/starVLA"
    launcher = (REPO_ROOT / "scripts/run_robodojo_starvla_nominal.sh").read_text()
    server = (root / "setup_eval_policy_server.sh").read_text()
    config = (root / "deploy.yml").read_text()
    assert "inspection_shadow=${CARVE_INSPECTION_SHADOW:-false}" in launcher
    assert 'STARVLA_CARVE_INSPECTION_SHADOW="${inspection_shadow}"' in launcher
    assert 'STARVLA_CARVE_PLANNER_MAX_IMAGES="${CARVE_PLANNER_MAX_IMAGES:-3}"' in launcher
    assert (
        'starvla_carve_inspection_shadow="${STARVLA_CARVE_INSPECTION_SHADOW:-false}"'
        in server
    )
    assert 'carve_inspection_shadow="${starvla_carve_inspection_shadow}"' in server
    assert "carve_inspection_shadow: false" in config


def test_native_request_trace_opt_in_is_forwarded_and_defaults_off():
    """The B0 (native) arm can be instrumented, but only when asked for.

    Earlier frozen baseline runs were recorded with no native tracing at all, so
    the default has to stay off for their numbers to remain comparable. When the
    caller opts in, the resolved path must also land in run_config.json,
    otherwise an artifact directory cannot be audited for whether the per-request
    evidence was on.
    """
    root = REPO_ROOT / "third_party/robodojo_official/XPolicyLab/policy/starVLA"
    launcher = (REPO_ROOT / "scripts/run_robodojo_starvla_nominal.sh").read_text()
    server = (root / "setup_eval_policy_server.sh").read_text()
    config = (root / "deploy.yml").read_text()

    assert "native_request_trace=${CARVE_NATIVE_REQUEST_TRACE:-false}" in launcher
    assert 'native_trace_path="${artifact_dir}/runtime_trace.jsonl"' in launcher
    assert 'native_trace_path=""' in launcher
    assert 'STARVLA_CARVE_NATIVE_TRACE_PATH="${native_trace_path}"' in launcher
    assert '"native_request_trace": ${native_request_trace},' in launcher
    assert '"native_trace_path": "${native_trace_path}",' in launcher

    assert (
        'starvla_carve_native_trace_path="${STARVLA_CARVE_NATIVE_TRACE_PATH:-}"'
        in server
    )
    assert 'carve_native_trace_path="${starvla_carve_native_trace_path}"' in server
    assert "carve_native_trace_path: null" in config


def test_payload_capture_opt_in_is_forwarded_and_requires_the_trace():
    """Capture is opt-in, bounded, and cannot be enabled without the trace.

    Each captured payload is keyed by the digests the native trace computes, so
    capturing without tracing would produce files nothing can verify.
    """
    root = REPO_ROOT / "third_party/robodojo_official/XPolicyLab/policy/starVLA"
    launcher = (REPO_ROOT / "scripts/run_robodojo_starvla_nominal.sh").read_text()
    server = (root / "setup_eval_policy_server.sh").read_text()
    config = (root / "deploy.yml").read_text()

    assert "native_payload_capture=${CARVE_NATIVE_PAYLOAD_CAPTURE:-false}" in launcher
    assert (
        "native_payload_capture_max=${CARVE_NATIVE_PAYLOAD_CAPTURE_MAX:-16}" in launcher
    )
    assert 'CARVE_NATIVE_PAYLOAD_CAPTURE requires CARVE_NATIVE_REQUEST_TRACE=true' in launcher
    assert (
        'STARVLA_CARVE_NATIVE_PAYLOAD_CAPTURE_DIR="${native_payload_capture_dir}"'
        in launcher
    )
    assert '"native_payload_capture": ${native_payload_capture},' in launcher

    assert (
        'starvla_carve_native_payload_capture_dir="${STARVLA_CARVE_NATIVE_PAYLOAD_CAPTURE_DIR:-}"'
        in server
    )
    assert (
        'carve_native_payload_capture_dir="${starvla_carve_native_payload_capture_dir}"'
        in server
    )
    assert "carve_native_payload_capture_dir: null" in config
    assert "carve_native_payload_capture_max: 16" in config


def test_stall_evidence_limits_are_wired_and_default_to_derived():
    """The two scheduling limits reach the model, and null means "derive"."""
    root = REPO_ROOT / "third_party/robodojo_official/XPolicyLab/policy/starVLA"
    launcher = (REPO_ROOT / "scripts/run_robodojo_starvla_nominal.sh").read_text()
    server = (root / "setup_eval_policy_server.sh").read_text()
    config = (root / "deploy.yml").read_text()

    assert (
        'STARVLA_CARVE_SEMANTIC_MIN_STALL_INTERVAL_STEPS="${CARVE_SEMANTIC_MIN_STALL_INTERVAL_STEPS:-}"'
        in launcher
    )
    assert (
        'STARVLA_CARVE_SEMANTIC_OBSERVATION_VALIDITY_STEPS="${CARVE_SEMANTIC_OBSERVATION_VALIDITY_STEPS:-}"'
        in launcher
    )
    assert (
        'starvla_carve_semantic_min_stall_interval_steps="${STARVLA_CARVE_SEMANTIC_MIN_STALL_INTERVAL_STEPS:-}"'
        in server
    )
    assert (
        'carve_semantic_min_stall_interval_steps="${starvla_carve_semantic_min_stall_interval_steps}"'
        in server
    )
    assert (
        'carve_semantic_observation_validity_steps="${starvla_carve_semantic_observation_validity_steps}"'
        in server
    )
    # null, not a number: the derivation lives in the model only.
    assert "carve_semantic_min_stall_interval_steps: null" in config
    assert "carve_semantic_observation_validity_steps: null" in config


def test_min_unchanged_steps_is_wired_and_defaults_to_derived():
    """Span-based stall evidence must reach the model, deriving when unset."""
    root = REPO_ROOT / "third_party/robodojo_official/XPolicyLab/policy/starVLA"
    launcher = (REPO_ROOT / "scripts/run_robodojo_starvla_nominal.sh").read_text()
    server = (root / "setup_eval_policy_server.sh").read_text()
    config = (root / "deploy.yml").read_text()

    assert (
        'STARVLA_CARVE_SEMANTIC_MIN_UNCHANGED_STEPS="${CARVE_SEMANTIC_MIN_UNCHANGED_STEPS:-}"'
        in launcher
    )
    assert (
        'starvla_carve_semantic_min_unchanged_steps="${STARVLA_CARVE_SEMANTIC_MIN_UNCHANGED_STEPS:-}"'
        in server
    )
    assert (
        'carve_semantic_min_unchanged_steps="${starvla_carve_semantic_min_unchanged_steps}"'
        in server
    )
    assert "carve_semantic_min_unchanged_steps: null" in config
