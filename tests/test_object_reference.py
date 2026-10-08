import dataclasses
import json

import numpy as np
import pytest

from agentic_vla.runtime.agent import PolicyServiceVisionPlanner
from agentic_vla.toolchain import (
    NativeObjectObserver, ObjectEvidenceMemory, ObjectRole,
    ReferenceConditionedObjectObserver, ToolExecutionContext,
)

ROLES = (ObjectRole("base", "long board"), ObjectRole("upper", "short board"))
LABELS = {role.description: role.role_id for role in ROLES}
RAW = '[{"label":"long board","bbox_2d":[0,0,600,400]},' \
      '{"label":"short board","bbox_2d":[100,500,700,950]}]'


def ctx(step=0, episode="e"):
    return ToolExecutionContext(episode, step, False, ("read_object_evidence",))


def setup(raw=RAW):
    memory = ObjectEvidenceMemory()
    memory.reset("e")
    image = np.arange(64 * 96 * 3, dtype=np.uint8).reshape(64, 96, 3)
    frame = memory.capture("head", image, ctx())
    observer = NativeObjectObserver(lambda _: raw, label_to_role=LABELS, user_prompt="Locate boards")
    result = observer.observe(memory, frame, ROLES, context=ctx(), current_context=ctx)
    return memory, image, frame, result


def observer(reference, infer):
    return ReferenceConditionedObjectObserver(infer, reference=reference,
        label_to_role=LABELS, user_prompt="Locate boards")


def test_reference_survives_new_frame_but_does_not_supply_current_positions():
    memory, image, old_frame, _ = setup()
    reference = memory.pin_reference(ctx())
    original = image.copy()
    image[:] = 0
    frame = memory.capture("head", image, ctx(16))
    assert memory.read(ctx(16))["current"] is None
    calls = []

    def infer(request):
        calls.append(request)
        np.testing.assert_array_equal(request["frames"]["reference_global"], original)
        assert not request["frames"]["current"].any()
        request["frames"]["reference_global"][:] = 0
        return "[]"

    result = observer(reference, infer).observe(memory, frame, ROLES,
        context=ctx(16), current_context=lambda: ctx(16))
    assert result["accepted"]
    assert len(calls[0]["required_frame_names"]) == 4
    objects = memory.read(ctx(16))["current"]["objects"]
    assert all(obj["state"] == "unknown" and obj["current_box"] is None for obj in objects)
    assert all(obj["reference_observation_ref"].startswith(old_frame["frame_id"]) for obj in objects)
    assert all(obj["identity_authority"] == "model_association_hypothesis_not_verified" for obj in objects)
    images, metadata = reference.resolve(frame, ROLES)
    np.testing.assert_array_equal(images["reference_global"], original)
    assert metadata["authority"] == "historical_model_hypothesis_not_verified_identity"
    metadata["objects"][0]["reference_box"][0] = 999
    assert reference.provenance["objects"][0]["reference_box"][0] == 0


@pytest.mark.parametrize("raw", ["[]", "invalid", '[{"label":"long board","bbox_2d":[0,0,600,400]}]',
    '[{"label":"long board","bbox_2d":[0,0,100,100]},{"label":"short board","bbox_2d":[100,500,700,950]}]',
    '[{"label":"long board","bbox_2d":[0,0,600,400]},{"label":"short board","bbox_2d":[0,0,600,400]}]'])
def test_unusable_hypotheses_cannot_be_pinned(raw):
    memory, _, _, _ = setup(raw)
    with pytest.raises(ValueError):
        memory.pin_reference(ctx())


def test_expired_observation_cannot_be_pinned():
    memory, _, _, _ = setup()
    with pytest.raises(ValueError, match="fresh"):
        memory.pin_reference(ctx(1))


@pytest.mark.parametrize("change", [dict(episode_id="other"), dict(timestep=0),
    dict(timestep=-1), dict(camera="wrist")])
def test_reference_context_mismatch_rejected_before_provider(change):
    memory, image, _, _ = setup()
    reference = memory.pin_reference(ctx())
    frame = memory.capture("head", image, ctx(16))
    with pytest.raises(ValueError):
        reference.resolve({**frame, **change}, ROLES)


def test_role_change_rejected():
    memory, image, _, _ = setup()
    reference = memory.pin_reference(ctx())
    frame = memory.capture("head", image, ctx(16))
    with pytest.raises(ValueError, match="roles changed"):
        reference.resolve(frame, (dataclasses.replace(ROLES[0], description="block"), ROLES[1]))


@pytest.mark.parametrize("during_call", [False, True])
def test_same_episode_reset_revokes_reference(during_call):
    memory, image, _, _ = setup()
    reference = memory.pin_reference(ctx())
    frame = memory.capture("head", image, ctx(16))
    calls = []

    def infer(request):
        calls.append(request)
        memory.reset("e")
        return RAW

    if not during_call:
        memory.reset("e")
        frame = memory.capture("head", image, ctx(16))
    result = observer(reference, infer).observe(memory, frame, ROLES,
        context=ctx(16), current_context=lambda: ctx(16))
    assert not result["accepted"]
    assert len(calls) == int(during_call)
    with pytest.raises(ValueError, match="revoked"):
        _ = reference.provenance


def test_required_current_image_cannot_be_silently_truncated():
    memory, image, _, _ = setup()
    reference = memory.pin_reference(ctx())
    frame = memory.capture("head", image, ctx(16))
    calls = []
    provider = PolicyServiceVisionPlanner(lambda payload: calls.append(payload), max_images=1)
    result = observer(reference, provider).observe(memory, frame, ROLES,
        context=ctx(16), current_context=lambda: ctx(16))
    assert not result["accepted"] and not calls
    assert "truncated" in result["error"]


def test_native_response_remains_current_frame_hypothesis_only():
    memory, image, _, _ = setup()
    reference = memory.pin_reference(ctx())
    frame = memory.capture("head", image, ctx(16))
    result = observer(reference, lambda _: RAW).observe(memory, frame, ROLES,
        context=ctx(16), current_context=lambda: ctx(16))
    assert result["accepted"]
    current = memory.read(ctx(16))["current"]
    assert current["frame"] == frame
    assert all(obj["authority"] == "model_localization_hypothesis_only" for obj in current["objects"])
    assert "completed" not in json.dumps(current)
