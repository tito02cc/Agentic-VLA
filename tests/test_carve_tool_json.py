import pytest

from agentic_vla.toolchain._json import decode_json_object


@pytest.mark.parametrize(
    "raw",
    [
        '{"inspect": null}',
        ' \n```json\n{"inspect": null}\n```\n',
        '```JSON\n{"inspect": null}\n```',
        '```\n{"inspect": null}\n```',
        {"inspect": None},
    ],
)
def test_one_complete_object_without_repair(raw):
    assert decode_json_object(raw) == {"inspect": None}


@pytest.mark.parametrize(
    "raw",
    [
        'Here is JSON:\n```json\n{}\n```',
        '```json\n{}\n```\nNow execute.',
        '```json\n{}',
        '```python\n{}\n```',
        '```json\n{}\n```\n```json\n{}\n```',
        '{} {}',
        '{"inspect": null, "inspect": {}}',
        '{"outer": {"state": "unknown", "state": "present"}}',
        '{"confidence": NaN}',
        '{"confidence": Infinity}',
        '{"confidence": -Infinity}',
        '[]',
        'null',
        None,
        ' ' * 16385,
    ],
)
def test_ambiguous_unbounded_or_non_object_envelopes_rejected(raw):
    with pytest.raises(ValueError):
        decode_json_object(raw)
