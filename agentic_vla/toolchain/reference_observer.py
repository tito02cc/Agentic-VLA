"""Reference-conditioned reacquisition, without promoting hypotheses to identity."""

import json

from .object_memory import ObjectReferenceSnapshot
from .reobservation import NativeObjectObserver


class ReferenceConditionedObjectObserver(NativeObjectObserver):
    """Use an earlier full view and model-selected crops as appearance memory.

    The same native response parser and current-frame commit checks are used as
    the single-frame observer. Missing or ambiguous matches remain unknown.
    This does not authorize recovery or verify a successful manipulation.
    """

    def __init__(self, infer, *, reference, label_to_role, user_prompt,
                 system_prompt="You are a helpful assistant."):
        super().__init__(infer, label_to_role=label_to_role, user_prompt=user_prompt,
                         system_prompt=system_prompt)
        if not isinstance(reference, ObjectReferenceSnapshot):
            raise TypeError("reference must come from object evidence memory")
        self.reference = reference

    def _request(self, image, frame, roles):
        request = super()._request(image, frame, roles)
        frames, provenance = self.reference.resolve(frame, roles)
        frames["current"] = image
        return {
            **request,
            "user_prompt": (
                "Locate objects ONLY in the image named current. reference_global is an "
                "earlier image of this episode; reference_crop images are crops of that "
                "same earlier image, not additional views or independent evidence. "
                "Use their appearance to match the same physical objects after movement. "
                "The reference labels and boxes are fallible model hypotheses. Do not "
                "copy reference coordinates into the current image or treat similar "
                "appearance as proof of identity. Omit any role whose current identity "
                "or position is ambiguous or hidden; do not substitute a nearby block. "
                + self.user_prompt + "\nReference provenance: "
                + json.dumps(provenance, separators=(",", ":"))
            ),
            "frames": frames,
            "required_frame_names": list(frames),
        }

    def _parse_response(self, raw, roles, shape):
        objects = super()._parse_response(raw, roles, shape)
        references = {obj["role_id"]: obj for obj in self.reference.provenance["objects"]}
        for obj in objects:
            obj["reference_observation_ref"] = references[obj["role_id"]]["observation_ref"]
            obj["identity_authority"] = "model_association_hypothesis_not_verified"
        return objects
