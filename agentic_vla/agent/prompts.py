"""Prompt templates for grasp planning agent."""

from typing import List, Dict, Any, Optional


class PromptTemplates:
    """Prompt templates for VLM-based grasp planning."""

    # System prompts

    SYSTEM_GRASP_EXPERT = """You are a robotic grasp planning expert. Your task is to analyze the scene image and generate a 6D grasp pose for the target object.

## Workflow
1. Analyze the scene to identify the target object and surrounding objects (shape, material, size)
2. Evaluate grasp feasibility of the target object (graspable regions, surface properties, weight)
3. Refer to successful past experiences and perform category-based reasoning
4. Output a complete grasp pose (position, orientation, gripper_width)

## Output Format
Output in JSON format as follows:
```json
{
    "grasp_pose": {
        "position": {"x": float, "y": float, "z": float},
        "orientation": {"qx": float, "qy": float, "qz": float, "qw": float},
        "gripper_width": float,
        "approach_vector": {"x": float, "y": float, "z": float}
    },
    "confidence": float,
    "reasoning": string
}
```

## Notes
- position: in meters, relative to the robot base frame
- orientation: quaternion representation
- confidence: value between 0 and 1 indicating the estimated grasp success probability
- reasoning: a brief explanation of the grasp strategy"""

    SYSTEM_GRASP_WITH_MEMORY = """You are a robotic grasp planning expert with access to an "experience memory" that stores previously successful grasp strategies.

## Workflow
1. Analyze the scene and identify the target object
2. Retrieve successful grasp experiences for similar objects of the same category
3. Adapt the grasp pose based on these experiences and current object properties
4. Output the optimized grasp plan

## Guidelines
- Prioritize object-specific experiences
- Leverage experiences from same-category objects for cross-instance transfer
- Adjust grasp parameters based on the current object's specific properties
- If no relevant experiences exist, rely on object property-based reasoning

## Output Format
Output in JSON format including the grasp pose and reasoning."""

    # Task prompts

    TASK_SIMPLE_GRASP = """## Grasp Task
Analyze the image and plan a grasp for the target object. Output a complete grasp pose.

Target object: {target_object}
Task description: {task_description}

{scene_description}"""

    TASK_WITH_EXAMPLES = """## Successful Grasp Experiences
{examples}

## Current Task
Analyze the scene image and plan a grasp for the target object.

Target object: {target_object}
Task description: {task_description}

Based on the above experiences, output the optimal grasp pose for this object. Consider the differences in object position and orientation."""

    TASK_MULTI_OBJECT = """## Multi-Object Grasp Task
The scene contains multiple objects. Plan a grasp strategy for the target object.

Target object: {target_object}
Scene objects: {object_list}
Task description: {task_description}

Please output:
1. The position and orientation of the target object in the scene
2. The recommended grasp pose
3. Any collision avoidance considerations"""

    TASK_SEQUENCE = """## Sequential Grasp Task
Plan a sequence of grasps to complete the given task.

Task goal: {task_goal}
Scene objects: {object_list}

Please output:
1. The ordered sequence of grasps (with execution priority)
2. The grasp pose for each object
3. Considerations for inter-step dependencies"""

    # Example formatting templates

    EXAMPLE_FORMAT = """### Experience {index}: {object_name}
- Description: {description}
- Category: {category}
- Successful grasp pose:
  - Position: ({pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f}) m
  - Gripper width: {gripper_width:.3f} m
- Similarity score: {similarity:.2f}
"""

    EXAMPLE_FORMAT_DETAILED = """### Experience {index}: {object_name}
**Object Properties**
- Description: {description}
- Category: {category}
- Material: {material}

**Successful Grasp Configuration**
```json
{grasp_pose_json}
```

**Key Insight**
- {key_insight}

**Similarity Score**: {similarity:.2f}
"""

    @classmethod
    def format_examples(
        cls,
        examples: List[Dict[str, Any]],
        detailed: bool = False,
    ) -> str:
        """Format retrieved grasp experiences into prompt text."""
        if not examples:
            return "(No relevant experiences found)"

        formatted = []
        template = cls.EXAMPLE_FORMAT_DETAILED if detailed else cls.EXAMPLE_FORMAT

        for i, ex in enumerate(examples):
            grasp = ex.get("grasp_pose", {})
            pos = grasp.get("position", {})

            if detailed:
                import json
                text = template.format(
                    index=i + 1,
                    object_name=ex.get("object_name", "unknown"),
                    description=ex.get("description", ""),
                    category=ex.get("category", ""),
                    material=ex.get("material", "unknown"),
                    grasp_pose_json=json.dumps(grasp, indent=2, ensure_ascii=False),
                    key_insight=ex.get("key_insight", "Standard grasp approach"),
                    similarity=ex.get("score", ex.get("combined_score", 0)),
                )
            else:
                text = template.format(
                    index=i + 1,
                    object_name=ex.get("object_name", "unknown"),
                    description=ex.get("description", ""),
                    category=ex.get("category", ""),
                    pos_x=pos.get("x", 0),
                    pos_y=pos.get("y", 0),
                    pos_z=pos.get("z", 0),
                    gripper_width=grasp.get("gripper_width", 0.08),
                    similarity=ex.get("score", ex.get("combined_score", 0)),
                )

            formatted.append(text)

        return "\n".join(formatted)

    @classmethod
    def build_grasp_prompt(
        cls,
        target_object: str,
        task_description: str = "Grasp the target object",
        examples: Optional[List[Dict[str, Any]]] = None,
        scene_description: str = "",
        use_memory: bool = True,
    ) -> tuple:
        """Build complete grasp planning prompt with system and user messages."""
        # Select system prompt
        if use_memory and examples:
            system = cls.SYSTEM_GRASP_WITH_MEMORY
        else:
            system = cls.SYSTEM_GRASP_EXPERT

        # Build user prompt
        if examples:
            examples_text = cls.format_examples(examples)
            user = cls.TASK_WITH_EXAMPLES.format(
                examples=examples_text,
                target_object=target_object,
                task_description=task_description,
            )
        else:
            user = cls.TASK_SIMPLE_GRASP.format(
                target_object=target_object,
                task_description=task_description,
                scene_description=scene_description,
            )

        return system, user

    @classmethod
    def build_sequence_prompt(
        cls,
        task_goal: str,
        object_list: List[str],
    ) -> tuple:
        """Build sequential grasp planning prompt."""
        system = cls.SYSTEM_GRASP_EXPERT
        user = cls.TASK_SEQUENCE.format(
            task_goal=task_goal,
            object_list=", ".join(object_list),
        )
        return system, user


# Scene-specific prompt augmentations


class ScenePrompts:
    """Scene-specific prompt augmentations for different environments."""

    KITCHEN = """## Scene Context: Kitchen
- Common objects: cups, bowls, utensils, bottles
- Surface types: countertop, shelf
- Grasp considerations: handle fragile items with care"""

    WORKSHOP = """## Scene Context: Workshop
- Common objects: tools, screws, brackets
- Surface types: metal workbench, toolbox
- Grasp considerations: tools may be heavy or have irregular shapes"""

    OFFICE = """## Scene Context: Office
- Common objects: pens, mouse, keyboard
- Surface types: desk, drawer
- Grasp considerations: lightweight objects, avoid disturbing arrangement"""

    WAREHOUSE = """## Scene Context: Warehouse
- Common objects: boxes, packages, containers
- Surface types: shelving, pallet
- Grasp considerations: consider object weight and stacking stability"""


class SafetyPrompts:
    """Safety-related prompt augmentations for special object properties."""

    FRAGILE_OBJECT = """Warning: The target object is classified as [FRAGILE].
- Use minimal gripper force
- Approach slowly with reduced speed
- Avoid contact with other nearby objects"""

    HEAVY_OBJECT = """Warning: The target object is heavy.
- Ensure sufficient gripper opening width
- Use a stable grasp pose with full contact
- Verify gripper payload capacity"""

    HOT_OBJECT = """Warning: The target object may be hot.
- Use indirect grasp points if possible
- Grasp with minimal contact duration"""

    SHARP_OBJECT = """Warning: The target object has sharp edges.
- Avoid grasping near sharp edges
- Prefer blunt surfaces for contact
- Use protective grasp strategy with wider gripper opening"""
