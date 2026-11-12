---
name: Behavior Tree
description: Create behavior trees for Minecraft automation using py-trees and the workshop action library
---

## When to use this skill

Use it when you need to design, build, or debug a behavior tree for the MAUVC automation workshop. It applies to any script under `src/automation_scripts/` that constructs a `py_trees.BehaviourTree`.

## References

Read these files before writing a tree:

- `src/automation_scripts/actions.py` — all available action nodes (movement, yaw, camera, detection)
- `src/automation_scripts/level_4_DIY_behavior_tree.py` — the template for a new level; fill in its `root.add_children(...)` block
- `src/automation_scripts/level_3_yolo_behavior_tree.py` — full YOLO + behavior tree example (gate and blue pail navigation)

## Concepts

Every behaviour has one of three statuses: `RUNNING`, `SUCCESS`, or `FAILURE`. A tree stops when the root returns something other than `RUNNING`.

- **Composites** (`Sequence`, `Selector`, `Parallel`) manage children.
- **Leaves** are atomic actions from `actions.py`.
- **Decorators** (`Retry`, `Timeout`, `Inverter`, `SuccessIsFailure`, `FailureIsSuccess`) wrap leaves or composites.

## Building a tree

1. Create the node: `node = MinecraftOverrideConfig(override=True)`.
2. Build sub-trees as named `Sequence`/`Selector` objects with `memory=True` so multi-second actions are not re-started every tick.
3. Attach them to a root sequence: `root.add_children([...])`.
4. Wrap the root in a `Timeout(...)` if you need an overall run limit.
5. Run it with the shared loop from `level_2_behavior_tree.py` (tick at 20 Hz, exit when `tree.root.status != RUNNING`).

## Common mistakes

- Forgetting `memory=True` — composites re-evaluate children from index 0 every tick and timed actions break.
- Using the wrong YOLO label name in `DetecedObject` or `MoveBasedOnDetection_*` (labels must match exactly, e.g. `"Blue pail"` not `"blue pail"`).
- Reusing a single action instance in multiple children — construct fresh instances for each child.
