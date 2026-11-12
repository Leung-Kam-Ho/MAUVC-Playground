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

Example trees (see "Patterns" below):

- `scripts/move_in_square.py` — square movement with `memory=True` sequences
- `scripts/search_and_approach.py` — search for a detected object, then approach it
- `scripts/robust_navigation.py` — retries and an overall safety timeout

## Concepts

Every behaviour has one of three statuses: `RUNNING`, `SUCCESS`, or `FAILURE`. A tree stops when the root returns something other than `RUNNING`.

- **Composites** (`Sequence`, `Selector`, `Parallel`) manage children.
- **Leaves** are atomic actions from `actions.py`.
- **Decorators** (`Retry`, `Timeout`, `Inverter`, `SuccessIsFailure`, `FailureIsSuccess`) wrap leaves or composites.

### Composites: when to use which

- **Sequence** — "do this, then that". Returns `RUNNING` while a child is active; `SUCCESS` only when every child succeeded. Use for step-by-step plans (e.g. move forward → yaw 90°).
- **Selector** — "try this, fall back to that". Ticks children left-to-right until one returns `SUCCESS`. Use for alternatives (e.g. check right, else check left).
- **Parallel** — run children at the same time. With `ParallelPolicy.SuccessOnOne()` it succeeds when any child succeeds; with `SuccessOnAll()` all must succeed. Use when two actions must happen together (e.g. move sideways while tracking a detection).

### Decorators: what they do

- **`Retry(child, num_failures)`** — keep trying `child` until it succeeds or failures are exhausted. Wrap expensive/failing steps like detection-based movement with `num_failures=inf`.
- **`Timeout(child, duration)`** — return `FAILURE` if `child` has not completed within `duration` seconds. Use for safety nets around the whole tree.
- **`SuccessIsFailure(child)` / `FailureIsSuccess(child)`** — invert a child's status. Useful for "reset to centre" steps that must only run when something else failed.
- **`Inverter(child)`** — same as `SuccessIsFailure`.

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
- Forgetting that `MoveForward`, `Yaw`, and similar leaves send their command once and return `RUNNING` until the duration elapses; they stop automatically via `terminate()`.
