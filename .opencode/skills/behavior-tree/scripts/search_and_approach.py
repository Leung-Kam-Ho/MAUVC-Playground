"""Example: search for a detected object and approach it.

Demonstrates py_trees concepts used most often in this workshop:
  * `Selector` with `memory=True` to try alternatives until one succeeds.
  * `DetecedObject` as a leaf that reports SUCCESS when the target is found.
  * `MoveBasedOnDetection_Sway` to centre and approach an object on camera.
  * Keeping search behaviour separate from movement behaviour in nested Sequences.

Available detections: use whatever class name your YOLO model outputs, e.g.
"Gate", "Blue pail". Labels must match exactly (case-sensitive).

Run: uv run python src/automation_scripts/level_3_yolo_behavior_tree.py
"""

from time import sleep, perf_counter
import py_trees
from py_trees.decorators import Retry, SuccessIsFailure, FailureIsSuccess, Inverter, Timeout
from py_trees.composites import Selector, Sequence, Parallel
from minecraft.minecraft_override_config import MinecraftOverrideConfig
from pathlib import Path
from math import inf
from automation_scripts.actions import (
    InitWorld,
    SwitchCameraFront,
    SwitchCameraBottom,
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    Yaw,
    Stop,
    MoveBasedOnDetection_Sway,
    MoveBasedOnDetection_Sway_Surge,
    AvoidBasedOnDetection_Sway,
    DetecedObject
)

# --- Configuration ---
TARGET_CLASS = "Blue pail"      # exact label from your detection model
SEARCH_ANGLE = 40.0             # degrees to yaw left/right while searching
CONF_THRESHOLD = 0.8            # minimum confidence to consider a detection valid


if __name__ == "__main__":

    node = MinecraftOverrideConfig(override=True)

    root = py_trees.composites.Sequence("RootSequence", memory=True)

    # --- Automation Starts Here ---
    root.add_children([
        InitWorld(node),
        SwitchCameraBottom(node),
        SwitchCameraFront(node),

        # Try to find the target on either side, then move towards it.
        Selector("Find Target", memory=True, children=[

            Sequence("Check Right", memory=True, children=[
                Yaw(node, -SEARCH_ANGLE),
                DetecedObject(TARGET_CLASS, conf_threshold=CONF_THRESHOLD),
                Yaw(node, SEARCH_ANGLE),
                Parallel("Move to Target", policy=py_trees.common.ParallelPolicy.SuccessOnOne(), children=[
                    Retry(name="Retry MoveBasedOnDetection", child=MoveBasedOnDetection_Sway(node, TARGET_CLASS), num_failures=inf),
                    MoveRight(node, inf),
                ]),
            ]),

            SuccessIsFailure("Reset to center", child=Yaw(node, SEARCH_ANGLE)),

            Sequence("Check Left", memory=True, children=[
                Yaw(node, SEARCH_ANGLE),
                DetecedObject(TARGET_CLASS, conf_threshold=CONF_THRESHOLD),
                Yaw(node, -SEARCH_ANGLE),
                Parallel("Move to Target", policy=py_trees.common.ParallelPolicy.SuccessOnOne(), children=[
                    Retry(name="Retry MoveBasedOnDetection", child=MoveBasedOnDetection_Sway(node, TARGET_CLASS), num_failures=inf),
                    MoveLeft(node, inf),
                ]),
            ]),
        ]),

        # Fallback: approach until the target is detected or timeout.
        FailureIsSuccess(
            name="MoveBasedOnDetection",
            child=Timeout(name="Timeout MoveBasedOnDetection", duration=30,
                          child=Retry(name="Retry MoveBasedOnDetection_Surge",
                                      child=MoveBasedOnDetection_Sway_Surge(node, TARGET_CLASS),
                                      num_failures=inf))),
    ])
    # --- Automation Ends Here ---

    tree = py_trees.trees.BehaviourTree(root)
    tree.setup(timeout=1.0)

    count = 0
    RATE_HZ = 20.0          # Display update rate
    TICK_HZ = 20.0         # Actual tick rate (simulatet/apc pitch 0 real-time updates)
    period_display = 1.0 / RATE_HZ
    period_tick = 1.0 / TICK_HZ

    print("Starting in 5 seconds...")
    sleep(5)
    print("Starting...")
    last_display = perf_counter()
    next_tick = perf_counter()

    while True:
        now = perf_counter()

        # Tick the tree frequently (real-time)
        if now >= next_tick:
            tree.tick()
            next_tick += period_tick

        # Display status at a slower rate
        if now - last_display >= period_display:
            count += 1
            print(f"\n--- Tick {count} ---")
            child_states = ", ".join(f"{c.name}:{c.status.name}" for c in tree.root.children)
            print(py_trees.display.ascii_tree(tree.root, show_status=True))
            print("\n")
            last_display = now

        # Exit condition
        if tree.root.status != py_trees.common.Status.RUNNING:
            print("Tree finished execution.")
            break

        sleep(0.001)  # small sleep to avoid busy waiting
