"""Example: robust navigation with retries and a safety timeout.

Demonstrates py_trees concepts used most often in this workshop:
  * `Retry` to keep trying a failing step (useful for detection-based actions).
  * `Timeout` to bound how long any part of the tree can run, so the robot does
    not move forever if something fails.
  * Wrapping the whole root with an overall timeout as a safety net.
  * Using `inf` durations on movement actions that should only stop when told.

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
    DetecedObject
)

# --- Configuration ---
NAV_TARGET = "Gate"                 # exact label from your detection model
SEARCH_ANGLE = 40.0                 # degrees to yaw left/right while searching
MAX_NAV_TIME = 120.0                # overall safety timeout in seconds


if __name__ == "__main__":

    node = MinecraftOverrideConfig(override=True)

    root = py_trees.composites.Sequence("RootSequence", memory=True)

    # --- Automation Starts Here ---
    root.add_children([
        InitWorld(node),
        SwitchCameraBottom(node),
        SwitchCameraFront(node),

        # Navigate to the gate: keep trying until the detection is found and centred.
        Retry(name="Retry Navigate to Gate", child=Sequence("Navigate to Gate", memory=True, children=[
            Selector("Find Gate", memory=True, children=[
                Sequence("Check Gate Right", memory=True, children=[
                    Yaw(node, -SEARCH_ANGLE),
                    DetecedObject(NAV_TARGET, conf_threshold=0.8),
                    Yaw(node, SEARCH_ANGLE),
                    Parallel("Move to Gate", policy=py_trees.common.ParallelPolicy.SuccessOnOne(), children=[
                        Retry(name="Retry MoveBasedOnDetection", child=MoveBasedOnDetection_Sway(node, NAV_TARGET), num_failures=inf),
                        MoveRight(node, inf),
                    ]),
                ]),
                SuccessIsFailure("Reset to center", child=Yaw(node, SEARCH_ANGLE)),
                Sequence("Check Gate Left", memory=True, children=[
                    Yaw(node, SEARCH_ANGLE),
                    DetecedObject(NAV_TARGET, conf_threshold=0.8),
                    Yaw(node, -SEARCH_ANGLE),
                    Parallel("Move to Gate", policy=py_trees.common.ParallelPolicy.SuccessOnOne(), children=[
                        Retry(name="Retry MoveBasedOnDetection", child=MoveBasedOnDetection_Sway(node, NAV_TARGET), num_failures=inf),
                        MoveLeft(node, inf),
                    ]),
                ]),
            ]),
            MoveForward(node, 10),
        ]), num_failures=inf),

        # Leave the pail area: a simple time-limited sequence.
        Sequence("Leave Area", memory=True, children=[
            SwitchCameraBottom(node),
            MoveBackward(node, 5),
        ]),
    ])

    # Safety net: stop everything if the tree runs longer than MAX_NAV_TIME.
    root = Timeout(name="Overall Timeout", duration=MAX_NAV_TIME, child=root)
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
