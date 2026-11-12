"""Example: move forward along a square path.

Demonstrates py_trees concepts used most often in this workshop:
  * `memory=True` on Sequences so time-based actions (MoveForward, Yaw) are not
    restarted every tick and the tree resumes where it stopped.
  * A helper function to build repeated sub-trees cleanly.
  * Nested Sequences inside a parent Sequence for structured behaviour.

Run: uv run python src/automation_scripts/level_4_DIY_behavior_tree.py
"""

from time import sleep, perf_counter
import py_trees
from py_trees.decorators import Retry, SuccessIsFailure, FailureIsSuccess, Inverter, Timeout
from py_trees.composites import Selector, Sequence, Parallel
from minecraft.minecraft_override_config import MinecraftOverrideConfig
from pathlib import Path
from automation_scripts.actions import (
    InitWorld,
    SwitchCameraFront,
    SwitchCameraBottom,
    MoveForward,
    MoveBackward,
    MoveLeft,
    Yaw
)

# --- Configuration ---
SQUARE_SIDE_DURATION = 1.5   # seconds to move along one side of the square
TURN_ANGLE_DEG = 90.0        # degrees to turn at each corner (yaw setpoint)
NUMBER_OF_SIDES = 4          # number of sides to complete


if __name__ == "__main__":

    node = MinecraftOverrideConfig(override=True)

    root = py_trees.composites.Sequence("RootSequence", memory=True)

    # Build one side of the square: move forward then yaw to the next corner.
    def side_of_square(side_number: int):
        return Sequence(
            f"Side{side_number}", memory=True, children=[
                MoveForward(node, SQUARE_SIDE_DURATION),
                Yaw(node, TURN_ANGLE_DEG),
            ]
        )

    # --- Automation Starts Here ---
    root.add_children([
        InitWorld(node),
        SwitchCameraBottom(node),
        SwitchCameraFront(node),
        Sequence("MoveInSquare", memory=True, children=[
            side_of_square(1),
            side_of_square(2),
            side_of_square(3),
            side_of_square(4),
        ])
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
