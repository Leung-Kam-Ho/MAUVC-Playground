from time import sleep, perf_counter
import py_trees
from py_trees.decorators import Retry, SuccessIsFailure, FailureIsSuccess, Inverter, Timeout
from py_trees.composites import Selector, Sequence,Parallel
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


if __name__ == "__main__":
    
    node = MinecraftOverrideConfig(override=True)

    root = py_trees.composites.Sequence("RootSequence", memory=True)
    
    # you can add your own actions here to create a behavior tree.
    # below is the list of available actions you can use:
    # 1) InitWorld(node)
    # 2) SwitchCameraFront(node)
    # 3) SwitchCameraBottom(node)
    # 4) Move(name, twist)
    # 5) MoveForward(node, distance)
    # 6) MoveBackward(node, distance)
    # 7) MoveLeft(node, distance)
    # 8) MoveRight(node, distance)
    # 9) Yaw(node, angle)
    # 10) Stop(node)
    # 11) MoveBasedOnDetection_Sway(node, detection_name)
    # 12) MoveBasedOnDetection_Sway_Surge(node, detection_name)
    # 13) DetectedObject(detection_name)



    # --- Automation Starts Here ---
    root.add_children([
        InitWorld(node),
        SwitchCameraBottom(node),
        SwitchCameraFront(node),
        # Continue adding your own actions here to create a behavior tree.
        Sequence("Something Something", memory=True, children=[
            
        ])

    ])
    # --- Automation Ends Here ---



    tree = py_trees.trees.BehaviourTree(root)
    tree.setup(timeout=1.0)

    # # Directory to store rendered tree images
    # path = Path() / "media"
    # path.mkdir(parents=True, exist_ok=True)
    # py_trees.display.render_dot_tree(tree.root, name="behavior_tree_based", target_directory=path)

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

        sleep(0.001)  # small sleep to avoid busy waitingInitWorld(node),
