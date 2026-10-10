from pathlib import Path

import mujoco
import mujoco.viewer

import robots.gripper_connection as gripper_connection

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROJECT_DIR = Path(__file__).resolve().parent.parent

ROBOT_PATH = (
    PROJECT_DIR
    / "models"
    / "universal_robots_ur5e"
    / "scene.xml"
)

GRIPPER_PATH = (
    PROJECT_DIR
    / "models"
    / "robotiq_2f85"
    / "2f85.xml"
)

model, data = gripper_connection.create_robot(ROBOT_PATH, GRIPPER_PATH)

print("Joints:", model.njnt)
print("Actuators:", model.nu)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        with viewer.lock():
            mujoco.mj_step(model, data)

        viewer.sync()