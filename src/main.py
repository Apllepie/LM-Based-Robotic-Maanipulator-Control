from pathlib import Path

import mujoco
import mujoco.viewer

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Путь к XML относительно корня проекта
MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "universal_robots_ur5e"
    / "scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print("Joints:", model.njnt)
print("Actuators:", model.nu)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        with viewer.lock():
            mujoco.mj_step(model, data)

        viewer.sync()