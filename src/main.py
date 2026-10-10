import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(
    "/home/phuryn/projects/LM-Based-Robotic-Maanipulator-Control/models/universal_robots_ur5e/scene.xml"
)
data = mujoco.MjData(model)

print("Joints:", model.njnt)
print("Actuators:", model.nu)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        with viewer.lock():
            mujoco.mj_step(model, data)

        viewer.sync()