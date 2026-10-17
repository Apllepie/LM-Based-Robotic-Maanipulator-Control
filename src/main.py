from pathlib import Path
import time
import numpy as np

import mujoco
import mujoco.viewer

import robots.gripper_connection as gripper_connection
from trajectory import CircleTrajectory
from kinematics import DifferentialIK

PROJECT_DIR = Path(__file__).resolve().parent.parent

ROBOT_PATH = PROJECT_DIR / "models" / "universal_robots_ur5e" / "scene.xml"
GRIPPER_PATH = PROJECT_DIR / "models" / "robotiq_2f85" / "2f85.xml"

# 1. Tworzenie modelu i danych robota
model, data = gripper_connection.create_robot(ROBOT_PATH, GRIPPER_PATH)

# 2. Pozycja startowa (Home Pose)
q_home = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
data.qpos[:6] = q_home
data.ctrl[:6] = q_home
mujoco.mj_forward(model, data)

# 3. Przygotowanie trajektorii okręgu
ee_body_name = "wrist_3_link"
start_pos = data.body(ee_body_name).xpos.copy()
radius = 0.1
circle_center = start_pos - np.array([radius, 0.0, 0.0])

circle_trajectory = CircleTrajectory(center=circle_center, radius=radius, duration=5.0)

# 4. Inicjalizacja modułu kinematyki i zmiennych sterujących
ik_solver = DifferentialIK(model, data, body_name=ee_body_name, damping=1e-2)
q_target = q_home.copy()

# 5. Pętla symulacji
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = data.time

        # Odczyt pozycji z trajektorii i z symulatora
        target_pos = circle_trajectory.generate_circle_waypoints(t)
        current_pos = data.body(ee_body_name).xpos

        # Obliczenie wymaganych prędkości przegubów przez zewnętrzny moduł
        dq = ik_solver.compute_joint_velocities(
            target_pos=target_pos,
            current_pos=current_pos,
            q_current=data.qpos[:6],
            q_home=q_home,
            kp_task=5.0,
            kp_null=1.0,
            max_dq=1.5,
        )

        # Integracja celu i wysłanie do sterownika
        q_target = q_target + dq * model.opt.timestep
        data.ctrl[:6] = q_target

        # Krok fizyki
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)