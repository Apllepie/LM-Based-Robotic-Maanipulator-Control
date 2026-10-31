from pathlib import Path
import time
import threading
import numpy as np
import zmq

import mujoco
import mujoco.viewer

import robots.gripper_connection as gripper_connection

PROJECT_DIR = Path(__file__).resolve().parent.parent
ROBOT_PATH = PROJECT_DIR / "models" / "universal_robots_ur5e" / "scene.xml"
GRIPPER_PATH = PROJECT_DIR / "models" / "robotiq_2f85" / "2f85.xml"

ZMQ_ADDRESS = "tcp://*:5555"
# ZMQ_ADDRESS = "ipc:///tmp/mujoco.ipc"

MAX_JOINT_VELOCITY = 1.5  # rad/s dla przegubów
MAX_GRIPPER_VELOCITY = 255.0  # prędkość zmiany sygnału chwytaka

model, data = gripper_connection.create_robot(ROBOT_PATH, GRIPPER_PATH)

# Pobranie ID i zakresu sterowania chwytaka
gripper_actuator_id = gripper_connection.get_gripper_actuator_id(model)
ctrl_range = model.actuator_ctrlrange[gripper_actuator_id]
GRIPPER_MIN_CTRL, GRIPPER_MAX_CTRL = ctrl_range[0], ctrl_range[1]

# Pozycja startowa (Home Pose)
q_home = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
data.qpos[:6] = q_home
data.ctrl[:6] = q_home
# Chwytak domyślnie otwarty
data.ctrl[gripper_actuator_id] = GRIPPER_MIN_CTRL
mujoco.mj_forward(model, data)

# Cele pozycji dla przegubów i chwytaka
q_target = q_home.copy()
gripper_target = float(GRIPPER_MIN_CTRL)
lock = threading.Lock()


def zmq_server_thread():
    """Wątek serwera ZMQ do obsługi przegubów i chwytaka."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(ZMQ_ADDRESS)
    print(f"[MuJoCo 0MQ Server] Nasłuchiwanie na {ZMQ_ADDRESS}...")

    global gripper_target

    while True:
        try:
            payload = socket.recv_json()
            cmd_type = payload.get("type", "joint")

            if cmd_type == "joint":
                joint_idx = payload.get("joint_index")
                val = payload.get("angle_value")
                unit = payload.get("unit", "degrees")
                mode = payload.get("mode", "relative")

                if joint_idx is not None and 1 <= joint_idx <= 6:
                    angle_rad = np.radians(val) if unit == "degrees" else float(val)
                    idx = joint_idx - 1

                    with lock:
                        if mode == "relative":
                            q_target[idx] += angle_rad
                        else:
                            q_target[idx] = angle_rad

                        target_val = q_target[idx]

                    mode_str = "względnie o" if mode == "relative" else "bezwzględnie na"
                    msg = f"Przegub {joint_idx} zmieniony {mode_str} {val} {unit} (Cel: {np.degrees(target_val):.1f}° / {target_val:.3f} rad)."
                    socket.send_json({"status": "ok", "message": msg})
                else:
                    socket.send_json({"status": "error", "message": f"Nieprawidłowy numer przegubu: {joint_idx}"})

            elif cmd_type == "gripper":
                action = payload.get("action")
                val = payload.get("value", 0.0)

                with lock:
                    if action == "open":
                        gripper_target = float(GRIPPER_MIN_CTRL)
                        status_msg = "Chwytak otwarty."
                    elif action == "close":
                        gripper_target = float(GRIPPER_MAX_CTRL)
                        status_msg = "Chwytak zamknięty."
                    elif action == "set":
                        # Mapowanie wartości 0-100% na zakrojenie sygnału sterownika
                        pct = np.clip(val, 0.0, 100.0) / 100.0
                        gripper_target = GRIPPER_MIN_CTRL + pct * (GRIPPER_MAX_CTRL - GRIPPER_MIN_CTRL)
                        status_msg = f"Ustawiono otwarcie chwytaka na {val}%."
                    else:
                        status_msg = f"Nieznana akcja chwytaka: {action}"

                socket.send_json({"status": "ok", "message": status_msg})

        except Exception as e:
            try:
                socket.send_json({"status": "error", "message": f"Błąd serwera: {e}"})
            except Exception:
                pass


def main():
    server_thread = threading.Thread(target=zmq_server_thread, daemon=True)
    server_thread.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            dt = model.opt.timestep

            with lock:
                q_target_copy = q_target.copy()
                g_target_copy = gripper_target

            # 1. Płynna interpolacja dla 6 przegubów ramienia
            current_arm_ctrl = data.ctrl[:6].copy()
            arm_diff = q_target_copy - current_arm_ctrl
            max_arm_step = MAX_JOINT_VELOCITY * dt
            arm_step = np.clip(arm_diff, -max_arm_step, max_arm_step)
            data.ctrl[:6] = current_arm_ctrl + arm_step

            # 2. Płynna interpolacja dla chwytaka
            current_g_ctrl = data.ctrl[gripper_actuator_id]
            g_diff = g_target_copy - current_g_ctrl
            max_g_step = MAX_GRIPPER_VELOCITY * dt
            g_step = np.clip(g_diff, -max_g_step, max_g_step)
            data.ctrl[gripper_actuator_id] = current_g_ctrl + g_step

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    main()