from pathlib import Path
import time
import threading
import numpy as np
import zmq

import mujoco
import mujoco.viewer

import robots.gripper_connection as gripper_connection

# Scieżki do modeli
PROJECT_DIR = Path(__file__).resolve().parent.parent
ROBOT_PATH = PROJECT_DIR / "models" / "universal_robots_ur5e" / "scene.xml"
GRIPPER_PATH = PROJECT_DIR / "models" / "robotiq_2f85" / "2f85.xml"

# Adres komunikacji IPC przez ZeroMQ
#ZMQ_ADDRESS = "tcp://*:5555"
ZMQ_ADDRESS = "ipc:///tmp/mujoco.ipc"  # Adres IPC do komunikacji z MuJoCo

# 1. Tworzenie modelu i danych robota
model, data = gripper_connection.create_robot(ROBOT_PATH, GRIPPER_PATH)

# 2. Pozycja startowa (Home Pose)
q_home = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
data.qpos[:6] = q_home
data.ctrl[:6] = q_home
mujoco.mj_forward(model, data)

# Bufor pozycji docelowej i blokada wątkowa
q_target = q_home.copy()
q_lock = threading.Lock()


def zmq_server_thread():
    """Wątek serwera ZeroMQ odbierający polecenia ruchu w schemacie REP."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(ZMQ_ADDRESS)
    print(f"[MuJoCo 0MQ Server] Nasłuchiwanie na {ZMQ_ADDRESS}...")

    while True:
        try:
            payload = socket.recv_json()
            joint_idx = payload.get("joint_index")
            val = payload.get("angle_value")
            unit = payload.get("unit", "degrees")

            if joint_idx is not None and 1 <= joint_idx <= 6:
                # Przeliczenie stopni na radiany dla MuJoCo
                angle_rad = np.radians(val) if unit == "degrees" else float(val)
                
                with q_lock:
                    q_target[joint_idx - 1] = angle_rad

                msg = f"Przegub {joint_idx} ustawiony na {val} {unit} ({angle_rad:.3f} rad)."
                socket.send_json({"status": "ok", "message": msg})
            else:
                socket.send_json({"status": "error", "message": f"Nieprawidłowy numer przegubu: {joint_idx}"})

        except Exception as e:
            print(f"[!] Błąd serwera ZMQ: {e}")


def main():
    # Uruchomienie odbiornika ZeroMQ w tle
    server_thread = threading.Thread(target=zmq_server_thread, daemon=True)
    server_thread.start()

    # Pętla fizyki i wizualizacji MuJoCo
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            with q_lock:
                data.ctrl[:6] = q_target

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()