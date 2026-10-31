import json
import os
import re
import sys
import zmq
from ollama import Client

from prompts import BASE_SYSTEM_INSTRUCTION, ROBOT_TOOLS

MODEL_NAME = "qwen3.5:4b"
OLLAMA_HOST = "http://172.22.144.1:11434"
HISTORY_FILE = "chat_history.json"
MAX_CONTEXT_MESSAGES = 20

ZMQ_ADDRESS = "tcp://127.0.0.1:5555"
# ZMQ_ADDRESS = "ipc:///tmp/mujoco.ipc"

zmq_context = zmq.Context()


def sanitize_surrogates(data):
    if isinstance(data, str):
        return re.sub(r'[\ud800-\udfff]', '', data)
    elif isinstance(data, dict):
        return {k: sanitize_surrogates(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_surrogates(item) for item in data]
    return data


def send_to_mujoco_simulation(payload: dict) -> str:
    """Wysyła ramkę JSON przez dedykowane gniazdo ZeroMQ REQ."""
    socket = zmq_context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 2000)
    socket.setsockopt(zmq.SNDTIMEO, 2000)

    try:
        socket.connect(ZMQ_ADDRESS)
        socket.send_json(payload)
        response = socket.recv_json()
        return response.get("message", "Wykonano.")
    except zmq.Again:
        return "[!] Błąd 0MQ: Timeout. Upewnij się, że symulacja `python3 src/main.py` jest uruchomiona."
    except Exception as e:
        return f"[!] Błąd komunikacji 0MQ: {e}"
    finally:
        socket.close()


class ChatBot:
    def __init__(self, host: str, model: str, history_file: str):
        self.client = Client(host=host)
        self.model = model
        self.history_file = history_file
        self.history = self.load_history()

    def load_history(self) -> list:
        if os.path.exists(self.history_file) and os.path.getsize(self.history_file) > 0:
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return sanitize_surrogates(data)
            except Exception as e:
                print(f"[!] Błąd wczytywania historii: {e}")
        return []

    def save_history(self):
        try:
            clean_history = sanitize_surrogates(self.history)
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(clean_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[!] Błąd zapisu historii: {e}")

    def get_messages_for_model(self) -> list:
        recent_history = self.history[-MAX_CONTEXT_MESSAGES:] if len(self.history) > MAX_CONTEXT_MESSAGES else self.history
        messages = [{"role": "system", "content": BASE_SYSTEM_INSTRUCTION}] + recent_history
        return sanitize_surrogates(messages)

    def ask(self, user_input: str):
        clean_input = sanitize_surrogates(user_input)
        self.history.append({"role": "user", "content": clean_input})
        messages_to_send = self.get_messages_for_model()

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages_to_send,
                tools=ROBOT_TOOLS,
                think=False,
                options={"temperature": 0.0}
            )

            msg = response.get("message", {})

            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump()
            elif hasattr(msg, "__dict__"):
                msg_dict = dict(msg)
            else:
                msg_dict = msg

            msg_dict = sanitize_surrogates(msg_dict)
            tool_calls = msg_dict.get("tool_calls", [])
            content = msg_dict.get("content", "")

            if tool_calls:
                self.history.append(msg_dict)

                for call in tool_calls:
                    fn_name = call.get("function", {}).get("name")
                    args = call.get("function", {}).get("arguments", {})

                    if fn_name == "rotate_joint_by":
                        payload = {
                            "type": "joint",
                            "joint_index": args.get("joint_index"),
                            "angle_value": args.get("delta_angle"),
                            "unit": args.get("unit", "degrees"),
                            "mode": "relative"
                        }
                    elif fn_name == "set_joint_to":
                        payload = {
                            "type": "joint",
                            "joint_index": args.get("joint_index"),
                            "angle_value": args.get("target_angle"),
                            "unit": args.get("unit", "degrees"),
                            "mode": "absolute"
                        }
                    elif fn_name == "control_gripper":
                        payload = {
                            "type": "gripper",
                            "action": args.get("action"),
                            "value": args.get("value", 0.0)
                        }
                    else:
                        continue

                    res_msg = send_to_mujoco_simulation(payload)
                    print(f"\n[AKCJA ROBOTA] -> {res_msg}\n")
                    self.history.append({"role": "tool", "content": res_msg})

                self.save_history()

            elif content:
                print(f"\nAsystent: {content}\n")
                self.history.append({"role": "assistant", "content": content})
                self.save_history()

        except Exception as e:
            print(f"\n[!] Błąd komunikacji z Ollamą: {e}")
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()


def main():
    bot = ChatBot(host=OLLAMA_HOST, model=MODEL_NAME, history_file=HISTORY_FILE)
    print(f"=== Sterownik CzatBot ZMQ ({MODEL_NAME}) ===")
    print(f"Załadowano wiadomości z historii: {len(bot.history)}")
    print(f"Połączenie ZMQ: {ZMQ_ADDRESS}\n")

    while True:
        try:
            user_input = input("Ty: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break
            if user_input.lower() == "clear":
                bot.history = []
                bot.save_history()
                print("[i] Pamięć czatu wyczyszczona.\n")
                continue

            bot.ask(user_input)

        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    main()