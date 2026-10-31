import json
import os
import sys
from ollama import Client

MODEL_NAME = "qwen3.5:4b"
OLLAMA_HOST = "http://172.22.144.1:11434"
HISTORY_FILE = "chat_history.json"
MAX_CONTEXT_MESSAGES = 10

BASE_SYSTEM_INSTRUCTION = """Jesteś inteligentnym asystentem oraz precezyjnym sterownikiem manipulatora UR5e (6 przegubów).

ZASADY:
1. Przetwarzaj polecenia obrotu przegubów na wywołanie funkcji `set_joint_angle`.
2. Mapuj nazwy przegubów na numery 1-6 (np. "przedostatni" -> 5, "pierwszy" -> 1, "nadgarstek" -> 6).
3. Rozpoznawaj jednostki (stopnie lub radiany). Jeśli użytkownik nie poda jednostki, zdeceduj w zależności od wartości kąta.
4. Pamiętaj fakty, informacje o otoczeniu i tp., Odpowiadaj na nie na podstawie historii czatu.
5. Odpowiadaj zwięźle i dosłownie w języku polskim.
"""

ROBOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_joint_angle",
            "description": "Ustawia kąt wybranego przegubu robota.",
            "parameters": {
                "type": "object",
                "properties": {
                    "joint_index": {
                        "type": "integer",
                        "description": "Numer przegubu (1-6)."
                    },
                    "angle_value": {
                        "type": "number",
                        "description": "Wartość kąta."
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["degrees", "radians"],
                        "description": "Jednostka kąta: 'degrees' lub 'radians'."
                    }
                },
                "required": ["joint_index", "angle_value", "unit"]
            }
        }
    }
]


class ChatBot:
    def __init__(self, host: str, model: str, history_file: str):
        self.client = Client(host=host)
        self.model = model
        self.history_file = history_file
        self.history = self.load_history()

    def load_history(self) -> list:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception as e:
                print(f"[!] Błąd wczytywania historii: {e}")
        return []

    def save_history(self):
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[!] Błąd zapisu historii: {e}")

    def get_messages_for_model(self) -> list:
        recent_history = self.history[-MAX_CONTEXT_MESSAGES:] if len(self.history) > MAX_CONTEXT_MESSAGES else self.history
        return [{"role": "system", "content": BASE_SYSTEM_INSTRUCTION}] + recent_history

    def ask(self, user_input: str):
        self.history.append({"role": "user", "content": user_input})
        messages_to_send = self.get_messages_for_model()

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages_to_send,
                tools=ROBOT_TOOLS,
                options={"temperature": 0.0}
            )

            msg = response.get("message", {})
            
            # Konwersja obiektu `ollama.Message` na czysty dict Pythona (JSON serializable)
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump()
            elif hasattr(msg, "__dict__"):
                msg_dict = dict(msg)
            else:
                msg_dict = msg

            tool_calls = msg_dict.get("tool_calls", [])
            content = msg_dict.get("content", "")

            if tool_calls:
                for call in tool_calls:
                    fn_name = call.get("function", {}).get("name")
                    args = call.get("function", {}).get("arguments", {})
                    
                    print(f"\n[AKCJA ROBOTA] -> Wykonaj: {fn_name}(przegub={args.get('joint_index')}, kąt={args.get('angle_value')}, jednostka='{args.get('unit')}')\n")

                # Zapisujemy do historii przetworzony słownik `msg_dict`
                self.history.append(msg_dict)
                self.save_history()

            elif content:
                print(f"\nAsystent: {content}\n")
                self.history.append({"role": "assistant", "content": content})
                self.save_history()

        except Exception as e:
            print(f"\n[!] Błąd komunikacji z Ollama: {e}")
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()


def main():
    bot = ChatBot(host=OLLAMA_HOST, model=MODEL_NAME, history_file=HISTORY_FILE)
    print(f"=== Sterownik przegubów zainicjalizowany ({MODEL_NAME}) ===")
    print(f"Załadowano wiadomości z historii: {len(bot.history)}")

    print("Dostępne komendy: 'exit' (wyjście), 'clear' (wyczyszczenie pamięci czatu)\n")

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
                print("[i] Pamięć czatu wyczyszczona.")
                continue

            bot.ask(user_input)

        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    main()