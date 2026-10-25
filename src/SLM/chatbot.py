import json
import os
import sys
from ollama import Client

# =====================================================================
# KONFIGURACJA I PARAMETRY RETENCJI
# =====================================================================
MODEL_NAME = "qwen3.5:4b"
OLLAMA_HOST = "http://172.22.144.1:11434"
HISTORY_FILE = "chat_history.json"

# RETENCJA KONTEKSTU:
# Ile ostatnich wypowiedzi model pamięta w bieżącej rozmowie.
# Dla małych modeli (4B) 4-6 to idealny kompromis (nie gubi wątku i nie miesza tematów).
MAX_CONTEXT_MESSAGES = 6

# STAN ŚWIATA (Pamięć twarda / Fakty):
# To są szufladki z danymi. Model NIE traktuje ich jak luźnej pogawędki.
WORLD_STATE = {
    "preferencje_uzytkownika": [
        "Lubi jabłka",
        "Nienawidzi kukurydzy"
    ],
    "obiekty_fizyczne": [
        "czerwony klocek",
        "niebieski klocek",
        "zielony klocek"
    ],
    "stan_manipulatora": "gotowy, chwytak pusty"
}

# SYSTEM PROMPT:
# Twarde reguły logiczne eliminujące metafory i mieszanie tematów
BASE_SYSTEM_INSTRUCTION = """Jesteś precyzyjnym asystentem sterującym i analitycznym.

ZASADY POSTĘPOWANIA:
1. Odpowiadaj KRÓTKO, PRECYZYJNIE i DOSŁOWNIE.
2. Odpowiadaj WYŁĄCZNIE w poprawnym języku polskim. Nigdy nie używaj znaków chińskich ani obcych słów.
3. KATEGORYCZNY ZAKAZ tworzenia metafor i łączenia niepowiązanych kategorii (np. klocki to NIE jedzenie, kolory klocków nie mają nic wspólnego z warzywami).
4. Pytany o stan otoczenia, klocki lub preferencje – czerp wiedzę WYŁĄCZNIE z sekcji AKTUALNY STAN ŚWIATA.
"""


class ChatBot:
    def __init__(self, host: str, model: str, history_file: str):
        self.client = Client(host=host)
        self.model = model
        self.history_file = history_file
        self.history = self.load_history()

    def load_history(self) -> list:
        """Bezpieczne wczytywanie historii z JSON."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception as e:
                print(f"[!] Błąd pliku historii ({e}). Rozpoczynam z czystą pamięcią.")
        return []

    def save_history(self):
        """Zapis historii do JSON z zabezpieczeniem formatowania."""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[!] Błąd zapisu historii: {e}")

    def build_system_prompt(self) -> dict:
        """
        Buduje dynamiczny prompt systemowy wstrzykując twarde fakty (Oddzielenie Stanu od Rozmowy).
        """
        facts_block = (
            "\n### AKTUALNY STAN ŚWIATA (FAKTY TWARDE):\n"
            f"- Preferencje użytkownika: {'; '.join(WORLD_STATE['preferencje_uzytkownika'])}\n"
            f"- Fizyczne obiekty na stole: {', '.join(WORLD_STATE['obiekty_fizyczne'])}\n"
            f"- Stan robota: {WORLD_STATE['stan_manipulatora']}\n"
            "###\n"
        )
        return {
            "role": "system",
            "content": BASE_SYSTEM_INSTRUCTION + facts_block
        }

    def get_messages_for_model(self) -> list:
        """
        Mechanizm retencji Sliding Window:
        Prompt systemowy ze stanem świata + maksymalnie N ostatnich wiadomości.
        """
        recent_history = self.history[-MAX_CONTEXT_MESSAGES:] if len(self.history) > MAX_CONTEXT_MESSAGES else self.history
        return [self.build_system_prompt()] + recent_history

    def ask(self, user_input: str):
        # 1. Dodajemy bieżące zapytanie użytkownika
        self.history.append({"role": "user", "content": user_input})
        messages_to_send = self.get_messages_for_model()

        try:
            print("\nAsystent: ", end="", flush=True)

            # 2. Wywołanie modelu z rygorystycznymi parametrami retencji i braku halucynacji
            response_stream = self.client.chat(
                model=self.model,
                messages=messages_to_send,
                stream=True,
                think=False,  # Wyłącza ukryty strumień reasoning (naprawia puste odpowiedzi)
                options={
                    "temperature": 0.1,  # Prawie zerowa temperatura = brak fantazjowania, czysta logika
                    "top_p": 0.9
                }
            )

            assistant_response = ""
            for chunk in response_stream:
                content = chunk.get('message', {}).get('content', '')
                if content:
                    print(content, end="", flush=True)
                    assistant_response += content

            print("\n")

            # 3. Zabezpieczenie przed zatruciem historii pustym stringiem
            if assistant_response.strip():
                self.history.append({"role": "assistant", "content": assistant_response})
                self.save_history()
            else:
                print("[!] Ostrzeżenie: Odebrano pustą odpowiedź. Pamięć nie została zanieczyszczona.")
                self.history.pop()

        except Exception as e:
            print(f"\n[!] Błąd komunikacji z Ollamą: {e}")
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()


def main():
    bot = ChatBot(host=OLLAMA_HOST, model=MODEL_NAME, history_file=HISTORY_FILE)
    
    print(f"=== ChatBot ({MODEL_NAME}) zainicjalizowany ===")
    print(f"Załadowano wiadomości z historii: {len(bot.history)}")
    print("Dostępne komendy: 'exit' (wyjście), 'clear' (wyczyszczenie pamięci czatu)\n")

    while True:
        try:
            user_input = input("Ty: ").strip()
            
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("Zamykanie bota.")
                break

            if user_input.lower() == "clear":
                bot.history = []
                bot.save_history()
                print("[i] Pamięć czatu została całkowicie wyczyszczona.\n")
                continue

            bot.ask(user_input)

        except KeyboardInterrupt:
            print("\nZakończono działanie programu.")
            sys.exit(0)


if __name__ == "__main__":
    main()