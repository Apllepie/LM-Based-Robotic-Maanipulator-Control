BASE_SYSTEM_INSTRUCTION = """Jesteś inteligentnym asystentem oraz precyzyjnym sterownikiem manipulatora UR5e z chwytakiem Robotiq 2F-85.

ZASADY POSTĘPOWANIA:
1. Rozpoznawaj intencję użytkownika i wybieraj właściwe narzędzie:
   - `rotate_joint_by`: zmiana/obrót obecnej pozycji przegubu (1-6) O podaną wartość (ruch względny).
   - `set_joint_to`: ustawienie konkretnego kąta docelowego przegubu (1-6) NA podaną wartość (ruch bezwzględny).
   - `control_gripper`: sterowanie chwytakiem (otwieranie, zamykanie lub ustawianie procentowe 0-100%).
2. Mapuj nazwy przegubów na numery 1-6 (1 = podstawa/base, 2 = ramię, 3 = łokieć/środkowy, 5 = przedostatni, 6 = nadgarstek).
3. Domyślnie dla przegubów przyjmuj jednostkę stopnie ("degrees").
4. Domyślnie dla chwytaka: "otwórz" = open, "zamknij" = close, "uchyl/chwyć na X%" = set z wartością 0-100%.
5. Odpowiadaj zwięźle w języku polskim.
"""

ROBOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rotate_joint_by",
            "description": "Obraca przegub O różnicę kąta względem obecnej pozycji (ruch względny/delta).",
            "parameters": {
                "type": "object",
                "properties": {
                    "joint_index": {"type": "integer", "description": "Numer przegubu (1-6)."},
                    "delta_angle": {"type": "number", "description": "Wartość kąta o jaką należy obrócić przegub."},
                    "unit": {"type": "string", "enum": ["degrees", "radians"], "description": "Jednostka kąta."}
                },
                "required": ["joint_index", "delta_angle", "unit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_joint_to",
            "description": "Ustawia przegub NA konkretną pozycję docelową w przestrzeni (ruch bezwzględny).",
            "parameters": {
                "type": "object",
                "properties": {
                    "joint_index": {"type": "integer", "description": "Numer przegubu (1-6)."},
                    "target_angle": {"type": "number", "description": "Docelowa wartość kąta."},
                    "unit": {"type": "string", "enum": ["degrees", "radians"], "description": "Jednostka kąta."}
                },
                "required": ["joint_index", "target_angle", "unit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "control_gripper",
            "description": "Steruje otwarciem/zamknięciem chwytaka Robotiq 2F-85.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["open", "close", "set"],
                        "description": "'open' - pełne otwarcie, 'close' - pełne zamknięcie, 'set' - ustawienie konkretnej wartości %."
                    },
                    "value": {
                        "type": "number",
                        "description": "Procent zamknięcia chwytaka (0 = w pełni otwarty, 100 = w pełni zamknięty). Wymagane tylko dla action='set'."
                    }
                },
                "required": ["action"]
            }
        }
    }
]