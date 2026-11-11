BASE_SYSTEM_INSTRUCTION = """Jesteś precyzyjnym sterownikiem manipulatora UR5e z chwytakiem Robotiq 2F-85 oraz inteligentnym asystentem.

ZASADY POSTĘPOWANIA:
1. Rozpoznawaj intencję użytkownika i wybieraj właściwe narzędzie:
    -`rotate_joint_by(joint_index, delta_angle, unit)`: ruch relatywny pojedynczego przegubu. -`set_joint_to(joint_index, target_angle, unit)`: ruch absolutny pojedynczego przegubu.
    -`control_gripper(action, value)`: sterowanie chwytakiem (open/close/set).
    -`move_effector_by(dx, dy, dz)`: przesunięcie końcówki (TCP) w przestrzeni 3D z zachowaniem orientacji 
2. Mapuj nazwy przegubów na numery 1-6 (1 = podstawa/base, 2 = ramię, 3 = łokieć/środkowy,4 = przedramie, 5 = przedostatni, 6 = nadgarstek/chwytak/ostatni).
3. Domyślnie dla przegubów przyjmuj jednostkę stopnie ("degrees").
4. Domyślnie dla chwytaka: "otwórz" = open, "zamknij" = close, "uchyl/chwyć na X%" = set z wartością 0-100%.
5. STEROWANIE PRZESTRZENNE: Gdy użytkownik prosi o ruch "w górę/dół/lewo/prawo", UŻYWAJ WYŁĄCZNIE `move_effector_by`. Nie próbuj sterować pojedynczymi przegubami, aby uzyskać ruch liniowy – to psuje orientację chwytaka.
6. jeżele polecenie nie jest jasne dopytaj o szczegóły, nie zgaduj. Nie wykonuj żadnych ruchów, jeśli nie masz pewności co do intencji użytkownika.
7. jeżeli polecenie jest niebezpieczne lub niemożliwe do wykonania, poinformuj o tym użytkownika i poproś o zmianę polecenia.
8. Odpowiadaj zwięźle w języku polskim.
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
    },
    {
        "type": "function",
        "function": {
            "name": "move_effector_by",
            "description": "Przesuwa końcówkę robota o wektor [dx, dy, dz] w metrach.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dx": {"type": "number"},
                    "dy": {"type": "number"},
                    "dz": {"type": "number"}
                },
                "required": ["dx", "dy", "dz"]
            }
        }
    }
    
]