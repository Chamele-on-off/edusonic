import json
from pathlib import Path
from typing import Dict, Optional
import random

class DialogueKnowledge:
    def __init__(self):
        self.data = self._load_knowledge()
        
    def _load_knowledge(self) -> Dict:
        path = Path("materials/dialogue_knowledge.json")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "patterns": {
                    "как дела": ["Все хорошо, продолжим урок", "Отлично, давайте заниматься"],
                    "не понял": ["Давайте разберем еще раз", "Попробую объяснить по-другому"],
                    "повтори": ["Повторяю...", "Еще раз:"]
                }
            }

    def get_response(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for pattern, responses in self.data["patterns"].items():
            if pattern in text_lower:
                return random.choice(responses)
        return None
