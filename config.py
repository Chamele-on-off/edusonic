import json
import os
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / 'api_config.json'

# Конфигурация по умолчанию
DEFAULT_CONFIG = {
    "llm": {
        "api_key": "",
        "model": "meta-llama/llama-3.3-8b-instruct:free",
        "api_url": "https://openrouter.ai/api/v1/chat/completions",
        "provider": "openrouter"
    },
    "openrouter": {
        "api_key": "",
        "model": "meta-llama/llama-3.3-8b-instruct:free",
        "api_url": "https://openrouter.ai/api/v1/chat/completions"
    },
    "fallback": {
        "enabled": True,
        "model": "meta-llama/llama-3.3-8b-instruct:free"
    },
    "llm_query_mode": {
        "default_mode": "traditional",
        "available_modes": ["traditional", "llm_first"]
    },
    "dialogue_settings": {
        "max_response_length": 3,
        "context_window": 8,
        "subject_selection_prompt": "Ты - дружелюбный учитель. Сначала ответь на вопрос ученика кратко и понятно (1-2 предложения), "
                                  "а затем мягко подведи к выбору предмета для урока. "
                                  "Не переключайся резко на выбор урока, сначала ответь на вопрос.",
        "general_prompt": "Ты - helpful учитель. Отвечай на вопросы ученика кратко и информативно (1-2 предложения). "
                         "Будь дружелюбным и поддерживающим.",
        "greeting_prompt": "Ты - приветливый учитель. Познакомься с учеником, ответь на приветствие, "
                          "спроси как дела или что интересует, и мягко подведи к выбору предмета."
    }
}

def load_config():
    """Загрузка конфигурации из файла"""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {e}")
    
    return DEFAULT_CONFIG

def save_config(config):
    """Сохранение конфигурации в файл"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Ошибка сохранения конфигурации: {e}")
        return False

def update_api_key(provider, api_key):
    """Обновление API ключа для указанного провайдера"""
    config = load_config()
    
    if provider in config:
        config[provider]["api_key"] = api_key.strip()
        return save_config(config)
    
    return False

def get_api_key(provider):
    """Получение API ключа для указанного провайдера"""
    config = load_config()
    return config.get(provider, {}).get("api_key", "")

def get_model_config(provider):
    """Получение конфигурации модели для указанного провайдера"""
    config = load_config()
    return config.get(provider, {})

def get_llm_mode():
    """Получение режима работы LLM"""
    config = load_config()
    return config.get("llm_query_mode", {}).get("default_mode", "traditional")

def set_llm_mode(mode):
    """Установка режима работы LLM"""
    if mode not in ["traditional", "llm_first"]:
        return False
        
    config = load_config()
    
    if 'llm_query_mode' not in config:
        config['llm_query_mode'] = {}
    
    config['llm_query_mode']['default_mode'] = mode
    return save_config(config)

def get_dialogue_settings():
    """Получение настроек диалога"""
    config = load_config()
    return config.get("dialogue_settings", DEFAULT_CONFIG["dialogue_settings"])

# Загружаем конфигурацию при импорте
LLM_CONFIG = load_config().get("llm", DEFAULT_CONFIG["llm"])
OPENROUTER_CONFIG = load_config().get("openrouter", DEFAULT_CONFIG["openrouter"])
FALLBACK_CONFIG = load_config().get("fallback", DEFAULT_CONFIG["fallback"])
DIALOGUE_SETTINGS = get_dialogue_settings()