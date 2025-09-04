# config.py
import json
import os
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / 'api_config.json'

# Конфигурация по умолчанию
DEFAULT_CONFIG = {
    "llm": {
        "api_key": "",
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "api_url": "https://openrouter.ai/api/v1/chat/completions"
    },
    "qwen": {
        "api_key": "", 
        "model": "meta-llama/llama-3.3-8b-instruct:free",
        "api_url": "https://openrouter.ai/api/v1/chat/completions"
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
    
    # Если файла нет или ошибка, возвращаем конфигурацию по умолчанию
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

# Загружаем конфигурацию при импорте
LLM_CONFIG = load_config().get("llm", DEFAULT_CONFIG["llm"])
QWEN_CONFIG = load_config().get("qwen", DEFAULT_CONFIG["qwen"])
