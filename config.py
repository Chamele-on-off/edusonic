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
        "default_mode": "traditional",  # "traditional" или "llm_first"
        "available_modes": ["traditional", "llm_first"]
    },
    "system_mode": {
        "default_mode": "normal",  # "normal" или "demo"
        "available_modes": ["normal", "demo"]
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

def get_system_mode():
    """Получение режима работы системы"""
    config = load_config()
    return config.get("system_mode", {}).get("default_mode", "normal")

def set_system_mode(mode):
    """Установка режима работы системы"""
    if mode not in ["normal", "demo"]:
        return False
        
    config = load_config()
    
    if 'system_mode' not in config:
        config['system_mode'] = {}
    
    config['system_mode']['default_mode'] = mode
    return save_config(config)

# Загружаем конфигурацию при импорте
LLM_CONFIG = load_config().get("llm", DEFAULT_CONFIG["llm"])
OPENROUTER_CONFIG = load_config().get("openrouter", DEFAULT_CONFIG["openrouter"])
FALLBACK_CONFIG = load_config().get("fallback", DEFAULT_CONFIG["fallback"])
