import json
import os
from pathlib import Path
import random
from typing import Dict, List

CONFIG_FILE = Path(__file__).parent / 'api_config.json'

# Конфигурация по умолчанию
DEFAULT_CONFIG = {
    "llm": {
        "api_keys": {},
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

def update_api_key(provider, api_key, key_name=None):
    """Обновление API ключа для указанного провайдера"""
    config = load_config()
    
    if provider in config:
        if provider == 'llm' and key_name:
            # Для LLM провайдера используем словарь ключей
            if 'api_keys' not in config[provider]:
                config[provider]['api_keys'] = {}
            config[provider]['api_keys'][key_name] = api_key.strip()
        else:
            # Для других провайдеров обычный ключ
            config[provider]["api_key"] = api_key.strip()
        return save_config(config)
    
    return False

def get_api_key(provider, key_name=None):
    """Получение API ключа для указанного провайдера"""
    config = load_config()
    
    if provider == 'llm' and key_name:
        # Для LLM провайдера получаем конкретный ключ по имени
        return config.get(provider, {}).get('api_keys', {}).get(key_name, "")
    elif provider == 'llm':
        # Для LLM провайдера без указания имени - случайный ключ
        api_keys = config.get(provider, {}).get('api_keys', {})
        if api_keys:
            return random.choice(list(api_keys.values()))
        return ""
    else:
        # Для других провайдеров
        return config.get(provider, {}).get("api_key", "")

def get_all_llm_keys():
    """Получение всех LLM ключей"""
    config = load_config()
    return config.get('llm', {}).get('api_keys', {})

def get_llm_keys_count():
    """Получение количества LLM ключей"""
    config = load_config()
    return len(config.get('llm', {}).get('api_keys', {}))

def delete_llm_key(key_name):
    """Удаление LLM ключа"""
    config = load_config()
    
    if 'llm' in config and 'api_keys' in config['llm']:
        if key_name in config['llm']['api_keys']:
            del config['llm']['api_keys'][key_name]
            return save_config(config)
    
    return False

def get_next_llm_key_name():
    """Генерация следующего имени для LLM ключа"""
    config = load_config()
    existing_keys = config.get('llm', {}).get('api_keys', {}).keys()
    
    # Ищем следующее доступное имя в формате lm001, lm002, etc.
    for i in range(1, 1000):
        key_name = f"lm{i:03d}"
        if key_name not in existing_keys:
            return key_name
    
    return f"lm{len(existing_keys) + 1:03d}"

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

# Загружаем конфигурацию при импорте
LLM_CONFIG = load_config().get("llm", DEFAULT_CONFIG["llm"])
OPENROUTER_CONFIG = load_config().get("openrouter", DEFAULT_CONFIG["openrouter"])
FALLBACK_CONFIG = load_config().get("fallback", DEFAULT_CONFIG["fallback"])