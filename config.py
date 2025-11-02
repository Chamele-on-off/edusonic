import json
import os
from pathlib import Path
import time

CONFIG_FILE = Path(__file__).parent / 'api_config.json'

# Конфигурация по умолчанию
DEFAULT_CONFIG = {
    "llm": {
        "api_key": "",
        "model": "meta-llama/llama-3.3-8b-instruct:free",
        "api_url": "https://openrouter.ai/api/v1/chat/completions",
        "provider": "openrouter",
        "max_tokens": 500,
        "temperature": 0.4,
        "timeout": 120,
        "priority": "openrouter_first"
    },
    "openrouter": {
        "api_key": "",
        "model": "meta-llama/llama-3.3-8b-instruct:free",
        "api_url": "https://openrouter.ai/api/v1/chat/completions",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    "local_llm": {
        "enabled": True,
        "base_url": "http://localhost:11434",
        "model": "llama3.2:3b",
        "timeout": 120,
        "max_retries": 2
    },
    "fallback": {
        "enabled": True,
        "model": "meta-llama/llama-3.3-8b-instruct:free",
        "max_retries": 3,
        "retry_delay": 2.0
    },
    "llm_query_mode": {
        "default_mode": "traditional",
        "available_modes": ["traditional", "llm_first"],
        "auto_switch_threshold": 5
    },
    "dialogue_settings": {
        "max_response_length": 3,
        "context_window": 10,
        "subject_selection_prompt": "Ты - дружелюбный учитель. Помоги ученику выбрать предмет для изучения. Будь кратким и понятным. Отвечай на русском языке. Поддерживай естественный диалог.",
        "dialogue_timeout": 30,
        "max_llm_retries": 2,
        "confidence_threshold": 0.8,
        "enable_context_awareness": True,
        "max_conversation_history": 20
    },
    "knowledge_base": {
        "similarity_threshold": 0.5,
        "llm_similarity_threshold": 0.8,
        "max_search_results": 5,
        "auto_save_interval": 300,
        "backup_enabled": True,
        "backup_count": 5
    },
    "audio_settings": {
        "speech_rate": 1.0,
        "voice_type": "female",
        "language": "ru",
        "volume": 1.0,
        "timeout": 10,
        "chunk_size": 1024
    },
    "animation_settings": {
        "fps": 10,
        "blink_interval": 30,
        "mouth_move_speed": 0.1,
        "smooth_transitions": True,
        "idle_animations": True,
        "idle_timeout": 5
    },
    "caching": {
        "enabled": True,
        "max_size": 1000,
        "ttl": 3600,
        "persistent": True,
        "auto_cleanup": True
    },
    "logging": {
        "level": "INFO",
        "file": "app.log",
        "max_size": 10485760,
        "backup_count": 5,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    },
    "performance": {
        "max_threads": 10,
        "request_timeout": 30,
        "connection_pool_size": 10,
        "keepalive": True,
        "compression": True
    },
    "security": {
        "cors_enabled": True,
        "allowed_origins": ["*"],
        "rate_limiting": True,
        "max_requests_per_minute": 60,
        "api_key_required": False
    },
    "ui_settings": {
        "theme": "light",
        "language": "ru",
        "avatar_size": "medium",
        "show_timestamps": True,
        "auto_scroll": True,
        "font_size": 14
    },
    "version": "2.0.0",
    "environment": "development"
}

def load_config():
    """Загрузка конфигурации из файла"""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print("✅ Конфигурация успешно загружена")
                return config
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
    
    # Если файла нет или ошибка, возвращаем конфигурацию по умолчанию
    print("⚠️ Используется конфигурация по умолчанию")
    return DEFAULT_CONFIG

def save_config(config):
    """Сохранение конфигурации в файл"""
    try:
        # Создаем backup текущего конфига если он существует
        if CONFIG_FILE.exists():
            backup_file = CONFIG_FILE.parent / f"api_config_backup_{int(time.time())}.json"
            try:
                import shutil
                shutil.copy2(CONFIG_FILE, backup_file)
                print(f"✅ Создан backup конфигурации: {backup_file.name}")
            except Exception as backup_error:
                print(f"⚠️ Не удалось создать backup: {backup_error}")
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print("✅ Конфигурация успешно сохранена")
        return True
    except Exception as e:
        print(f"❌ Ошибка сохранения конфигурации: {e}")
        return False

def update_api_key(provider, api_key):
    """Обновление API ключа для указанного провайдера"""
    config = load_config()
    
    if provider in config:
        old_key = config[provider].get("api_key", "")
        config[provider]["api_key"] = api_key.strip()
        
        if save_config(config):
            print(f"✅ API ключ для {provider} успешно обновлен")
            
            # Логируем изменение (без самого ключа)
            if old_key:
                print(f"🔑 Ключ изменен: {old_key[:4]}... -> {api_key[:4]}...")
            else:
                print(f"🔑 Новый ключ установлен: {api_key[:4]}...")
                
            return True
    
    print(f"❌ Не удалось обновить API ключ для {provider}")
    return False

def get_api_key(provider):
    """Получение API ключа для указанного провайдера"""
    config = load_config()
    return config.get(provider, {}).get("api_key", "")

def get_model_config(provider):
    """Получение конфигурации модели для указанного провайдера"""
    config = load_config()
    provider_config = config.get(provider, {})
    
    # Возвращаем полную конфигурацию провайдера
    return provider_config.copy()

def get_llm_mode():
    """Получение режима работы LLM"""
    config = load_config()
    return config.get("llm_query_mode", {}).get("default_mode", "traditional")

def set_llm_mode(mode):
    """Установка режима работы LLM"""
    if mode not in ["traditional", "llm_first"]:
        print(f"❌ Неверный режим: {mode}. Допустимые значения: 'traditional', 'llm_first'")
        return False
        
    config = load_config()
    
    if 'llm_query_mode' not in config:
        config['llm_query_mode'] = {}
    
    old_mode = config['llm_query_mode'].get('default_mode', 'traditional')
    config['llm_query_mode']['default_mode'] = mode
    
    if save_config(config):
        print(f"✅ Режим LLM изменен: {old_mode} -> {mode}")
        return True
    else:
        print("❌ Не удалось сохранить изменения режима LLM")
        return False

def get_llm_priority():
    """Получение приоритета моделей LLM"""
    config = load_config()
    return config.get("llm", {}).get("priority", "local_first")

def set_llm_priority(priority):
    """Установка приоритета моделей LLM"""
    valid_priorities = ["local_first", "openrouter_first", "local_only", "openrouter_only"]
    
    if priority not in valid_priorities:
        print(f"❌ Неверный приоритет: {priority}. Допустимые значения: {valid_priorities}")
        return False
        
    config = load_config()
    
    if 'llm' not in config:
        config['llm'] = {}
    
    old_priority = config['llm'].get('priority', 'local_first')
    config['llm']['priority'] = priority
    
    if save_config(config):
        print(f"✅ Приоритет LLM изменен: {old_priority} -> {priority}")
        return True
    else:
        print("❌ Не удалось сохранить изменения приоритета LLM")
        return False

def get_dialogue_settings():
    """Получение настроек диалога"""
    config = load_config()
    dialogue_settings = config.get("dialogue_settings", {})
    
    # Объединяем с настройками по умолчанию
    default_dialogue = DEFAULT_CONFIG.get("dialogue_settings", {})
    return {**default_dialogue, **dialogue_settings}

def get_knowledge_settings():
    """Получение настроек базы знаний"""
    config = load_config()
    knowledge_settings = config.get("knowledge_base", {})
    
    # Объединяем с настройками по умолчанию
    default_knowledge = DEFAULT_CONFIG.get("knowledge_base", {})
    return {**default_knowledge, **knowledge_settings}

def get_audio_settings():
    """Получение настроек аудио"""
    config = load_config()
    audio_settings = config.get("audio_settings", {})
    
    # Объединяем с настройками по умолчанию
    default_audio = DEFAULT_CONFIG.get("audio_settings", {})
    return {**default_audio, **audio_settings}

def get_animation_settings():
    """Получение настроек анимации"""
    config = load_config()
    animation_settings = config.get("animation_settings", {})
    
    # Объединяем с настройками по умолчанию
    default_animation = DEFAULT_CONFIG.get("animation_settings", {})
    return {**default_animation, **animation_settings}

def get_cache_settings():
    """Получение настроек кэширования"""
    config = load_config()
    cache_settings = config.get("caching", {})
    
    # Объединяем с настройками по умолчанию
    default_cache = DEFAULT_CONFIG.get("caching", {})
    return {**default_cache, **cache_settings}

def get_logging_settings():
    """Получение настроек логирования"""
    config = load_config()
    logging_settings = config.get("logging", {})
    
    # Объединяем с настройками по умолчанию
    default_logging = DEFAULT_CONFIG.get("logging", {})
    return {**default_logging, **logging_settings}

def get_performance_settings():
    """Получение настроек производительности"""
    config = load_config()
    performance_settings = config.get("performance", {})
    
    # Объединяем с настройками по умолчанию
    default_performance = DEFAULT_CONFIG.get("performance", {})
    return {**default_performance, **performance_settings}

def get_security_settings():
    """Получение настроек безопасности"""
    config = load_config()
    security_settings = config.get("security", {})
    
    # Объединяем с настройками по умолчанию
    default_security = DEFAULT_CONFIG.get("security", {})
    return {**default_security, **security_settings}

def get_ui_settings():
    """Получение настроек интерфейса"""
    config = load_config()
    ui_settings = config.get("ui_settings", {})
    
    # Объединяем с настройками по умолчанию
    default_ui = DEFAULT_CONFIG.get("ui_settings", {})
    return {**default_ui, **ui_settings}

def get_local_llm_settings():
    """Получение настроек локальной LLM"""
    config = load_config()
    local_settings = config.get("local_llm", {})
    
    # Объединяем с настройками по умолчанию
    default_local = DEFAULT_CONFIG.get("local_llm", {})
    return {**default_local, **local_settings}

def get_setting(section, key, default=None):
    """Получение конкретной настройки"""
    config = load_config()
    return config.get(section, {}).get(key, default)

def set_setting(section, key, value):
    """Установка конкретной настройки"""
    config = load_config()
    
    if section not in config:
        config[section] = {}
    
    config[section][key] = value
    
    return save_config(config)

def reset_to_defaults():
    """Сброс конфигурации к значениям по умолчанию"""
    try:
        if CONFIG_FILE.exists():
            # Создаем backup текущего конфига
            backup_file = CONFIG_FILE.parent / f"api_config_backup_{int(time.time())}.json"
            import shutil
            shutil.copy2(CONFIG_FILE, backup_file)
            print(f"✅ Создан backup текущей конфигурации: {backup_file.name}")
        
        # Сохраняем конфигурацию по умолчанию
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
        
        print("✅ Конфигурация сброшена к значениям по умолчанию")
        return True
    except Exception as e:
        print(f"❌ Ошибка сброса конфигурации: {e}")
        return False

def validate_config():
    """Проверка валидности конфигурации"""
    config = load_config()
    errors = []
    
    # Проверяем обязательные поля
    required_sections = ['llm', 'openrouter', 'llm_query_mode']
    for section in required_sections:
        if section not in config:
            errors.append(f"Отсутствует секция: {section}")
    
    # Проверяем режим LLM
    llm_mode = config.get('llm_query_mode', {}).get('default_mode')
    if llm_mode not in ['traditional', 'llm_first']:
        errors.append(f"Неверный режим LLM: {llm_mode}")
    
    # Проверяем приоритет LLM
    llm_priority = config.get('llm', {}).get('priority')
    if llm_priority and llm_priority not in ['local_first', 'openrouter_first', 'local_only', 'openrouter_only']:
        errors.append(f"Неверный приоритет LLM: {llm_priority}")
    
    # Проверяем настройки диалога
    dialogue_settings = config.get('dialogue_settings', {})
    if not isinstance(dialogue_settings.get('context_window', 0), int):
        errors.append("context_window должен быть числом")
    
    if errors:
        print("❌ Ошибки в конфигурации:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ Конфигурация валидна")
        return True

def export_config(filename=None):
    """Экспорт конфигурации в файл"""
    config = load_config()
    export_file = filename or f"ai_teacher_config_export_{int(time.time())}.json"
    
    try:
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"✅ Конфигурация экспортирована в: {export_file}")
        return True
    except Exception as e:
        print(f"❌ Ошибка экспорта конфигурации: {e}")
        return False

def import_config(filename):
    """Импорт конфигурации из файла"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            imported_config = json.load(f)
        
        # Валидируем импортированную конфигурацию
        if not isinstance(imported_config, dict):
            print("❌ Неверный формат конфигурации")
            return False
        
        # Сохраняем импортированную конфигурацию
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(imported_config, f, ensure_ascii=False, indent=2)
        
        print("✅ Конфигурация успешно импортирована")
        return True
    except Exception as e:
        print(f"❌ Ошибка импорта конфигурации: {e}")
        return False

# Загружаем конфигурацию при импорте
LLM_CONFIG = load_config().get("llm", DEFAULT_CONFIG["llm"])
OPENROUTER_CONFIG = load_config().get("openrouter", DEFAULT_CONFIG["openrouter"])
LOCAL_LLM_CONFIG = get_local_llm_settings()
FALLBACK_CONFIG = load_config().get("fallback", DEFAULT_CONFIG["fallback"])
DIALOGUE_SETTINGS = get_dialogue_settings()
KNOWLEDGE_SETTINGS = get_knowledge_settings()
AUDIO_SETTINGS = get_audio_settings()
ANIMATION_SETTINGS = get_animation_settings()
CACHE_SETTINGS = get_cache_settings()
LOGGING_SETTINGS = get_logging_settings()
PERFORMANCE_SETTINGS = get_performance_settings()
SECURITY_SETTINGS = get_security_settings()
UI_SETTINGS = get_ui_settings()

# Выводим информацию о конфигурации при загрузке
if __name__ == "__main__":
    print("=" * 50)
    print("AI Teacher Configuration Module")
    print("=" * 50)
    print(f"Config file: {CONFIG_FILE}")
    print(f"Config exists: {CONFIG_FILE.exists()}")
    print(f"LLM Mode: {get_llm_mode()}")
    print(f"LLM Priority: {get_llm_priority()}")
    print(f"OpenRouter API Key: {'Set' if get_api_key('openrouter') else 'Not set'}")
    print(f"Local LLM Enabled: {LOCAL_LLM_CONFIG.get('enabled', False)}")
    print("=" * 50)
    
    # Проверяем валидность конфигурации
    validate_config()
