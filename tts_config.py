# tts_config.py - Конфигурация для TTS сервиса
import os
from pathlib import Path

# Настройки TTS сервиса
TTS_CONFIG = {
    'enabled': True,
    'primary': 'gtts',  # zindaki или gtts
    'zindaki': {
        'base_url': os.environ.get('TTS_SERVICE_URL', 'https://tts.zindaki-edu.ru'),
        'timeout': 30,
        'retries': 2,
        'language_mapping': {
            'ru': 'ru',
            'en': 'en',
            'auto': 'ru'
        },
        'speaker_mapping': {
            'female': 'baya',
            'male': 'aidar',
            'teacher': 'baya',
            'default': 'baya'
        }
    },
    'gtts': {
        'fallback': True,
        'timeout': 10
    }
}

def get_tts_config():
    """Получение конфигурации TTS"""
    return TTS_CONFIG.copy()

def update_tts_config(key, value):
    """Обновление конфигурации TTS"""
    keys = key.split('.')
    config = TTS_CONFIG
    
    for k in keys[:-1]:
        if k in config:
            config = config[k]
        else:
            return False
    
    if keys[-1] in config:
        config[keys[-1]] = value
        return True
    
    return False
