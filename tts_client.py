# tts_client.py - Клиент для работы с TTS сервисом Zindaki
import requests
import base64
import io
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ZindakiTTSClient:
    """Клиент для работы с TTS сервисом Zindaki"""
    
    def __init__(self, base_url: str = "http://tts.zindaki-edu.ru"):
        """
        Инициализация клиента TTS
        
        Args:
            base_url: URL TTS сервиса (по умолчанию: tts.zindaki-edu.ru)
        """
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/tts"
        self.health_url = f"{self.base_url}/api/health"
        self.timeout = 30  # таймаут в секундах
        self.max_retries = 2
        self.available = self._check_availability()
        
        logger.info(f"Zindaki TTS Client initialized. Base URL: {base_url}")
        logger.info(f"TTS Service available: {self.available}")
    
    def _check_availability(self) -> bool:
        """Проверка доступности TTS сервиса"""
        try:
            response = requests.get(self.health_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('status') == 'healthy'
        except Exception as e:
            logger.warning(f"TTS service check failed: {e}")
        return False
    
    def generate_speech(self, text: str, language: str = 'ru', speaker: str = 'baya', 
                        sample_rate: int = 16000) -> Optional[bytes]:
        """
        Генерация речи через TTS сервис
        
        Args:
            text: Текст для озвучки
            language: Язык ('ru' или 'en')
            speaker: Голос ('baya', 'kseniya', 'aidar', etc.)
            sample_rate: Частота дискретизации
            
        Returns:
            Аудио данные в формате WAV или None при ошибке
        """
        if not self.available:
            logger.warning("TTS service not available")
            return None
        
        if not text.strip():
            logger.warning("Empty text provided")
            return None
        
        # Подготовка запроса
        payload = {
            'text': text,
            'language': language,
            'speaker': speaker,
            'sample_rate': sample_rate
        }
        
        logger.info(f"Generating TTS for text: '{text[:50]}...' (lang: {language}, speaker: {speaker})")
        
        # Попытка генерации с повторными попытками
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout
                )
                
                generation_time = time.time() - start_time
                
                if response.status_code == 200:
                    # Получаем аудио данные
                    audio_data = response.content
                    
                    # Проверяем, что это валидный WAV файл
                    if len(audio_data) > 44 and audio_data[:4] == b'RIFF':
                        cache_hit = response.headers.get('X-Cache-Hit', 'false') == 'true'
                        
                        logger.info(f"✅ TTS generated successfully in {generation_time:.2f}s "
                                   f"(Cache: {'HIT' if cache_hit else 'MISS'}, "
                                   f"Size: {len(audio_data)} bytes)")
                        
                        return audio_data
                    else:
                        logger.error(f"❌ Invalid audio data received (size: {len(audio_data)})")
                        
                elif response.status_code == 429:
                    # Сервис перегружен
                    wait_time = 2 ** attempt  # экспоненциальная задержка
                    logger.warning(f"TTS service busy (429). Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    error_text = response.text[:200] if response.text else "No error message"
                    logger.error(f"❌ TTS generation failed: HTTP {response.status_code} - {error_text}")
                    
                    # Если это 4xx ошибка (кроме 429) - не повторяем
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        break
                        
            except requests.exceptions.Timeout:
                logger.error(f"❌ TTS request timeout (attempt {attempt + 1}/{self.max_retries + 1})")
                if attempt < self.max_retries:
                    continue
                    
            except requests.exceptions.ConnectionError:
                logger.error(f"❌ TTS connection error (attempt {attempt + 1}/{self.max_retries + 1})")
                self.available = False  # Помечаем сервис как недоступный
                break
                
            except Exception as e:
                logger.error(f"❌ Unexpected TTS error: {e}")
                break
        
        return None
    
    def generate_speech_base64(self, text: str, language: str = 'ru', 
                               speaker: str = 'baya') -> Optional[str]:
        """
        Генерация речи и возврат в формате base64
        
        Returns:
            Base64 строка с аудио данными или None при ошибке
        """
        audio_data = self.generate_speech(text, language, speaker)
        
        if audio_data:
            return base64.b64encode(audio_data).decode('utf-8')
        return None
    
    def get_service_status(self) -> Dict[str, Any]:
        """Получение статуса TTS сервиса"""
        try:
            response = requests.get(self.health_url, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get TTS service status: {e}")
        
        return {'status': 'unavailable', 'available': False}
    
    def get_available_voices(self) -> Dict[str, Any]:
        """Получение списка доступных голосов"""
        try:
            response = requests.get(f"{self.base_url}/api/voices", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get available voices: {e}")
        
        return {'all_voices': {}, 'loaded_voices': {}}
    
    def clear_cache(self, days_old: Optional[int] = None) -> bool:
        """Очистка кэша TTS сервиса"""
        try:
            payload = {}
            if days_old:
                payload['days_old'] = days_old
            
            response = requests.post(
                f"{self.base_url}/api/cache/clear",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    logger.info(f"✅ TTS cache cleared: {data.get('message')}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to clear TTS cache: {e}")
        
        return False

# Создаем глобальный экземпляр клиента
_tts_client_instance = None

def get_tts_client() -> ZindakiTTSClient:
    """Получение или создание экземпляра TTS клиента"""
    global _tts_client_instance
    if _tts_client_instance is None:
        _tts_client_instance = ZindakiTTSClient()
    return _tts_client_instance
