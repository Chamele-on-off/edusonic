#!/usr/bin/env python3
"""
ZINDAKI TTS SERVICE - Полная исправленная версия с правильной сигнатурой apply_tts
"""

import os
import sys
import uuid
import torch
import torchaudio
import omegaconf
import io
import tempfile
import atexit
import shutil
import threading
import time
from datetime import datetime

# ВАЖНО: Устанавливаем переменные окружения ПЕРВЫМ ДЕЛОМ
os.environ['TORCH_HOME'] = '/app/cache'
os.environ['HF_HOME'] = '/app/cache'
os.environ['XDG_CACHE_HOME'] = '/app/cache'

# Создаем директории кэша
os.makedirs('/app/cache', exist_ok=True)
os.makedirs('/app/cache/torch/hub', exist_ok=True)

# Импорты Flask
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from pydantic import BaseModel, ValidationError

# Импорты Redis и очередей
import redis
from rq import Queue
from rq.job import Job

# ========== ИНИЦИАЛИЗАЦИЯ REDIS ==========
redis_conn = redis.Redis(
    host=os.getenv('REDIS_HOST', 'tts-redis'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=1,
    socket_connect_timeout=10,
    socket_timeout=30,
    retry_on_timeout=True,
    decode_responses=False
)

# Создаем очередь задач
q = Queue(connection=redis_conn, default_timeout=600)

# ========== ПРАВИЛЬНЫЕ ИМЕНА ДИКТОРОВ SILERO ==========
CORRECT_SPEAKERS = {
    'ru': {
        'baya': 'baya_16khz',
        'kseniya': 'kseniya_16khz',
        'xenia': 'kseniya_16khz',
        'aidar': 'aidar_16khz',
        'irina': 'irina_16khz',
        'natasha': 'natasha_16khz',
        'ruslan': 'ruslan_16khz',
    },
    'en': {
        'en_1': 'lj_16khz',
        'en_3': 'lj_16khz',
    }
}

# ========== ИНИЦИАЛИЗАЦИЯ FLASK ==========
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

# ========== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ==========
tts_models = {}
models_loading = False
models_loaded = False
startup_time = datetime.now()

# ========== МОДЕЛЬ ЗАПРОСА ==========
class TTSRequest(BaseModel):
    """Модель для валидации входящих запросов TTS"""
    text: str
    language: str = 'ru'
    speaker: str = 'baya'
    sample_rate: int = 16000
    put_accent: bool = True
    put_yo: bool = True
    
    class Config:
        extra = 'forbid'

# ========== ФУНКЦИЯ ЗАГРУЗКИ МОДЕЛЕЙ ==========
def load_all_models():
    """Загрузка всех женских моделей Silero TTS при старте сервиса"""
    global tts_models, models_loading, models_loaded
    
    models_loading = True
    
    print("\n" + "=" * 70)
    print("🚀 НАЧИНАЮ ЗАГРУЗКУ МОДЕЛЕЙ SILERO TTS")
    print("=" * 70)
    
    # Проверяем доступность torch
    print(f"PyTorch версия: {torch.__version__}")
    print(f"TorchAudio версия: {torchaudio.__version__}")
    print(f"Кэш директория: {os.environ.get('TORCH_HOME')}")
    print("=" * 70)
    
    # Женские голоса для загрузки (с правильными именами)
    female_voices = [
        ('ru', 'baya_16khz'),      # Русский женский голос
        ('ru', 'kseniya_16khz'),   # Русский женский голос
        ('en', 'lj_16khz'),        # Английский женский голос
    ]
    
    loaded_count = 0
    
    for language, correct_speaker in female_voices:
        # Создаем удобный ключ для хранения
        display_name = correct_speaker.replace('_16khz', '')
        model_key = f"{language}_{display_name}"
        
        try:
            print(f"\n📥 Загружаю модель: {language.upper()} - '{correct_speaker}'")
            
            # Устанавливаем директорию кэша
            torch.hub.set_dir('/app/cache/torch/hub')
            
            # Загружаем модель - возвращает кортеж из 5 элементов
            tts_result = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=language,
                speaker=correct_speaker,
                force_reload=False,
                verbose=False,
                trust_repo=True
            )
            
            print(f"   ✅ Получено {len(tts_result)} элементов в кортеже")
            
            # Распаковываем кортеж
            model = tts_result[0]           # Модель TTS
            symbols = tts_result[1]         # Алфавит/символы
            sample_rate = tts_result[2]     # Частота дискретизации
            example_text = tts_result[3]    # Пример текста
            apply_tts_func = tts_result[4]  # Функция apply_tts
            
            print(f"   📊 Sample rate: {sample_rate}")
            print(f"   📝 Пример текста: {example_text[:50]}...")
            print(f"   🔤 Символы: {symbols[:50]}...")
            
            # Перемещаем модель на CPU
            device = torch.device('cpu')
            model.to(device)
            
            # Тестируем генерацию с ПРАВИЛЬНЫМИ параметрами
            try:
                test_text = "Привет, это тестовая фраза." if language == 'ru' else "Hello, this is a test phrase."
                
                # ПРАВИЛЬНЫЙ вызов apply_tts!
                audio = apply_tts_func(
                    texts=[test_text],      # Должен быть список!
                    model=model,
                    sample_rate=sample_rate,
                    symbols=symbols,
                    device=device
                )
                
                test_passed = True
                print(f"   🔊 Тест генерации пройден")
                print(f"   ⏱️  Размер аудио: {audio.shape}")
            except Exception as test_error:
                print(f"   ⚠️ Тест генерации не удался: {test_error}")
                import traceback
                traceback.print_exc()
                test_passed = False
            
            # Сохраняем модель и все компоненты
            tts_models[model_key] = {
                'model': model,
                'symbols': symbols,
                'sample_rate': sample_rate,
                'example_text': example_text,
                'apply_tts_func': apply_tts_func,
                'correct_speaker': correct_speaker,
                'device': device,
                'tested': test_passed,
                'loaded_at': datetime.now().isoformat()
            }
            
            loaded_count += 1
            print(f"   🎯 Успешно загружено: {model_key}")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки {correct_speaker}: {str(e)}")
            import traceback
            print(f"Детали ошибки:")
            for line in traceback.format_exc().split('\n')[-10:]:
                if line.strip():
                    print(f"   {line}")
    
    models_loading = False
    models_loaded = True if loaded_count > 0 else False
    
    print("\n" + "=" * 70)
    print(f"🎯 ЗАГРУЗКА МОДЕЛЕЙ ЗАВЕРШЕНА")
    print(f"   Успешно загружено: {loaded_count} из {len(female_voices)} моделей")
    if loaded_count > 0:
        print(f"   Загруженные модели: {list(tts_models.keys())}")
    else:
        print(f"   ⚠️ Модели не загружены! Проверьте ошибки выше.")
    print("=" * 70)

# ========== ФУНКЦИЯ ПОЛУЧЕНИЯ МОДЕЛИ ==========
def get_model(language, user_speaker):
    """Получает модель по языку и имени диктора (преобразует в правильное имя)"""
    # Преобразуем имя диктора в правильный формат
    if language in CORRECT_SPEAKERS and user_speaker in CORRECT_SPEAKERS[language]:
        correct_speaker = CORRECT_SPEAKERS[language][user_speaker]
    else:
        # По умолчанию используем baya_16khz для русского и lj_16khz для английского
        correct_speaker = 'baya_16khz' if language == 'ru' else 'lj_16khz'
    
    model_key = f"{language}_{user_speaker}"
    
    # Если модель еще не загружена
    if model_key not in tts_models:
        print(f"📥 Загружаю модель по требованию: {model_key} -> {correct_speaker}")
        
        try:
            torch.hub.set_dir('/app/cache/torch/hub')
            
            # Загружаем модель
            tts_result = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=language,
                speaker=correct_speaker,
                force_reload=False,
                trust_repo=True
            )
            
            # Распаковываем кортеж
            model = tts_result[0]
            symbols = tts_result[1]
            sample_rate = tts_result[2]
            example_text = tts_result[3]
            apply_tts_func = tts_result[4]
            
            # Перемещаем на CPU
            device = torch.device('cpu')
            model.to(device)
            
            tts_models[model_key] = {
                'model': model,
                'symbols': symbols,
                'sample_rate': sample_rate,
                'example_text': example_text,
                'apply_tts_func': apply_tts_func,
                'correct_speaker': correct_speaker,
                'device': device,
                'loaded_at': datetime.now().isoformat()
            }
            
            print(f"✅ Модель загружена: {model_key}")
            print(f"   Sample rate: {sample_rate}")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели {model_key}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return tts_models[model_key]

# ========== ФУНКЦИЯ ГЕНЕРАЦИИ АУДИО ==========
def generate_audio(text, language, user_speaker, sample_rate, put_accent=True, put_yo=True):
    """Генерация аудио из текста"""
    try:
        print(f"\n🎵 Начинаю генерацию аудио")
        print(f"   Текст: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        print(f"   Язык: {language}, Запрошенный голос: {user_speaker}")
        print(f"   Длина текста: {len(text)} символов")
        
        # Получаем модель с правильным именем диктора
        model_info = get_model(language, user_speaker)
        model = model_info['model']
        symbols = model_info['symbols']
        target_sample_rate = model_info['sample_rate']
        apply_tts_func = model_info['apply_tts_func']
        device = model_info['device']
        correct_speaker = model_info['correct_speaker']
        
        print(f"   🔊 Использую голос: {correct_speaker}")
        print(f"   🎚️  Частота дискретизации: {target_sample_rate}Hz")
        print(f"   ⚙️  Устройство: {device}")
        
        start_time = time.time()
        
        # Проверяем длину текста
        if len(text) > 1000:
            print(f"   ⚠️ Текст длинный, может занять время...")
        
        # Генерация аудио с ПРАВИЛЬНЫМИ параметрами
        # ВАЖНО: texts должен быть списком!
        audio = apply_tts_func(
            texts=[text],                # ДОЛЖЕН БЫТЬ СПИСКОМ!
            model=model,
            sample_rate=target_sample_rate,
            symbols=symbols,
            device=device
        )
        
        # Сохраняем во временный файл
        temp_dir = '/app/temp_audio'
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav', 
            delete=False,
            dir=temp_dir
        )
        
        torchaudio.save(
            temp_file.name, 
            audio.unsqueeze(0), 
            target_sample_rate,
            format='wav'
        )
        
        generation_time = time.time() - start_time
        audio_duration = audio.shape[1] / target_sample_rate
        
        print(f"✅ Аудио успешно сгенерировано!")
        print(f"   ⏱️  Время генерации: {generation_time:.2f} секунд")
        print(f"   🕒 Длительность аудио: {audio_duration:.2f} секунд")
        print(f"   📁 Файл: {temp_file.name}")
        print(f"   📊 Размер файла: {os.path.getsize(temp_file.name) / 1024:.1f} KB")
        
        return temp_file.name
        
    except Exception as e:
        print(f"❌ Ошибка генерации аудио: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# ========== FLASK ROUTES ==========
@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/api/tts', methods=['POST'])
def tts_request():
    """Endpoint для озвучки текста"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided', 'status': 'error'}), 400
        
        req = TTSRequest(**data)
        
        # Автоматически устанавливаем sample_rate из модели
        if req.sample_rate != 16000:
            print(f"⚠️ Изменяю sample_rate с {req.sample_rate} на 16000 для совместимости")
            req.sample_rate = 16000
        
        # Проверяем поддержку языка
        if req.language not in ['ru', 'en']:
            return jsonify({
                'error': f'Unsupported language: {req.language}',
                'supported_languages': ['ru', 'en'],
                'status': 'error'
            }), 400
        
        # Проверяем длину текста
        if len(req.text) == 0:
            return jsonify({'error': 'Text cannot be empty', 'status': 'error'}), 400
        
        if len(req.text) > 5000:
            return jsonify({
                'error': f'Text too long ({len(req.text)} characters). Maximum is 5000.',
                'status': 'error'
            }), 400
        
        print(f"\n📨 Получен TTS запрос:")
        print(f"   🌐 Язык: {req.language}")
        print(f"   🗣️  Голос: {req.speaker}")
        print(f"   📝 Длина текста: {len(req.text)} символов")
        
        # Создаем задачу в очереди
        job = q.enqueue(
            generate_audio,
            args=(req.text, req.language, req.speaker, req.sample_rate, req.put_accent, req.put_yo),
            job_timeout=300,  # 5 минут таймаут
            result_ttl=3600,  # Хранить результат 1 час
            failure_ttl=1800  # Хранить информацию о неудачных задачах 30 минут
        )
        
        return jsonify({
            'job_id': job.get_id(),
            'status': 'queued',
            'message': 'Задача добавлена в очередь обработки',
            'estimated_time': '5-30 секунд в зависимости от длины текста',
            'models_available': list(tts_models.keys()),
            'speaker_mapping': CORRECT_SPEAKERS[req.language] if req.language in CORRECT_SPEAKERS else {},
            'timestamp': datetime.now().isoformat()
        }), 202
        
    except ValidationError as e:
        print(f"❌ Ошибка валидации: {e}")
        return jsonify({
            'error': 'Invalid request data',
            'details': e.errors(),
            'status': 'validation_error'
        }), 400
        
    except Exception as e:
        print(f"❌ Ошибка в tts_request: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'status': 'error'
        }), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Проверка статуса задачи и получение результата"""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        
        if job.is_finished:
            result = job.result
            if result is None:
                return jsonify({
                    'error': 'Job result is empty',
                    'status': 'error'
                }), 500
            
            # Проверяем, что результат - это путь к файлу
            if isinstance(result, str) and os.path.exists(result):
                try:
                    # Отправляем аудио файл
                    response = send_file(
                        result,
                        mimetype='audio/wav',
                        as_attachment=True,
                        download_name=f'tts_{job_id}.wav'
                    )
                    
                    # Удаляем файл после отправки
                    @response.call_on_close
                    def cleanup_file():
                        try:
                            if os.path.exists(result):
                                os.remove(result)
                                print(f"🗑️ Удален временный файл: {result}")
                        except Exception as e:
                            print(f"⚠️ Ошибка удаления файла {result}: {e}")
                    
                    return response
                except Exception as e:
                    print(f"❌ Ошибка отправки файла: {str(e)}")
                    return jsonify({
                        'error': 'Error sending audio file',
                        'details': str(e),
                        'status': 'error'
                    }), 500
            else:
                return jsonify({
                    'error': 'Invalid job result format',
                    'status': 'error'
                }), 500
                
        elif job.is_failed:
            error_msg = str(job.exc_info) if job.exc_info else 'Unknown error'
            print(f"❌ Задача {job_id} завершилась с ошибкой: {error_msg}")
            return jsonify({
                'error': 'Job failed',
                'details': error_msg,
                'status': 'failed'
            }), 500
            
        else:
            # Задача все еще выполняется или в очереди
            position = job.get_position() if hasattr(job, 'get_position') else 'unknown'
            return jsonify({
                'status': job.get_status(),
                'position': position,
                'job_id': job_id,
                'models_loaded': list(tts_models.keys()),
                'queue_length': len(q),
                'timestamp': datetime.now().isoformat()
            }), 200
            
    except Exception as e:
        print(f"❌ Ошибка в get_status для {job_id}: {str(e)}")
        return jsonify({
            'error': f'Job not found: {str(e)}',
            'status': 'not_found'
        }), 404

@app.route('/api/voices', methods=['GET'])
def get_available_voices():
    """Возвращает список доступных голосов и их статус"""
    
    # Полный список голосов (женские)
    all_voices = {
        'ru': [
            {'id': 'baya', 'name': 'Байя', 'actual': 'baya_16khz', 'gender': 'female', 'sample_rate': 16000},
            {'id': 'kseniya', 'name': 'Ксения', 'actual': 'kseniya_16khz', 'gender': 'female', 'sample_rate': 16000},
            {'id': 'aidar', 'name': 'Айдар', 'actual': 'aidar_16khz', 'gender': 'male', 'sample_rate': 16000},
            {'id': 'irina', 'name': 'Ирина', 'actual': 'irina_16khz', 'gender': 'female', 'sample_rate': 16000},
        ],
        'en': [
            {'id': 'en_1', 'name': 'English Female', 'actual': 'lj_16khz', 'gender': 'female', 'sample_rate': 16000},
        ]
    }
    
    # Фильтруем только загруженные голоса
    loaded_voices = {}
    for lang, voices in all_voices.items():
        loaded_voices[lang] = [
            voice for voice in voices 
            if f"{lang}_{voice['id']}" in tts_models
        ]
    
    return jsonify({
        'all_voices': all_voices,
        'loaded_voices': loaded_voices,
        'total_loaded': len(tts_models),
        'models_loading': models_loading,
        'service_status': 'ready' if models_loaded else 'loading',
        'speaker_mapping': CORRECT_SPEAKERS,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса и состояния системы"""
    try:
        # Проверяем соединение с Redis
        redis_status = 'connected' if redis_conn.ping() else 'disconnected'
        
        # Собираем системную информацию
        system_info = {
            'service': 'zindaki-tts-female-corrected',
            'status': 'healthy',
            'redis': redis_status,
            'models_loaded': list(tts_models.keys()),
            'models_loading': models_loading,
            'models_loaded_count': len(tts_models),
            'queue_size': len(q),
            'torch_version': torch.__version__,
            'torch_available': torch.cuda.is_available() if hasattr(torch.cuda, 'is_available') else False,
            'torchaudio_version': torchaudio.__version__,
            'python_version': sys.version.split()[0],
            'cache_dir': os.environ.get('TORCH_HOME'),
            'uptime': str(datetime.now() - startup_time),
            'startup_time': startup_time.isoformat(),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(system_info), 200
        
    except Exception as e:
        print(f"❌ Ошибка health check: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'models_loaded': list(tts_models.keys()),
            'torch_version': torch.__version__ if 'torch' in globals() else 'not loaded',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/load-models', methods=['POST'])
def force_load_models_endpoint():
    """Принудительная загрузка всех моделей"""
    if models_loading:
        return jsonify({
            'message': 'Models are already loading',
            'status': 'loading',
            'existing_models': list(tts_models.keys())
        }), 200
    
    # Запускаем загрузку в отдельном потоке
    thread = threading.Thread(target=load_all_models)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Model loading started in background',
        'loading': True,
        'existing_models': list(tts_models.keys()),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test-voice/<language>/<speaker>', methods=['GET'])
def test_voice(language, speaker):
    """Тестирование конкретного голоса"""
    try:
        model_key = f"{language}_{speaker}"
        
        if model_key not in tts_models:
            return jsonify({
                'error': f'Voice {speaker} for language {language} not loaded',
                'status': 'not_found'
            }), 404
        
        test_text = "Привет, это тестовое сообщение для проверки работы TTS сервиса." if language == 'ru' else "Hello, this is a test message to verify TTS service operation."
        
        job = q.enqueue(
            generate_audio,
            args=(test_text, language, speaker, 16000, True, True),
            job_timeout=60,
            result_ttl=300
        )
        
        return jsonify({
            'job_id': job.get_id(),
            'message': f'Test audio generation started for {speaker} ({language})',
            'test_text': test_text,
            'status': 'queued'
        }), 202
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
def cleanup_temp_files():
    """Очистка временных файлов при завершении работы"""
    temp_dir = '/app/temp_audio'
    if os.path.exists(temp_dir):
        try:
            deleted_count = 0
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Ошибка удаления {file_path}: {e}")
            
            if deleted_count > 0:
                print(f"🗑️ Удалено {deleted_count} временных файлов из {temp_dir}")
        except Exception as e:
            print(f"❌ Ошибка очистки временной директории: {e}")

# ========== ЗАПУСК ПРИЛОЖЕНИЯ ==========
if __name__ == '__main__':
    # Создаем необходимые директории
    os.makedirs('/app/temp_audio', exist_ok=True)
    os.makedirs('/app/cache', exist_ok=True)
    os.makedirs('/app/cache/torch/hub', exist_ok=True)
    
    # Регистрируем очистку при завершении
    atexit.register(cleanup_temp_files)
    
    print("\n" + "=" * 70)
    print("🎵 ZINDAKI TTS SERVICE - ИСПРАВЛЕННАЯ ВЕРСИЯ")
    print("=" * 70)
    print(f"📅 Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python версия: {sys.version.split()[0]}")
    print(f"🔥 PyTorch версия: {torch.__version__}")
    print(f"🎵 TorchAudio версия: {torchaudio.__version__}")
    print(f"📁 Кэш директория: {os.environ.get('TORCH_HOME')}")
    print(f"🔗 Redis хост: {os.getenv('REDIS_HOST', 'tts-redis')}:{os.getenv('REDIS_PORT', 6379)}")
    print("=" * 70)
    
    # Загружаем модели СРАЗУ в основном потоке
    print("\n⏳ Загружаю модели Silero TTS...")
    load_all_models()
    
    # Даем немного времени на инициализацию
    if len(tts_models) > 0:
        print(f"\n✅ Сервис готов! Загружено {len(tts_models)} моделей.")
    else:
        print(f"\n⚠️ Сервис запущен, но модели не загружены!")
        print(f"   Проверьте логи выше для диагностики ошибок.")
    
    # Запускаем Flask приложение
    print("\n🚀 Запускаю Flask сервер...")
    print(f"🌐 Сервер доступен по адресу: http://0.0.0.0:5000")
    print(f"🔧 Режим отладки: {'ВКЛЮЧЕН' if app.debug else 'ВЫКЛЮЧЕН'}")
    print("=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )