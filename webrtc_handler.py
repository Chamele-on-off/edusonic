import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from flask_socketio import emit

class WebRTCHandler:
    """Обработчик WebRTC соединений для видеоконференций и аудио транскрипции"""
    
    def __init__(self, socketio):
        """
        Инициализация обработчика WebRTC
        
        :param socketio: Экземпляр SocketIO для отправки событий клиентам
        """
        self.socketio = socketio
        self.pcs: Dict[str, RTCPeerConnection] = {}  # {room_id: peer_connection}
        self.audio_processors = {}  # {room_id: audio_processor}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)
        
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def handle_offer(self, offer: dict, room: str, sid: str):
        """
        Обработка входящего WebRTC оффера
        
        :param offer: WebRTC оффер от клиента
        :param room: ID комнаты
        :param sid: ID сокета
        """
        self.executor.submit(self._async_handle_offer, offer, room, sid)

    def _async_handle_offer(self, offer: dict, room: str, sid: str):
        """
        Асинхронная обработка оффера (запускается в отдельном потоке)
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._process_offer(offer, room, sid))
        except Exception as e:
            self.logger.error(f"Ошибка обработки оффера: {str(e)}")
            self.socketio.emit('webrtc_error', {
                "error": str(e),
                "room": room
            }, room=sid)
        finally:
            loop.close()

    async def _process_offer(self, offer: dict, room: str, sid: str):
        """
        Основная логика обработки WebRTC оффера
        
        :param offer: WebRTC оффер {type: 'offer', sdp: '...'}
        :param room: ID комнаты
        :param sid: ID сокета
        """
        # Создаем новое соединение PeerConnection
        pc = RTCPeerConnection()
        self.pcs[room] = pc
        
        # Обработчик изменения состояния ICE соединения
        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            state = pc.iceConnectionState
            self.logger.info(f"ICE соединение изменило состояние: {state}")
            if state == "failed":
                await self._cleanup_room(room)

        # Обработчик входящих медиапотоков
        @pc.on("track")
        def on_track(track: MediaStreamTrack):
            self.logger.info(f"Получен трек {track.kind} от {room}")
            
            if track.kind == "audio":
                # Создаем обработчик аудио для транскрипции
                from .audio_processor import AudioProcessor
                self.audio_processors[room] = AudioProcessor(
                    track, 
                    room, 
                    self.socketio, 
                    sid
                )
                self.audio_processors[room].start()

        try:
            # Устанавливаем удаленное описание (оффер от клиента)
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
            )
            
            # Создаем ответ (answer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            # Отправляем ответ клиенту
            self.socketio.emit('webrtc_answer', {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "room": room
            }, room=sid)
            
            self.logger.info(f"WebRTC соединение установлено для комнаты {room}")

        except Exception as e:
            self.logger.error(f"Ошибка установки соединения: {str(e)}")
            await self._cleanup_room(room)
            raise

    async def _cleanup_room(self, room: str):
        """Очистка ресурсов для указанной комнаты"""
        if room in self.pcs:
            pc = self.pcs[room]
            await pc.close()
            del self.pcs[room]
            self.logger.info(f"Соединение для комнаты {room} закрыто")
        
        if room in self.audio_processors:
            await self.audio_processors[room].stop()
            del self.audio_processors[room]
            self.logger.info(f"Аудио обработчик для комнаты {room} остановлен")

    async def cleanup_all(self):
        """Очистка всех соединений и ресурсов"""
        for room in list(self.pcs.keys()):
            await self._cleanup_room(room)
        self.logger.info("Все WebRTC соединения закрыты")

    def stop(self):
        """Остановка обработчика"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.cleanup_all())
        loop.close()
        self.executor.shutdown()
        self.logger.info("WebRTCHandler остановлен")


class AudioProcessor:
    """Обработчик аудиопотока для транскрипции речи"""
    
    def __init__(self, track: MediaStreamTrack, room: str, socketio, sid: str):
        """
        Инициализация обработчика аудио
        
        :param track: Аудио трек
        :param room: ID комнаты
        :param socketio: Экземпляр SocketIO
        :param sid: ID сокета
        """
        self.track = track
        self.room = room
        self.socketio = socketio
        self.sid = sid
        self._running = False
        self.logger = logging.getLogger(__name__)
        
        # Буфер для накопления аудиоданных
        self.audio_buffer = bytearray()
        self.sample_rate = 16000
        self.sample_width = 2  # 16-bit
        self.channels = 1

    async def start(self):
        """Запуск обработки аудиопотока"""
        self._running = True
        asyncio.create_task(self._process_audio())

    async def stop(self):
        """Остановка обработки аудиопотока"""
        self._running = False

    async def _process_audio(self):
        """Основной цикл обработки аудио"""
        self.logger.info(f"Начата обработка аудио для комнаты {self.room}")
        
        try:
            while self._running:
                frame = await self.track.recv()
                
                # Добавляем аудиоданные в буфер
                self.audio_buffer.extend(frame.to_ndarray().tobytes())
                
                # Если накопилось достаточно данных - отправляем на транскрипцию
                if len(self.audio_buffer) >= self.sample_rate * self.sample_width * 0.5:  # 0.5 секунды
                    await self._transcribe_audio()
                    
        except Exception as e:
            self.logger.error(f"Ошибка обработки аудио: {str(e)}")
        finally:
            self.logger.info(f"Остановлена обработка аудио для комнаты {self.room}")

    async def _transcribe_audio(self):
        """Транскрипция накопленного аудио"""
        try:
            # Используем SpeechRecognition для транскрипции
            from speech_recognition import AudioData, Recognizer
            
            # Создаем временный файл WAV
            import wave
            with BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(self.sample_width)
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(self.audio_buffer)
                
                wav_data = wav_buffer.getvalue()
            
            # Конвертируем в base64 для передачи через SocketIO
            audio_b64 = base64.b64encode(wav_data).decode('utf-8')
            
            # Отправляем аудио на сервер для обработки
            self.socketio.emit('audio_data', {
                "audio": audio_b64,
                "room": self.room
            }, room=self.sid)
            
            # Очищаем буфер
            self.audio_buffer.clear()
            
        except Exception as e:
            self.logger.error(f"Ошибка транскрипции: {str(e)}")
            self.audio_buffer.clear()


def create_webrtc_handler(socketio) -> WebRTCHandler:
    """Фабрика для создания экземпляра WebRTCHandler"""
    return WebRTCHandler(socketio)