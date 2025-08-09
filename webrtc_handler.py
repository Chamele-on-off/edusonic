import os
import logging
from aiortc import RTCPeerConnection, RTCSessionDescription
from flask_socketio import emit
from concurrent.futures import ThreadPoolExecutor

class WebRTCHandler:
    def __init__(self, socketio):
        self.socketio = socketio
        self.pcs = set()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)

    def handle_offer(self, offer: dict, room: str):
        """Обработка WebRTC оффера в отдельном потоке"""
        self._executor.submit(self._async_handle_offer, offer, room)

    def _async_handle_offer(self, offer: dict, room: str):
        """Асинхронная обработка оффера"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._process_offer(offer, room))
        loop.close()

    async def _process_offer(self, offer: dict, room: str):
        """Основная логика обработки оффера"""
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            if pc.iceConnectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        try:
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
            )
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            self.socketio.emit('webrtc_answer', {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            }, room=room)
            
            self.logger.info(f"WebRTC соединение установлено для комнаты {room}")

        except Exception as e:
            self.logger.error(f"WebRTC ошибка: {str(e)}")
            self.socketio.emit('webrtc_error', {
                "error": str(e)
            }, room=room)

    async def cleanup(self):
        """Очистка соединений"""
        for pc in self.pcs:
            await pc.close()
        self.pcs.clear()
        self.logger.info("Все WebRTC соединения закрыты")

    def stop(self):
        """Остановка обработчика"""
        self._executor.shutdown()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.cleanup())
        loop.close()