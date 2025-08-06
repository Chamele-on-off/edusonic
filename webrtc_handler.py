import os
import logging
from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import WebSocket

class WebRTCHandler:
    def __init__(self):
        self.pcs = set()
        self.logger = logging.getLogger(__name__)

    async def handle_offer(self, websocket: WebSocket, offer: dict):
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            if pc.iceConnectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
        )
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }

    async def cleanup(self):
        for pc in self.pcs:
            await pc.close()
        self.pcs.clear()
