import threading
import queue
import time
from typing import Dict, Optional
from local_llm import LocalLLM
import json

class LocalLLMManager:
    def __init__(self):
        self.local_llm = LocalLLM()
        self.request_queue = queue.Queue()
        self.response_queues: Dict[str, queue.Queue] = {}
        self.worker_thread = None
        self.running = False
        self.room_callbacks = {}
        
    def start(self):
        """Запуск менеджера в отдельном потоке"""
        if self.worker_thread and self.worker_thread.is_alive():
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("✅ LocalLLM Manager запущен в отдельном потоке")
        
    def stop(self):
        """Остановка менеджера"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
            
    def _worker_loop(self):
        """Основной цикл обработки запросов"""
        while self.running:
            try:
                # Неблокирующее получение запроса
                try:
                    request_id, prompt, system_prompt, max_tokens, room_id = self.request_queue.get(timeout=1)
                except queue.Empty:
                    continue
                    
                print(f"🔧 [LocalLLM Worker] Обработка запроса для комнаты {room_id}")
                
                # Генерация ответа
                response = self.local_llm.generate(prompt, system_prompt, max_tokens)
                
                # Отправка ответа через callback
                if room_id in self.room_callbacks:
                    try:
                        self.room_callbacks[room_id](request_id, response, room_id)
                    except Exception as e:
                        print(f"❌ Ошибка вызова callback для комнаты {room_id}: {e}")
                
                self.request_queue.task_done()
                
            except Exception as e:
                print(f"❌ Ошибка в LocalLLM worker: {e}")
                time.sleep(0.1)
                
    def submit_request(self, prompt: str, system_prompt: str = "", max_tokens: int = 1000, 
                      room_id: str = "default") -> str:
        """Добавление запроса в очередь"""
        request_id = f"{room_id}_{int(time.time()*1000)}"
        
        self.request_queue.put((request_id, prompt, system_prompt, max_tokens, room_id))
        print(f"📨 [LocalLLM] Запрос добавлен в очередь: {request_id}")
        
        return request_id
        
    def register_room_callback(self, room_id: str, callback):
        """Регистрация callback для комнаты"""
        self.room_callbacks[room_id] = callback
        print(f"🔧 [LocalLLM] Зарегистрирован callback для комнаты {room_id}")
        
    def unregister_room_callback(self, room_id: str):
        """Удаление callback для комнаты"""
        if room_id in self.room_callbacks:
            del self.room_callbacks[room_id]
            
    def get_queue_size(self) -> int:
        """Получение размера очереди"""
        return self.request_queue.qsize()
        
    def get_status(self) -> Dict:
        """Получение статуса менеджера"""
        return {
            "running": self.running,
            "queue_size": self.get_queue_size(),
            "worker_alive": self.worker_thread.is_alive() if self.worker_thread else False,
            "registered_rooms": list(self.room_callbacks.keys()),
            "llm_status": self.local_llm.get_status()
        }

# Глобальный экземпляр
llm_manager = LocalLLMManager()

def get_llm_manager() -> LocalLLMManager:
    return llm_manager
