# async_llm.py
import threading
import queue
import time
from typing import Dict, Callable, Optional
import json
from local_llm import LocalLLM
from llm import LLMIntegration

class AsyncLLMManager:
    """Менеджер асинхронных запросов к LLM"""
    
    def __init__(self):
        self.request_queue = queue.Queue()
        self.response_queues: Dict[str, queue.Queue] = {}
        self.room_callbacks: Dict[str, Callable] = {}
        self.workers = []
        self.running = True
        
        # Создаем пул из 5 worker-потоков
        for i in range(5):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        print(f"🔧 AsyncLLMManager запущен с {len(self.workers)} потоками")
    
    def _worker_loop(self, worker_id: int):
        """Цикл обработки запросов в отдельном потоке"""
        local_llm = LocalLLM()
        llm = LLMIntegration()
        
        while self.running:
            try:
                # Получаем запрос из очереди
                try:
                    request_data = self.request_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                room_id = request_data['room_id']
                prompt = request_data['prompt']
                system_prompt = request_data['system_prompt']
                max_tokens = request_data['max_tokens']
                request_id = request_data['request_id']
                use_local = request_data.get('use_local', False)
                
                print(f"🧵 [Worker-{worker_id}] Обработка запроса {request_id} для комнаты {room_id}")
                
                # Выполняем запрос (не блокирует главный поток)
                if use_local and local_llm.is_available():
                    response = local_llm.generate(prompt, system_prompt, max_tokens)
                else:
                    response = llm.query(prompt, "", request_data.get('subject', ''))
                
                # Сохраняем ответ
                if room_id not in self.response_queues:
                    self.response_queues[room_id] = queue.Queue()
                
                self.response_queues[room_id].put({
                    'request_id': request_id,
                    'response': response,
                    'timestamp': time.time(),
                    'room_id': room_id
                })
                
                # Вызываем callback если есть
                if room_id in self.room_callbacks:
                    try:
                        self.room_callbacks[room_id](request_id, response, room_id)
                    except Exception as e:
                        print(f"❌ Ошибка callback: {e}")
                
                self.request_queue.task_done()
                
            except Exception as e:
                print(f"❌ Ошибка в worker-{worker_id}: {e}")
                time.sleep(0.1)
    
    def submit_request(self, 
                      prompt: str,
                      system_prompt: str = "",
                      max_tokens: int = 1000,
                      room_id: str = "default",
                      request_id: str = None,
                      use_local: bool = False,
                      subject: str = "") -> str:
        """Добавляет запрос в очередь"""
        if request_id is None:
            request_id = f"{room_id}_{int(time.time()*1000)}"
        
        request_data = {
            'request_id': request_id,
            'prompt': prompt,
            'system_prompt': system_prompt,
            'max_tokens': max_tokens,
            'room_id': room_id,
            'use_local': use_local,
            'subject': subject,
            'timestamp': time.time()
        }
        
        self.request_queue.put(request_data)
        print(f"📨 Запрос добавлен в очередь: {request_id}, очередь: {self.request_queue.qsize()}")
        return request_id
    
    def get_response(self, room_id: str, request_id: str = None, timeout: float = 0.1):
        """Получает ответ из очереди"""
        if room_id not in self.response_queues:
            return None
        
        try:
            # Проверяем все ответы в очереди
            for _ in range(self.response_queues[room_id].qsize()):
                response = self.response_queues[room_id].get_nowait()
                
                if request_id is None or response['request_id'] == request_id:
                    return response
                else:
                    # Возвращаем обратно в очередь
                    self.response_queues[room_id].put(response)
            
        except queue.Empty:
            pass
        
        return None
    
    def register_callback(self, room_id: str, callback: Callable):
        """Регистрирует callback для комнаты"""
        self.room_callbacks[room_id] = callback
        print(f"🔧 Зарегистрирован callback для комнаты {room_id}")
    
    def get_queue_stats(self) -> Dict:
        """Возвращает статистику очереди"""
        return {
            'queue_size': self.request_queue.qsize(),
            'active_workers': sum(1 for w in self.workers if w.is_alive()),
            'rooms_with_callbacks': len(self.room_callbacks),
            'response_queues': {k: v.qsize() for k, v in self.response_queues.items()}
        }

# Глобальный экземпляр
_async_llm_manager = None

def get_async_llm_manager() -> AsyncLLMManager:
    """Возвращает глобальный экземпляр менеджера"""
    global _async_llm_manager
    if _async_llm_manager is None:
        _async_llm_manager = AsyncLLMManager()
    return _async_llm_manager
