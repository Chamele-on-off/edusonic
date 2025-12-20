# app.py
"""
Точка входа для AI Teacher System.
Модульная архитектура: все компоненты разнесены по файлам.
"""
from core import app, socketio

# Импортируем все модули для регистрации роутов и сокетов
import routes_main
import routes_lessons
import routes_students
import routes_avatars
import sockets

if __name__ == '__main__':
    print("🚀 Запуск AI Teacher System...")
    print("🌐 Доступно на http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
