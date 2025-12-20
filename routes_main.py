# routes_main.py
from core import app, socketio, BASE_DIR, debug_log, teacher_required
from flask import render_template, send_from_directory, request, redirect, url_for, jsonify, session
import os
import secrets
import json
from pathlib import Path

# ==============================
# 🏠 Основные страницы
# ==============================

@app.route('/')
def index():
    """Главная страница — выбор режима: ученик или учитель"""
    debug_log("🏠 Запрошена главная страница")
    return render_template('index.html')

@app.route('/room/<room_id>')
def room(room_id):
    """Страница комнаты для ученика (до активации учителя)"""
    if not room_id or len(room_id) < 5:
        debug_log("❌ Некорректный ID комнаты")
        return "Invalid room ID", 400
    debug_log(f"🚪 Открыта комната: {room_id}")
    return render_template('room.html', room_id=room_id)

@app.route('/conference/<room_id>')
def conference(room_id):
    """Основная страница конференции с AI-учителем"""
    if not room_id or len(room_id) < 5:
        debug_log("❌ Некорректный ID комнаты для конференции")
        return "Invalid room ID", 400
    debug_log(f"🎥 Открыта конференция: {room_id}")
    return render_template('conference.html', room_id=room_id)

@app.route('/teacher')
@teacher_required
def teacher_panel():
    """Панель учителя (админка)"""
    debug_log("👨‍🏫 Открыта панель учителя")
    return render_template('teacher.html')

# ==============================
# 📁 Отдача статических файлов (для поддержки старых ссылок)
# ==============================

@app.route('/static/frames/<path:filename>')
def custom_avatars(filename):
    """Явная отдача аватаров (на случай, если Flask не обрабатывает static/frames)"""
    frames_dir = BASE_DIR / 'static' / 'frames'
    return send_from_directory(frames_dir, filename)

# ==============================
# 🔐 Управление сессией (для демонстрации; замените на свою авторизацию)
# ==============================

@app.route('/api/login/teacher', methods=['POST'])
def login_teacher():
    """Вход в панель учителя (временная реализация)"""
    data = request.get_json()
    password = data.get('password', '')
    # ⚠️ Замените на реальную проверку!
    if password == 'admin123':  # ← ЗАМЕНИТЕ НА СЕКРЕТ ИЛИ БД
        session['is_teacher'] = True
        debug_log("✅ Успешный вход в панель учителя")
        return jsonify({"success": True})
    else:
        debug_log("❌ Неверный пароль учителя")
        return jsonify({"success": False, "error": "Неверный пароль"}), 403

@app.route('/api/logout/teacher', methods=['POST'])
def logout_teacher():
    """Выход из панели учителя"""
    session.pop('is_teacher', None)
    debug_log("🚪 Выход из панели учителя")
    return jsonify({"success": True})

# ==============================
# 🧪 Вспомогательные эндпоинты (для отладки)
# ==============================

@app.route('/health')
def health_check():
    """Простая проверка работоспособности сервера"""
    return jsonify({
        "status": "ok",
        "timestamp": "live",
        "rooms_active": len([r for r, a in app.config.get('room_ai_activated', {}).items() if a])
    })

# ==============================
# 📜 Дополнительные страницы (если используются)
# ==============================

@app.route('/privacy')
def privacy_policy():
    return render_template('privacy.html')

@app.route('/terms')
def terms_of_use():
    return render_template('terms.html')

# ==============================
# ⚙️ Обработка ошибок
# ==============================

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    debug_log(f"🔥 Внутренняя ошибка: {str(e)}")
    return render_template('500.html'), 500

# ==============================
# 🔒 Расширение для декоратора teacher_required (если нужно)
# ==============================

# Уже определён в core.py, но если вы захотите его модифицировать — сделайте это там.
# Здесь он используется, но не переопределяется.

debug_log("✅ Роуты main зарегистрированы")
