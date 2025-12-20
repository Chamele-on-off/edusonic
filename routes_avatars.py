# routes_avatars.py
from core import (
    app, socketio, BASE_DIR, FRAMES_DIR, debug_log, teacher_required
)
from flask import jsonify, request, send_from_directory
from avatar_manager import AvatarManager
import os
import shutil
import json
from pathlib import Path

# Инициализация менеджера аватаров
avatar_manager = AvatarManager(FRAMES_DIR)

@app.route('/api/avatars/list', methods=['GET'])
@teacher_required
def get_avatars_list():
    """Получение списка доступных аватаров"""
    try:
        avatars = avatar_manager.list_avatars()
        return jsonify({
            "success": True,
            "avatars": avatars,
            "total": len(avatars)
        })
    except Exception as e:
        debug_log(f"❌ Ошибка получения списка аватаров: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/avatars/stats', methods=['GET'])
@teacher_required
def get_avatars_stats():
    """Получение статистики по аватаарам"""
    try:
        stats = avatar_manager.get_avatar_stats()
        return jsonify({
            "success": True,
            "stats": stats
        })
    except Exception as e:
        debug_log(f"❌ Ошибка получения статистики аватаров: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/avatars/upload', methods=['POST'])
@teacher_required
def upload_avatar():
    """Загрузка нового аватара (ZIP-файл с кадрами)"""
    try:
        # Проверка наличия файла
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "Файл не найден в запросе"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "Файл не выбран"}), 400
        
        # Проверка расширения
        if not file.filename.lower().endswith('.zip'):
            return jsonify({"success": False, "error": "Требуется ZIP-файл"}), 400
        
        # Дополнительные параметры
        avatar_name = request.form.get('name', '').strip()
        if not avatar_name:
            avatar_name = Path(file.filename).stem
        
        # Валидация имени аватара
        import re
        # Разрешаем только буквы, цифры, дефисы и подчеркивания
        if not re.match(r'^[\w\-]+$', avatar_name):
            return jsonify({"success": False, "error": "Недопустимые символы в имени аватара. Используйте буквы, цифры, дефисы и подчеркивания."}), 400
        
        if len(avatar_name) > 50:
            return jsonify({"success": False, "error": "Имя аватара слишком длинное (макс. 50 символов)"}), 400
        
        # Чтение содержимого файла в память
        file_content = file.read()
        if len(file_content) == 0:
            return jsonify({"success": False, "error": "Пустой файл"}), 400
        
        if len(file_content) > 50 * 1024 * 1024:  # 50 МБ
            return jsonify({"success": False, "error": "Файл слишком большой (макс. 50 МБ)"}), 400
        
        # Валидация и извлечение аватара
        validation_result = avatar_manager.validate_and_extract_avatar(
            file_content=file_content,
            avatar_name=avatar_name
        )
        
        if not validation_result['success']:
            return jsonify({"success": False, "error": validation_result.get('error', 'Неизвестная ошибка')})
        
        # Обновление списка аватаров через Socket.IO (для всех подключенных админов)
        updated_avatars = avatar_manager.list_avatars()
        socketio.emit('avatars_updated', {
            'avatars': updated_avatars,
            'message': f'Аватар "{avatar_name}" успешно загружен',
            'action': 'upload'
        }, namespace='/')
        
        return jsonify({
            "success": True,
            "message": f"Аватар '{avatar_name}' успешно загружен",
            "avatar_name": avatar_name,
            "frames_count": validation_result.get('frames_count', 0),
            "total_size": validation_result.get('total_size', 0)
        })
        
    except Exception as e:
        debug_log(f"❌ Ошибка загрузки аватара: {e}")
        return jsonify({"success": False, "error": f"Внутренняя ошибка сервера: {str(e)}"}), 500

@app.route('/api/avatars/delete', methods=['POST'])
@teacher_required
def delete_avatar():
    """Удаление аватара по имени"""
    try:
        data = request.get_json()
        avatar_name = data.get('avatar_name')
        
        if not avatar_name:
            return jsonify({"success": False, "error": "Не указано имя аватара"}), 400
        
        # Проверка, что аватар существует
        avatar_path = FRAMES_DIR / avatar_name
        if not avatar_path.exists() or not avatar_path.is_dir():
            return jsonify({"success": False, "error": "Аватар не найден"}), 404
        
        # Защита от удаления системных аватаров
        system_avatars = {'teacher', 'student', 'male', 'female'}
        if avatar_name in system_avatars:
            return jsonify({"success": False, "error": "Нельзя удалить системный аватар"}), 403
        
        # Удаление папки
        shutil.rmtree(avatar_path)
        
        # Обновление через Socket.IO
        updated_avatars = avatar_manager.list_avatars()
        socketio.emit('avatars_updated', {
            'avatars': updated_avatars,
            'message': f'Аватар "{avatar_name}" успешно удалён',
            'action': 'delete'
        }, namespace='/')
        
        return jsonify({
            "success": True,
            "message": f"Аватар '{avatar_name}' успешно удалён"
        })
        
    except Exception as e:
        debug_log(f"❌ Ошибка удаления аватара: {e}")
        return jsonify({"success": False, "error": f"Не удалось удалить аватар: {str(e)}"}), 500

@app.route('/api/avatars/bulk_delete', methods=['POST'])
@teacher_required
def bulk_delete_avatars():
    """Массовое удаление аватаров"""
    try:
        data = request.get_json()
        avatar_names = data.get('avatar_names', [])
        
        if not avatar_names:
            return jsonify({"success": False, "error": "Не указаны имена аватаров"}), 400
        
        system_avatars = {'teacher', 'student', 'male', 'female'}
        deleted = []
        errors = []
        
        for name in avatar_names:
            if name in system_avatars:
                errors.append(f"Нельзя удалить системный аватар: {name}")
                continue
            
            avatar_path = FRAMES_DIR / name
            if avatar_path.exists() and avatar_path.is_dir():
                try:
                    shutil.rmtree(avatar_path)
                    deleted.append(name)
                except Exception as e:
                    errors.append(f"Ошибка удаления {name}: {str(e)}")
            else:
                errors.append(f"Аватар не найден: {name}")
        
        # Обновление через Socket.IO
        updated_avatars = avatar_manager.list_avatars()
        socketio.emit('avatars_updated', {
            'avatars': updated_avatars,
            'message': f"Удалено {len(deleted)} аватаров",
            'action': 'bulk_delete'
        }, namespace='/')
        
        return jsonify({
            "success": True,
            "deleted": deleted,
            "errors": errors,
            "total_deleted": len(deleted)
        })
        
    except Exception as e:
        debug_log(f"❌ Ошибка массового удаления аватаров: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/avatars/preview/<avatar_name>')
@teacher_required
def preview_avatar_frame(avatar_name):
    """Просмотр случайного кадра аватара (для превью)"""
    try:
        avatar_path = FRAMES_DIR / avatar_name
        if not avatar_path.exists() or not avatar_path.is_dir():
            return jsonify({"success": False, "error": "Аватар не найден"}), 404
        
        # Ищем первый подходящий файл
        frame_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            frame_files.extend(avatar_path.glob(f"*{ext}"))
        
        if not frame_files:
            return jsonify({"success": False, "error": "Кадры не найдены"}), 404
        
        # Берём первый кадр для превью
        frame_file = sorted(frame_files)[0]
        return send_from_directory(avatar_path, frame_file.name)
        
    except Exception as e:
        debug_log(f"❌ Ошибка превью аватара: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/avatars/download/<avatar_name>')
@teacher_required
def download_avatar(avatar_name):
    """Скачивание аватара в виде ZIP-файла"""
    try:
        import tempfile
        import zipfile
        from datetime import datetime
        
        avatar_path = FRAMES_DIR / avatar_name
        if not avatar_path.exists() or not avatar_path.is_dir():
            return jsonify({"success": False, "error": "Аватар не найден"}), 404
        
        # Создаём временный ZIP
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
            for frame_file in avatar_path.iterdir():
                if frame_file.is_file():
                    zipf.write(frame_file, frame_file.name)
        
        temp_zip.close()
        
        return send_from_directory(
            os.path.dirname(temp_zip.name),
            os.path.basename(temp_zip.name),
            as_attachment=True,
            download_name=f"{avatar_name}_{datetime.now().strftime('%Y%m%d')}.zip"
        )
        
    except Exception as e:
        debug_log(f"❌ Ошибка скачивания аватара: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ==============================
# 🖼️ Поддержка отдачи кадров (если нужно вручную)
# ==============================

@app.route('/static/frames/<path:filename>')
def serve_frame(filename):
    """Явная отдача кадров аватара (резервный роут)"""
    try:
        return send_from_directory(FRAMES_DIR, filename)
    except Exception as e:
        debug_log(f"❌ Ошибка отдачи кадра: {e}")
        return "Frame not found", 404

debug_log("✅ Роуты аватаров зарегистрированы")
